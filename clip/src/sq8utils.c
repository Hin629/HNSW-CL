/*
 * sq8utils.c - SQ8 SIMD-accelerated distance functions
 * 
 * Implements scalar quantization distance computations with SIMD optimization.
 * References vsag's SQ8 implementation for the quantization approach.
 */
#include "postgres.h"

#include <math.h>
#include "sq8utils.h"

#if defined(__x86_64__)
#include <immintrin.h>
#if defined(HAVE__GET_CPUID)
#include <cpuid.h>
#endif
#endif

/* Function pointers */
float (*Sq8L2SquaredDistance)(int dim, const uint8 *ax, const uint8 *bx,
							   float lower_bound_a, float diff_a,
							   float lower_bound_b, float diff_b) = NULL;

float (*Sq8InnerProduct)(int dim, const uint8 *ax, const uint8 *bx,
						  float lower_bound_a, float diff_a,
						  float lower_bound_b, float diff_b) = NULL;

double (*Sq8CosineSimilarity)(int dim, const uint8 *ax, const uint8 *bx,
							   float lower_bound_a, float diff_a,
							   float lower_bound_b, float diff_b) = NULL;

float (*Sq8L1Distance)(int dim, const uint8 *ax, const uint8 *bx,
						float lower_bound_a, float diff_a,
						float lower_bound_b, float diff_b) = NULL;

int64 (*Sq8L2SquaredDistanceSameQuant)(int dim, const uint8 *ax, const uint8 *bx) = NULL;
int64 (*Sq8InnerProductSameQuant)(int dim, const uint8 *ax, const uint8 *bx) = NULL;

/* Batch implementations selected at init */
static void (*Sq8L2SquaredDistanceSameQuantBatchImpl)(int dim, const uint8 *query,
														  const uint8 **targets, int count,
														  int64 *results) = NULL;
static void (*Sq8InnerProductSameQuantBatchImpl)(int dim, const uint8 *query,
													 const uint8 **targets, int count,
													 int64 *results) = NULL;

/* Optimized L2 batch using sum2 - impl selected at init */
static void (*Sq8L2SquaredDistanceWithSum2BatchImpl)(int dim, const uint8 *query,
														 int32 query_sum, int32 query_sum2,
														 const uint8 **targets, const int32 *target_sum2s,
														 int count, int64 *results) = NULL;

/* Single-point L2 using sum2 - impl selected at init */
static int64 (*Sq8L2SquaredDistanceWithSum2SingleImpl)(int dim, const uint8 *ax,
													   int32 sum_a, int32 sum2_a,
													   const uint8 *bx, int32 sum2_b) = NULL;

/*
 * Generic (non-SIMD) implementations
 */

/*
 * Fast SQ8 math
 *
 * Let scale = diff / 255.
 * Decode: v = code * scale + lower_bound
 *
 * With integer accumulators:
 *   Sa  = sum(ax)
 *   Sb  = sum(bx)
 *   Sa2 = sum(ax^2)
 *   Sb2 = sum(bx^2)
 *   Sab = sum(ax*bx)
 *
 * We can compute exactly (up to float rounding) without per-element float decode:
 *   L2^2 = (sa^2)*Sa2 + (sb^2)*Sb2 - 2*sa*sb*Sab + 2*d*sa*Sa - 2*d*sb*Sb + dim*d^2
 *     where sa=scale_a, sb=scale_b, d=lb_a-lb_b
 *
 *   dot = sa*sb*Sab + sa*lb_b*Sa + sb*lb_a*Sb + dim*lb_a*lb_b
 *
 *   ||a||^2 = sa^2*Sa2 + 2*sa*lb_a*Sa + dim*lb_a^2
 */

typedef void (*Sq8AccumulateSumsFn)(int dim,
								 const uint8 *ax,
								 const uint8 *bx,
								 int64 *sa,
								 int64 *sb,
								 int64 *sa2,
								 int64 *sb2,
								 int64 *sab);

static Sq8AccumulateSumsFn Sq8AccumulateSums = NULL;

void
Sq8ReencodeToGlobalParams(const uint8 *local_codes, int dim,
							 float lower_bound_local, float diff_local,
							 float lower_bound_global, float diff_global,
							 uint8 *out_codes,
							 int32 *out_sum, int32 *out_sum2)
{
	int32 sum = 0;
	int32 sum2 = 0;
	double alpha;
	double beta;
	/* Guard against invalid diff_global */
	if (!(diff_global > 0.0f) || !isfinite(diff_global))
	{
		/* Fallback: produce zeros to keep behavior defined */
		if (out_sum)
			*out_sum = 0;
		if (out_sum2)
			*out_sum2 = 0;
		memset(out_codes, 0, dim);
		return;
	}

	/*
	 * code_g = round( (lb_l - lb_g) * 255/diff_g + (diff_l/diff_g) * code_l )
	 * Use double for stability.
	 */
	alpha = (double) diff_local / (double) diff_global;
	beta = ((double) lower_bound_local - (double) lower_bound_global) * 255.0 / (double) diff_global;

	for (int i = 0; i < dim; i++)
	{
		uint8 code;
		double codef = beta + alpha * (double) local_codes[i];
		if (codef < 0.0)
			codef = 0.0;
		else if (codef > 255.0)
			codef = 255.0;

		/* Round to nearest */
		code = (uint8) (codef + 0.5);
		out_codes[i] = code;
		sum += code;
		sum2 += (int32) code * (int32) code;
	}

	if (out_sum)
		*out_sum = sum;
	if (out_sum2)
		*out_sum2 = sum2;
}

static inline void
Sq8AccumulateSumsGeneric(int dim,
						const uint8 *ax,
						const uint8 *bx,
						int64 *sa,
						int64 *sb,
						int64 *sa2,
						int64 *sb2,
						int64 *sab)
{
	int64 tsa = 0;
	int64 tsb = 0;
	int64 tsa2 = 0;
	int64 tsb2 = 0;
	int64 tsab = 0;

	/* Auto-vectorized scalar loop */
	for (int i = 0; i < dim; i++)
	{
		uint32 a = ax[i];
		uint32 b = bx[i];
		tsa += a;
		tsb += b;
		tsa2 += (int64) a * (int64) a;
		tsb2 += (int64) b * (int64) b;
		tsab += (int64) a * (int64) b;
	}

	*sa = tsa;
	*sb = tsb;
	*sa2 = tsa2;
	*sb2 = tsb2;
	*sab = tsab;
}

#if defined(__x86_64__) && defined(__AVX2__)
__attribute__((target("avx2")))
static void
Sq8AccumulateSumsAVX2(int dim,
					const uint8 *ax,
					const uint8 *bx,
					int64 *sa,
					int64 *sb,
					int64 *sa2,
					int64 *sb2,
					int64 *sab)
{
	int i = 0;
	__m256i zero = _mm256_setzero_si256();

	/* Sum of bytes via SAD -> 4x uint64 lanes */
	__m256i sum_a64 = _mm256_setzero_si256();
	__m256i sum_b64 = _mm256_setzero_si256();

	/* Sum of squares / products via madd -> 8x int32 lanes */
	__m256i sum_a2_32 = _mm256_setzero_si256();
	__m256i sum_b2_32 = _mm256_setzero_si256();
	__m256i sum_ab_32 = _mm256_setzero_si256();

	for (; i + 32 <= dim; i += 32)
	{
		__m256i va;
		__m256i vb;
		__m256i va_lo;
		__m256i va_hi;
		__m256i vb_lo;
		__m256i vb_hi;
		
		va = _mm256_loadu_si256((const __m256i *) (ax + i));
		vb = _mm256_loadu_si256((const __m256i *) (bx + i));

		sum_a64 = _mm256_add_epi64(sum_a64, _mm256_sad_epu8(va, zero));
		sum_b64 = _mm256_add_epi64(sum_b64, _mm256_sad_epu8(vb, zero));

		va_lo = _mm256_unpacklo_epi8(va, zero);
		va_hi = _mm256_unpackhi_epi8(va, zero);
		vb_lo = _mm256_unpacklo_epi8(vb, zero);
		vb_hi = _mm256_unpackhi_epi8(vb, zero);

		sum_a2_32 = _mm256_add_epi32(sum_a2_32, _mm256_madd_epi16(va_lo, va_lo));
		sum_a2_32 = _mm256_add_epi32(sum_a2_32, _mm256_madd_epi16(va_hi, va_hi));
		sum_b2_32 = _mm256_add_epi32(sum_b2_32, _mm256_madd_epi16(vb_lo, vb_lo));
		sum_b2_32 = _mm256_add_epi32(sum_b2_32, _mm256_madd_epi16(vb_hi, vb_hi));
		sum_ab_32 = _mm256_add_epi32(sum_ab_32, _mm256_madd_epi16(va_lo, vb_lo));
		sum_ab_32 = _mm256_add_epi32(sum_ab_32, _mm256_madd_epi16(va_hi, vb_hi));
	}
	
	{
		/* Reduce sums */
		uint64 tmp64[4];
		uint32 tmp32[8];
		uint64 tsa;
		uint64 tsb;
		uint64 tsa2;
		uint64 tsb2;
		uint64 tsab;
		
		_mm256_storeu_si256((__m256i *) tmp64, sum_a64);
		tsa = tmp64[0] + tmp64[1] + tmp64[2] + tmp64[3];
		_mm256_storeu_si256((__m256i *) tmp64, sum_b64);
		tsb = tmp64[0] + tmp64[1] + tmp64[2] + tmp64[3];

		_mm256_storeu_si256((__m256i *) tmp32, sum_a2_32);
		tsa2 = (uint64) tmp32[0] + tmp32[1] + tmp32[2] + tmp32[3] + tmp32[4] + tmp32[5] + tmp32[6] + tmp32[7];
		_mm256_storeu_si256((__m256i *) tmp32, sum_b2_32);
		tsb2 = (uint64) tmp32[0] + tmp32[1] + tmp32[2] + tmp32[3] + tmp32[4] + tmp32[5] + tmp32[6] + tmp32[7];
		_mm256_storeu_si256((__m256i *) tmp32, sum_ab_32);
		tsab = (uint64) tmp32[0] + tmp32[1] + tmp32[2] + tmp32[3] + tmp32[4] + tmp32[5] + tmp32[6] + tmp32[7];

		/* Tail */
		for (; i < dim; i++)
		{
			uint32 a = ax[i];
			uint32 b = bx[i];
			tsa += a;
			tsb += b;
			tsa2 += (uint64) a * (uint64) a;
			tsb2 += (uint64) b * (uint64) b;
			tsab += (uint64) a * (uint64) b;
		}

		*sa = (int64) tsa;
		*sb = (int64) tsb;
		*sa2 = (int64) tsa2;
		*sb2 = (int64) tsb2;
		*sab = (int64) tsab;
	}
}
#endif

/*
 * True batch implementations
 * - Generic: one pass over dimensions, update all results.
 * - AVX2/AVX512: reuse unpacked query per block, compute 4 targets per group.
 */

static void
Sq8L2SquaredDistanceSameQuantBatchGenericImpl(int dim, const uint8 *query,
												  const uint8 **targets, int count,
												  int64 *results)
{
	for (int j = 0; j < count; j++)
		results[j] = 0;

	for (int i = 0; i < dim; i++)
	{
		int q = (int) query[i];
		for (int j = 0; j < count; j++)
		{
			int d = q - (int) targets[j][i];
			results[j] += (int64) d * (int64) d;
		}
	}
}

static void
Sq8InnerProductSameQuantBatchGenericImpl(int dim, const uint8 *query,
												 const uint8 **targets, int count,
												 int64 *results)
{
	for (int j = 0; j < count; j++)
		results[j] = 0;

	for (int i = 0; i < dim; i++)
	{
		uint32 q = query[i];
		for (int j = 0; j < count; j++)
			results[j] += (int64) q * (int64) targets[j][i];
	}
}

#if defined(__x86_64__) && defined(__AVX2__)
/*
 * Optimized horizontal sum using hadd instructions
 * This is more efficient on modern CPUs
 */
__attribute__((target("avx2")))
static inline uint64
Sq8ReduceAddEpi32AVX2(__m256i v)
{
	/* Combine 256-bit to 128-bit */
	__m128i lo = _mm256_castsi256_si128(v);
	__m128i hi = _mm256_extracti128_si256(v, 1);
	__m128i sum128 = _mm_add_epi32(lo, hi);
	/* Use hadd for horizontal reduction - 2 hadd ops reduce 4 elements to 1 */
	sum128 = _mm_hadd_epi32(sum128, sum128);
	sum128 = _mm_hadd_epi32(sum128, sum128);
	return (uint64) (uint32) _mm_cvtsi128_si32(sum128);
}

__attribute__((target("avx2")))
static inline int64
Sq8ReduceAddEpi32SignedAVX2(__m256i v)
{
	__m128i lo = _mm256_castsi256_si128(v);
	__m128i hi = _mm256_extracti128_si256(v, 1);
	__m128i sum128 = _mm_add_epi32(lo, hi);
	sum128 = _mm_hadd_epi32(sum128, sum128);
	sum128 = _mm_hadd_epi32(sum128, sum128);
	return (int64) (int32) _mm_cvtsi128_si32(sum128);
}

__attribute__((target("avx2")))
static inline uint64
Sq8ReduceAddEpi64AVX2(__m256i v)
{
	__m128i lo = _mm256_castsi256_si128(v);
	__m128i hi = _mm256_extracti128_si256(v, 1);
	__m128i sum128 = _mm_add_epi64(lo, hi);
	uint64 a = (uint64) _mm_cvtsi128_si64(sum128);
	uint64 b = (uint64) _mm_cvtsi128_si64(_mm_srli_si128(sum128, 8));
	return a + b;
}

__attribute__((target("avx2")))
static void
Sq8L2SquaredDistanceSameQuantBatchAVX2Impl(int dim, const uint8 *query,
												  const uint8 **targets, int count,
												  int64 *results)
{
	int i;
	__m256i zero = _mm256_setzero_si256();

	if (count <= 0)
		return;

	/* Fast path for the common batch size */
	if (count == 4)
	{
		const uint8 *t0 = targets[0];
		const uint8 *t1 = targets[1];
		const uint8 *t2 = targets[2];
		const uint8 *t3 = targets[3];
		__m256i acc0 = _mm256_setzero_si256();
		__m256i acc1 = _mm256_setzero_si256();
		__m256i acc2 = _mm256_setzero_si256();
		__m256i acc3 = _mm256_setzero_si256();

		i = 0;
		for (; i + 32 <= dim; i += 32)
		{
			__m256i vq;
			__m256i q_lo;
			__m256i q_hi;
			_mm_prefetch((const char *)(t0 + i + 256), _MM_HINT_T0);
			_mm_prefetch((const char *)(t1 + i + 256), _MM_HINT_T0);
			_mm_prefetch((const char *)(t2 + i + 256), _MM_HINT_T0);
			_mm_prefetch((const char *)(t3 + i + 256), _MM_HINT_T0);

			vq = _mm256_loadu_si256((const __m256i *) (query + i));
			q_lo = _mm256_unpacklo_epi8(vq, zero);
			q_hi = _mm256_unpackhi_epi8(vq, zero);

			{
				__m256i vt = _mm256_loadu_si256((const __m256i *) (t0 + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				__m256i d_lo = _mm256_sub_epi16(q_lo, t_lo);
				__m256i d_hi = _mm256_sub_epi16(q_hi, t_hi);
				acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d_lo, d_lo));
				acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d_hi, d_hi));
			}
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *) (t1 + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				__m256i d_lo = _mm256_sub_epi16(q_lo, t_lo);
				__m256i d_hi = _mm256_sub_epi16(q_hi, t_hi);
				acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(d_lo, d_lo));
				acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(d_hi, d_hi));
			}
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *) (t2 + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				__m256i d_lo = _mm256_sub_epi16(q_lo, t_lo);
				__m256i d_hi = _mm256_sub_epi16(q_hi, t_hi);
				acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(d_lo, d_lo));
				acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(d_hi, d_hi));
			}
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *) (t3 + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				__m256i d_lo = _mm256_sub_epi16(q_lo, t_lo);
				__m256i d_hi = _mm256_sub_epi16(q_hi, t_hi);
				acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(d_lo, d_lo));
				acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(d_hi, d_hi));
			}
		}

		results[0] = (int64) Sq8ReduceAddEpi32AVX2(acc0);
		results[1] = (int64) Sq8ReduceAddEpi32AVX2(acc1);
		results[2] = (int64) Sq8ReduceAddEpi32AVX2(acc2);
		results[3] = (int64) Sq8ReduceAddEpi32AVX2(acc3);

		/* Optimization 7: Use SIMD for tail when possible (16+ remaining) */
		if (i < dim)
		{
			int remaining = dim - i;
			if (remaining >= 16)
			{
				/* Process 16 bytes with SSE for all 4 targets */
				__m128i vq = _mm_loadu_si128((const __m128i *)(query + i));
				__m128i z = _mm_setzero_si128();
				__m128i q_lo = _mm_unpacklo_epi8(vq, z);
				__m128i q_hi = _mm_unpackhi_epi8(vq, z);
				
				#define PROCESS_TARGET_TAIL(idx, ptr) do { \
					__m128i vt = _mm_loadu_si128((const __m128i *)(ptr + i)); \
					__m128i t_lo = _mm_unpacklo_epi8(vt, z); \
					__m128i t_hi = _mm_unpackhi_epi8(vt, z); \
					__m128i d_lo = _mm_sub_epi16(q_lo, t_lo); \
					__m128i d_hi = _mm_sub_epi16(q_hi, t_hi); \
					__m128i sq_lo = _mm_madd_epi16(d_lo, d_lo); \
					__m128i sq_hi = _mm_madd_epi16(d_hi, d_hi); \
					__m128i s = _mm_add_epi32(sq_lo, sq_hi); \
					s = _mm_hadd_epi32(s, s); \
					s = _mm_hadd_epi32(s, s); \
					results[idx] += (int64)(uint32)_mm_cvtsi128_si32(s); \
				} while(0)
				
				PROCESS_TARGET_TAIL(0, t0);
				PROCESS_TARGET_TAIL(1, t1);
				PROCESS_TARGET_TAIL(2, t2);
				PROCESS_TARGET_TAIL(3, t3);
				#undef PROCESS_TARGET_TAIL
				
				i += 16;
			}
			/* Handle any remaining elements after SIMD tail */
			for (; i < dim; i++)
			{
				int q = (int) query[i];
				int d0 = q - (int) t0[i];
				int d1 = q - (int) t1[i];
				int d2 = q - (int) t2[i];
				int d3 = q - (int) t3[i];
				results[0] += (int64) d0 * (int64) d0;
				results[1] += (int64) d1 * (int64) d1;
				results[2] += (int64) d2 * (int64) d2;
				results[3] += (int64) d3 * (int64) d3;
			}
		}
		return;
	}

	else if (count < 4)
	{
		const uint8 *t0 = targets[0];
		const uint8 *t1 = (count > 1) ? targets[1] : targets[0]; /* Use valid ptr to avoid NULL */
		const uint8 *t2 = (count > 2) ? targets[2] : targets[0];
		/* Note: count < 4, so t3 is never used; removed unnecessary init */
		__m256i acc0 = _mm256_setzero_si256();
		__m256i acc1 = _mm256_setzero_si256();
		__m256i acc2 = _mm256_setzero_si256();
		/* acc3 removed - never used when count < 4 */

		i = 0;
		for (; i + 32 <= dim; i += 32)
		{
			__m256i vq = _mm256_loadu_si256((const __m256i *) (query + i));
			__m256i q_lo = _mm256_unpacklo_epi8(vq, zero);
			__m256i q_hi = _mm256_unpackhi_epi8(vq, zero);

			{
				__m256i vt = _mm256_loadu_si256((const __m256i *) (t0 + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				__m256i d_lo = _mm256_sub_epi16(q_lo, t_lo);
				__m256i d_hi = _mm256_sub_epi16(q_hi, t_hi);
				acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d_lo, d_lo));
				acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(d_hi, d_hi));
			}
			if (count > 1)
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *) (t1 + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				__m256i d_lo = _mm256_sub_epi16(q_lo, t_lo);
				__m256i d_hi = _mm256_sub_epi16(q_hi, t_hi);
				acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(d_lo, d_lo));
				acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(d_hi, d_hi));
			}
			if (count > 2)
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *) (t2 + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				__m256i d_lo = _mm256_sub_epi16(q_lo, t_lo);
				__m256i d_hi = _mm256_sub_epi16(q_hi, t_hi);
				acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(d_lo, d_lo));
				acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(d_hi, d_hi));
			}
			/* Note: count > 3 branch removed since count < 4 in this path */
		}

		results[0] = (int64) Sq8ReduceAddEpi32AVX2(acc0);
		if (count > 1)
			results[1] = (int64) Sq8ReduceAddEpi32AVX2(acc1);
		if (count > 2)
			results[2] = (int64) Sq8ReduceAddEpi32AVX2(acc2);
		/* count > 3 result removed - unreachable when count < 4 */

		/* Optimization 7: Use SIMD for tail when possible */
		if (i < dim)
		{
			int remaining = dim - i;
			if (remaining >= 16)
			{
				__m128i vq = _mm_loadu_si128((const __m128i *)(query + i));
				__m128i z = _mm_setzero_si128();
				__m128i q_lo = _mm_unpacklo_epi8(vq, z);
				__m128i q_hi = _mm_unpackhi_epi8(vq, z);
				
				/* Target 0 - always present */
				{
					__m128i vt = _mm_loadu_si128((const __m128i *)(t0 + i));
					__m128i t_lo = _mm_unpacklo_epi8(vt, z);
					__m128i t_hi = _mm_unpackhi_epi8(vt, z);
					__m128i d_lo = _mm_sub_epi16(q_lo, t_lo);
					__m128i d_hi = _mm_sub_epi16(q_hi, t_hi);
					__m128i sq = _mm_add_epi32(_mm_madd_epi16(d_lo, d_lo), _mm_madd_epi16(d_hi, d_hi));
					sq = _mm_hadd_epi32(sq, sq);
					sq = _mm_hadd_epi32(sq, sq);
					results[0] += (int64)(uint32)_mm_cvtsi128_si32(sq);
				}
				if (count > 1)
				{
					__m128i vt = _mm_loadu_si128((const __m128i *)(t1 + i));
					__m128i t_lo = _mm_unpacklo_epi8(vt, z);
					__m128i t_hi = _mm_unpackhi_epi8(vt, z);
					__m128i d_lo = _mm_sub_epi16(q_lo, t_lo);
					__m128i d_hi = _mm_sub_epi16(q_hi, t_hi);
					__m128i sq = _mm_add_epi32(_mm_madd_epi16(d_lo, d_lo), _mm_madd_epi16(d_hi, d_hi));
					sq = _mm_hadd_epi32(sq, sq);
					sq = _mm_hadd_epi32(sq, sq);
					results[1] += (int64)(uint32)_mm_cvtsi128_si32(sq);
				}
				if (count > 2)
				{
					__m128i vt = _mm_loadu_si128((const __m128i *)(t2 + i));
					__m128i t_lo = _mm_unpacklo_epi8(vt, z);
					__m128i t_hi = _mm_unpackhi_epi8(vt, z);
					__m128i d_lo = _mm_sub_epi16(q_lo, t_lo);
					__m128i d_hi = _mm_sub_epi16(q_hi, t_hi);
					__m128i sq = _mm_add_epi32(_mm_madd_epi16(d_lo, d_lo), _mm_madd_epi16(d_hi, d_hi));
					sq = _mm_hadd_epi32(sq, sq);
					sq = _mm_hadd_epi32(sq, sq);
					results[2] += (int64)(uint32)_mm_cvtsi128_si32(sq);
				}
				i += 16;
			}
			/* Remaining scalar elements */
			for (; i < dim; i++)
			{
				int q = (int) query[i];
				int d0 = q - (int) t0[i];
				results[0] += (int64) d0 * (int64) d0;
				if (count > 1)
				{
					int d1 = q - (int) t1[i];
					results[1] += (int64) d1 * (int64) d1;
				}
				if (count > 2)
				{
					int d2 = q - (int) t2[i];
					results[2] += (int64) d2 * (int64) d2;
				}
			}
		}
		return;
	}

	/*
	 * Optimization 1: For count > 4, use persistent accumulators per target
	 * and defer horizontal reduction until the end.
	 * We allocate accumulators on stack for up to 16 targets, beyond that
	 * we fall back to the simpler approach.
	 */
	if (count <= 16)
	{
		__m256i accumulators[16];
		
		/* Initialize all accumulators to zero */
		for (int j = 0; j < count; j++)
			accumulators[j] = _mm256_setzero_si256();
		
		/* Main SIMD loop - accumulate without reducing */
		i = 0;
		for (; i + 32 <= dim; i += 32)
		{
			__m256i vq = _mm256_loadu_si256((const __m256i *) (query + i));
			__m256i q_lo = _mm256_unpacklo_epi8(vq, zero);
			__m256i q_hi = _mm256_unpackhi_epi8(vq, zero);
			
			for (int j = 0; j < count; j++)
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *) (targets[j] + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				__m256i d_lo = _mm256_sub_epi16(q_lo, t_lo);
				__m256i d_hi = _mm256_sub_epi16(q_hi, t_hi);
				accumulators[j] = _mm256_add_epi32(accumulators[j], _mm256_madd_epi16(d_lo, d_lo));
				accumulators[j] = _mm256_add_epi32(accumulators[j], _mm256_madd_epi16(d_hi, d_hi));
			}
		}
		
		/* Final horizontal reduction for all accumulators */
		for (int j = 0; j < count; j++)
			results[j] = (int64) Sq8ReduceAddEpi32AVX2(accumulators[j]);
		
		/* Optimization 7: Handle remaining elements with SIMD when possible */
		if (i < dim)
		{
			int remaining = dim - i;
			if (remaining >= 16)
			{
				__m128i vq = _mm_loadu_si128((const __m128i *)(query + i));
				__m128i z = _mm_setzero_si128();
				__m128i q_lo = _mm_unpacklo_epi8(vq, z);
				__m128i q_hi = _mm_unpackhi_epi8(vq, z);
				
				for (int j = 0; j < count; j++)
				{
					__m128i vt = _mm_loadu_si128((const __m128i *)(targets[j] + i));
					__m128i t_lo = _mm_unpacklo_epi8(vt, z);
					__m128i t_hi = _mm_unpackhi_epi8(vt, z);
					__m128i d_lo = _mm_sub_epi16(q_lo, t_lo);
					__m128i d_hi = _mm_sub_epi16(q_hi, t_hi);
					__m128i sq = _mm_add_epi32(_mm_madd_epi16(d_lo, d_lo), _mm_madd_epi16(d_hi, d_hi));
					sq = _mm_hadd_epi32(sq, sq);
					sq = _mm_hadd_epi32(sq, sq);
					results[j] += (int64)(uint32)_mm_cvtsi128_si32(sq);
				}
				i += 16;
			}
			/* Remaining scalar elements */
			for (; i < dim; i++)
			{
				int q = (int) query[i];
				for (int j = 0; j < count; j++)
				{
					int d = q - (int) targets[j][i];
					results[j] += (int64) d * (int64) d;
				}
			}
		}
		return;
	}

	/* Fallback for very large count (> 16): original approach */
	for (int j = 0; j < count; j++)
		results[j] = 0;

	i = 0;
	for (; i + 32 <= dim; i += 32)
	{
		__m256i vq = _mm256_loadu_si256((const __m256i *) (query + i));
		__m256i q_lo = _mm256_unpacklo_epi8(vq, zero);
		__m256i q_hi = _mm256_unpackhi_epi8(vq, zero);

		for (int j = 0; j < count; j += 4)
		{
			int group;
			__m256i sum0;
			__m256i sum1;
			__m256i sum2;
			__m256i sum3;

			group = count - j;
			if (group > 4)
				group = 4;

			sum0 = _mm256_setzero_si256();
			sum1 = _mm256_setzero_si256();
			sum2 = _mm256_setzero_si256();
			sum3 = _mm256_setzero_si256();

			if (group >= 1)
			{
				_mm_prefetch((const char *)(targets[j + 0] + i + 256), _MM_HINT_T0);
				{
					__m256i vt = _mm256_loadu_si256((const __m256i *) (targets[j + 0] + i));
					__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
					__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
					__m256i d_lo = _mm256_sub_epi16(q_lo, t_lo);
					__m256i d_hi = _mm256_sub_epi16(q_hi, t_hi);
					sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(d_lo, d_lo));
					sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(d_hi, d_hi));
				}
			}
			if (group >= 2)
			{
				_mm_prefetch((const char *)(targets[j + 1] + i + 256), _MM_HINT_T0);
				{
					__m256i vt = _mm256_loadu_si256((const __m256i *) (targets[j + 1] + i));
					__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
					__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
					__m256i d_lo = _mm256_sub_epi16(q_lo, t_lo);
					__m256i d_hi = _mm256_sub_epi16(q_hi, t_hi);
					sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(d_lo, d_lo));
					sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(d_hi, d_hi));
				}
			}
			if (group >= 3)
			{
				_mm_prefetch((const char *)(targets[j + 2] + i + 256), _MM_HINT_T0);
				{
					__m256i vt = _mm256_loadu_si256((const __m256i *) (targets[j + 2] + i));
					__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
					__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
					__m256i d_lo = _mm256_sub_epi16(q_lo, t_lo);
					__m256i d_hi = _mm256_sub_epi16(q_hi, t_hi);
					sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(d_lo, d_lo));
					sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(d_hi, d_hi));
				}
			}
			if (group >= 4)
			{
				_mm_prefetch((const char *)(targets[j + 3] + i + 256), _MM_HINT_T0);
				{
					__m256i vt = _mm256_loadu_si256((const __m256i *) (targets[j + 3] + i));
					__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
					__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
					__m256i d_lo = _mm256_sub_epi16(q_lo, t_lo);
					__m256i d_hi = _mm256_sub_epi16(q_hi, t_hi);
					sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(d_lo, d_lo));
					sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(d_hi, d_hi));
				}
			}

			if (group >= 1)
				results[j + 0] += (int64) Sq8ReduceAddEpi32AVX2(sum0);
			if (group >= 2)
				results[j + 1] += (int64) Sq8ReduceAddEpi32AVX2(sum1);
			if (group >= 3)
				results[j + 2] += (int64) Sq8ReduceAddEpi32AVX2(sum2);
			if (group >= 4)
				results[j + 3] += (int64) Sq8ReduceAddEpi32AVX2(sum3);
		}
	}

	/* Optimization 7: Handle remaining elements with SIMD when possible */
	if (i < dim)
	{
		int remaining = dim - i;
		if (remaining >= 16)
		{
			__m128i vq = _mm_loadu_si128((const __m128i *)(query + i));
			__m128i z = _mm_setzero_si128();
			__m128i q_lo = _mm_unpacklo_epi8(vq, z);
			__m128i q_hi = _mm_unpackhi_epi8(vq, z);
			int j;
			
			for (j = 0; j < count; j++)
			{
				__m128i vt = _mm_loadu_si128((const __m128i *)(targets[j] + i));
				__m128i t_lo = _mm_unpacklo_epi8(vt, z);
				__m128i t_hi = _mm_unpackhi_epi8(vt, z);
				__m128i d_lo = _mm_sub_epi16(q_lo, t_lo);
				__m128i d_hi = _mm_sub_epi16(q_hi, t_hi);
				__m128i sq = _mm_add_epi32(_mm_madd_epi16(d_lo, d_lo), _mm_madd_epi16(d_hi, d_hi));
				sq = _mm_hadd_epi32(sq, sq);
				sq = _mm_hadd_epi32(sq, sq);
				results[j] += (int64)(uint32)_mm_cvtsi128_si32(sq);
			}
			i += 16;
		}
		/* Remaining scalar elements */
		for (; i < dim; i++)
		{
			int q = (int) query[i];
			int j;
			for (j = 0; j < count; j++)
			{
				int d = q - (int) targets[j][i];
				results[j] += (int64) d * (int64) d;
			}
		}
	}
}

__attribute__((target("avx2")))
static void
Sq8InnerProductSameQuantBatchAVX2Impl(int dim, const uint8 *query,
											 const uint8 **targets, int count,
											 int64 *results)
{
	int i;
	__m256i zero;
	for (int j = 0; j < count; j++)
		results[j] = 0;

	i = 0;
	zero = _mm256_setzero_si256();

	for (; i + 32 <= dim; i += 32)
	{
		__m256i vq = _mm256_loadu_si256((const __m256i *) (query + i));
		__m256i q_lo = _mm256_unpacklo_epi8(vq, zero);
		__m256i q_hi = _mm256_unpackhi_epi8(vq, zero);

		for (int j = 0; j < count; j += 4)
		{
			int group;
			__m256i sum0;
			__m256i sum1;
			__m256i sum2;
			__m256i sum3;

			group = count - j;
			if (group > 4)
				group = 4;

			sum0 = _mm256_setzero_si256();
			sum1 = _mm256_setzero_si256();
			sum2 = _mm256_setzero_si256();
			sum3 = _mm256_setzero_si256();

			if (group >= 1)
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *) (targets[j + 0] + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(q_lo, t_lo));
				sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(q_hi, t_hi));
			}
			if (group >= 2)
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *) (targets[j + 1] + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(q_lo, t_lo));
				sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(q_hi, t_hi));
			}
			if (group >= 3)
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *) (targets[j + 2] + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(q_lo, t_lo));
				sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(q_hi, t_hi));
			}
			if (group >= 4)
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *) (targets[j + 3] + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(q_lo, t_lo));
				sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(q_hi, t_hi));
			}

			if (group >= 1)
				results[j + 0] += (int64) Sq8ReduceAddEpi32AVX2(sum0);
			if (group >= 2)
				results[j + 1] += (int64) Sq8ReduceAddEpi32AVX2(sum1);
			if (group >= 3)
				results[j + 2] += (int64) Sq8ReduceAddEpi32AVX2(sum2);
			if (group >= 4)
				results[j + 3] += (int64) Sq8ReduceAddEpi32AVX2(sum3);
		}
	}

	/* Optimization 7: Handle remaining elements with SIMD when possible */
	if (i < dim)
	{
		int remaining = dim - i;
		if (remaining >= 16)
		{
			__m128i vq = _mm_loadu_si128((const __m128i *)(query + i));
			__m128i z = _mm_setzero_si128();
			__m128i q_lo = _mm_unpacklo_epi8(vq, z);
			__m128i q_hi = _mm_unpackhi_epi8(vq, z);
			int j;
			
			for (j = 0; j < count; j++)
			{
				__m128i vt = _mm_loadu_si128((const __m128i *)(targets[j] + i));
				__m128i t_lo = _mm_unpacklo_epi8(vt, z);
				__m128i t_hi = _mm_unpackhi_epi8(vt, z);
				__m128i prod = _mm_add_epi32(_mm_madd_epi16(q_lo, t_lo), _mm_madd_epi16(q_hi, t_hi));
				prod = _mm_hadd_epi32(prod, prod);
				prod = _mm_hadd_epi32(prod, prod);
				results[j] += (int64)(uint32)_mm_cvtsi128_si32(prod);
			}
			i += 16;
		}
		/* Remaining scalar elements */
		for (; i < dim; i++)
		{
			uint32 q = query[i];
			int j;
			for (j = 0; j < count; j++)
				results[j] += (int64) q * (int64) targets[j][i];
		}
	}
}
#endif

#if defined(__x86_64__)
__attribute__((target("avx512f")))
static inline uint64
Sq8ReduceAddEpi32U64AVX512(__m512i v)
{
	/*
	 * _mm512_reduce_add_epi32 returns a signed int. For our current SQ8 ranges
	 * (dim <= 16000), the sum always fits in 31 bits.
	 */
	return (uint64) (uint32) _mm512_reduce_add_epi32(v);
}

__attribute__((target("avx512f")))
static inline int64
Sq8ReduceAddEpi32SignedAVX512(__m512i v)
{
	return (int64) _mm512_reduce_add_epi32(v);
}
#endif /* __x86_64__ */

#if defined(__x86_64__)
__attribute__((target("avx512f,avx512bw")))
static void
Sq8L2SquaredDistanceSameQuantBatchAVX512Impl(int dim, const uint8 *query,
												 const uint8 **targets, int count,
												 int64 *results)
{
	int i;
	__m512i zero = _mm512_setzero_si512();

	if (count <= 0)
		return;

	/* Fast path for the common batch size of 4 - use persistent accumulators */
	if (count == 4)
	{
		const uint8 *t0 = targets[0];
		const uint8 *t1 = targets[1];
		const uint8 *t2 = targets[2];
		const uint8 *t3 = targets[3];
		__m512i acc0 = _mm512_setzero_si512();
		__m512i acc1 = _mm512_setzero_si512();
		__m512i acc2 = _mm512_setzero_si512();
		__m512i acc3 = _mm512_setzero_si512();

		i = 0;
		for (; i + 64 <= dim; i += 64)
		{
			__m512i vq, q_lo, q_hi;
			__m512i vt, t_lo, t_hi, d_lo, d_hi;

			_mm_prefetch((const char *)(t0 + i + 512), _MM_HINT_T0);
			_mm_prefetch((const char *)(t1 + i + 512), _MM_HINT_T0);
			_mm_prefetch((const char *)(t2 + i + 512), _MM_HINT_T0);
			_mm_prefetch((const char *)(t3 + i + 512), _MM_HINT_T0);

			vq = _mm512_loadu_si512((const void *) (query + i));
			q_lo = _mm512_unpacklo_epi8(vq, zero);
			q_hi = _mm512_unpackhi_epi8(vq, zero);

			/* Target 0 */
			vt = _mm512_loadu_si512((const void *) (t0 + i));
			t_lo = _mm512_unpacklo_epi8(vt, zero);
			t_hi = _mm512_unpackhi_epi8(vt, zero);
			d_lo = _mm512_sub_epi16(q_lo, t_lo);
			d_hi = _mm512_sub_epi16(q_hi, t_hi);
			acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(d_lo, d_lo));
			acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(d_hi, d_hi));

			/* Target 1 */
			vt = _mm512_loadu_si512((const void *) (t1 + i));
			t_lo = _mm512_unpacklo_epi8(vt, zero);
			t_hi = _mm512_unpackhi_epi8(vt, zero);
			d_lo = _mm512_sub_epi16(q_lo, t_lo);
			d_hi = _mm512_sub_epi16(q_hi, t_hi);
			acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(d_lo, d_lo));
			acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(d_hi, d_hi));

			/* Target 2 */
			vt = _mm512_loadu_si512((const void *) (t2 + i));
			t_lo = _mm512_unpacklo_epi8(vt, zero);
			t_hi = _mm512_unpackhi_epi8(vt, zero);
			d_lo = _mm512_sub_epi16(q_lo, t_lo);
			d_hi = _mm512_sub_epi16(q_hi, t_hi);
			acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(d_lo, d_lo));
			acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(d_hi, d_hi));

			/* Target 3 */
			vt = _mm512_loadu_si512((const void *) (t3 + i));
			t_lo = _mm512_unpacklo_epi8(vt, zero);
			t_hi = _mm512_unpackhi_epi8(vt, zero);
			d_lo = _mm512_sub_epi16(q_lo, t_lo);
			d_hi = _mm512_sub_epi16(q_hi, t_hi);
			acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(d_lo, d_lo));
			acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(d_hi, d_hi));
		}

		/* Reduce accumulators only once at the end */
		results[0] = (int64) _mm512_reduce_add_epi32(acc0);
		results[1] = (int64) _mm512_reduce_add_epi32(acc1);
		results[2] = (int64) _mm512_reduce_add_epi32(acc2);
		results[3] = (int64) _mm512_reduce_add_epi32(acc3);

		/* SIMD tail processing: use AVX2 for 32+ remaining bytes */
		if (i + 32 <= dim)
		{
			__m256i z256 = _mm256_setzero_si256();
			__m256i vq256 = _mm256_loadu_si256((const __m256i *)(query + i));
			__m256i q_lo256 = _mm256_unpacklo_epi8(vq256, z256);
			__m256i q_hi256 = _mm256_unpackhi_epi8(vq256, z256);
			__m256i vt256, t_lo256, t_hi256, d_lo256, d_hi256;
			__m256i a0, a1, a2, a3;

			vt256 = _mm256_loadu_si256((const __m256i *)(t0 + i));
			t_lo256 = _mm256_unpacklo_epi8(vt256, z256);
			t_hi256 = _mm256_unpackhi_epi8(vt256, z256);
			d_lo256 = _mm256_sub_epi16(q_lo256, t_lo256);
			d_hi256 = _mm256_sub_epi16(q_hi256, t_hi256);
			a0 = _mm256_add_epi32(_mm256_madd_epi16(d_lo256, d_lo256),
								  _mm256_madd_epi16(d_hi256, d_hi256));

			vt256 = _mm256_loadu_si256((const __m256i *)(t1 + i));
			t_lo256 = _mm256_unpacklo_epi8(vt256, z256);
			t_hi256 = _mm256_unpackhi_epi8(vt256, z256);
			d_lo256 = _mm256_sub_epi16(q_lo256, t_lo256);
			d_hi256 = _mm256_sub_epi16(q_hi256, t_hi256);
			a1 = _mm256_add_epi32(_mm256_madd_epi16(d_lo256, d_lo256),
								  _mm256_madd_epi16(d_hi256, d_hi256));

			vt256 = _mm256_loadu_si256((const __m256i *)(t2 + i));
			t_lo256 = _mm256_unpacklo_epi8(vt256, z256);
			t_hi256 = _mm256_unpackhi_epi8(vt256, z256);
			d_lo256 = _mm256_sub_epi16(q_lo256, t_lo256);
			d_hi256 = _mm256_sub_epi16(q_hi256, t_hi256);
			a2 = _mm256_add_epi32(_mm256_madd_epi16(d_lo256, d_lo256),
								  _mm256_madd_epi16(d_hi256, d_hi256));

			vt256 = _mm256_loadu_si256((const __m256i *)(t3 + i));
			t_lo256 = _mm256_unpacklo_epi8(vt256, z256);
			t_hi256 = _mm256_unpackhi_epi8(vt256, z256);
			d_lo256 = _mm256_sub_epi16(q_lo256, t_lo256);
			d_hi256 = _mm256_sub_epi16(q_hi256, t_hi256);
			a3 = _mm256_add_epi32(_mm256_madd_epi16(d_lo256, d_lo256),
								  _mm256_madd_epi16(d_hi256, d_hi256));

			/* Reduce AVX2 accumulators */
			{
				__m128i sum;
				sum = _mm_add_epi32(_mm256_castsi256_si128(a0), _mm256_extracti128_si256(a0, 1));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
				results[0] += (int64)(uint32)_mm_cvtsi128_si32(sum);

				sum = _mm_add_epi32(_mm256_castsi256_si128(a1), _mm256_extracti128_si256(a1, 1));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
				results[1] += (int64)(uint32)_mm_cvtsi128_si32(sum);

				sum = _mm_add_epi32(_mm256_castsi256_si128(a2), _mm256_extracti128_si256(a2, 1));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
				results[2] += (int64)(uint32)_mm_cvtsi128_si32(sum);

				sum = _mm_add_epi32(_mm256_castsi256_si128(a3), _mm256_extracti128_si256(a3, 1));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
				results[3] += (int64)(uint32)_mm_cvtsi128_si32(sum);
			}
			i += 32;
		}

		/* Scalar tail */
		for (; i < dim; i++)
		{
			int q = (int) query[i];
			int d0 = q - (int) t0[i];
			int d1 = q - (int) t1[i];
			int d2 = q - (int) t2[i];
			int d3 = q - (int) t3[i];
			results[0] += (int64) d0 * (int64) d0;
			results[1] += (int64) d1 * (int64) d1;
			results[2] += (int64) d2 * (int64) d2;
			results[3] += (int64) d3 * (int64) d3;
		}
		return;
	}

	/* General case: use persistent accumulators per target */
	{
		__m512i acc[4];
		int j;
		int remaining_targets = count;
		int base = 0;

		while (remaining_targets > 0)
		{
			int batch = (remaining_targets > 4) ? 4 : remaining_targets;

			for (j = 0; j < batch; j++)
				acc[j] = _mm512_setzero_si512();

			i = 0;
			for (; i + 64 <= dim; i += 64)
			{
				__m512i vq = _mm512_loadu_si512((const void *) (query + i));
				__m512i q_lo = _mm512_unpacklo_epi8(vq, zero);
				__m512i q_hi = _mm512_unpackhi_epi8(vq, zero);

				for (j = 0; j < batch; j++)
				{
					__m512i vt = _mm512_loadu_si512((const void *) (targets[base + j] + i));
					__m512i t_lo = _mm512_unpacklo_epi8(vt, zero);
					__m512i t_hi = _mm512_unpackhi_epi8(vt, zero);
					__m512i d_lo = _mm512_sub_epi16(q_lo, t_lo);
					__m512i d_hi = _mm512_sub_epi16(q_hi, t_hi);
					acc[j] = _mm512_add_epi32(acc[j], _mm512_madd_epi16(d_lo, d_lo));
					acc[j] = _mm512_add_epi32(acc[j], _mm512_madd_epi16(d_hi, d_hi));
				}
			}

			/* Reduce and store */
			for (j = 0; j < batch; j++)
				results[base + j] = (int64) _mm512_reduce_add_epi32(acc[j]);

			/* Scalar tail */
			for (; i < dim; i++)
			{
				int q = (int) query[i];
				for (j = 0; j < batch; j++)
				{
					int d = q - (int) targets[base + j][i];
					results[base + j] += (int64) d * (int64) d;
				}
			}

			base += batch;
			remaining_targets -= batch;
		}
	}
}

__attribute__((target("avx512f,avx512bw")))
static void
Sq8InnerProductSameQuantBatchAVX512Impl(int dim, const uint8 *query,
												const uint8 **targets, int count,
												int64 *results)
{
	int i;
	__m512i zero = _mm512_setzero_si512();

	if (count <= 0)
		return;

	/* Fast path for the common batch size of 4 - use persistent accumulators */
	if (count == 4)
	{
		const uint8 *t0 = targets[0];
		const uint8 *t1 = targets[1];
		const uint8 *t2 = targets[2];
		const uint8 *t3 = targets[3];
		__m512i acc0 = _mm512_setzero_si512();
		__m512i acc1 = _mm512_setzero_si512();
		__m512i acc2 = _mm512_setzero_si512();
		__m512i acc3 = _mm512_setzero_si512();

		i = 0;
		for (; i + 64 <= dim; i += 64)
		{
			__m512i vq, q_lo, q_hi;
			__m512i vt, t_lo, t_hi;

			_mm_prefetch((const char *)(t0 + i + 512), _MM_HINT_T0);
			_mm_prefetch((const char *)(t1 + i + 512), _MM_HINT_T0);
			_mm_prefetch((const char *)(t2 + i + 512), _MM_HINT_T0);
			_mm_prefetch((const char *)(t3 + i + 512), _MM_HINT_T0);

			vq = _mm512_loadu_si512((const void *) (query + i));
			q_lo = _mm512_unpacklo_epi8(vq, zero);
			q_hi = _mm512_unpackhi_epi8(vq, zero);

			/* Target 0 */
			vt = _mm512_loadu_si512((const void *) (t0 + i));
			t_lo = _mm512_unpacklo_epi8(vt, zero);
			t_hi = _mm512_unpackhi_epi8(vt, zero);
			acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(q_lo, t_lo));
			acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(q_hi, t_hi));

			/* Target 1 */
			vt = _mm512_loadu_si512((const void *) (t1 + i));
			t_lo = _mm512_unpacklo_epi8(vt, zero);
			t_hi = _mm512_unpackhi_epi8(vt, zero);
			acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(q_lo, t_lo));
			acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(q_hi, t_hi));

			/* Target 2 */
			vt = _mm512_loadu_si512((const void *) (t2 + i));
			t_lo = _mm512_unpacklo_epi8(vt, zero);
			t_hi = _mm512_unpackhi_epi8(vt, zero);
			acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(q_lo, t_lo));
			acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(q_hi, t_hi));

			/* Target 3 */
			vt = _mm512_loadu_si512((const void *) (t3 + i));
			t_lo = _mm512_unpacklo_epi8(vt, zero);
			t_hi = _mm512_unpackhi_epi8(vt, zero);
			acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(q_lo, t_lo));
			acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(q_hi, t_hi));
		}

		/* Reduce accumulators only once at the end */
		results[0] = (int64) _mm512_reduce_add_epi32(acc0);
		results[1] = (int64) _mm512_reduce_add_epi32(acc1);
		results[2] = (int64) _mm512_reduce_add_epi32(acc2);
		results[3] = (int64) _mm512_reduce_add_epi32(acc3);

		/* SIMD tail processing: use AVX2 for 32+ remaining bytes */
		if (i + 32 <= dim)
		{
			__m256i z256 = _mm256_setzero_si256();
			__m256i vq256 = _mm256_loadu_si256((const __m256i *)(query + i));
			__m256i q_lo256 = _mm256_unpacklo_epi8(vq256, z256);
			__m256i q_hi256 = _mm256_unpackhi_epi8(vq256, z256);
			__m256i vt256, t_lo256, t_hi256;
			__m256i a0, a1, a2, a3;

			vt256 = _mm256_loadu_si256((const __m256i *)(t0 + i));
			t_lo256 = _mm256_unpacklo_epi8(vt256, z256);
			t_hi256 = _mm256_unpackhi_epi8(vt256, z256);
			a0 = _mm256_add_epi32(_mm256_madd_epi16(q_lo256, t_lo256),
								  _mm256_madd_epi16(q_hi256, t_hi256));

			vt256 = _mm256_loadu_si256((const __m256i *)(t1 + i));
			t_lo256 = _mm256_unpacklo_epi8(vt256, z256);
			t_hi256 = _mm256_unpackhi_epi8(vt256, z256);
			a1 = _mm256_add_epi32(_mm256_madd_epi16(q_lo256, t_lo256),
								  _mm256_madd_epi16(q_hi256, t_hi256));

			vt256 = _mm256_loadu_si256((const __m256i *)(t2 + i));
			t_lo256 = _mm256_unpacklo_epi8(vt256, z256);
			t_hi256 = _mm256_unpackhi_epi8(vt256, z256);
			a2 = _mm256_add_epi32(_mm256_madd_epi16(q_lo256, t_lo256),
								  _mm256_madd_epi16(q_hi256, t_hi256));

			vt256 = _mm256_loadu_si256((const __m256i *)(t3 + i));
			t_lo256 = _mm256_unpacklo_epi8(vt256, z256);
			t_hi256 = _mm256_unpackhi_epi8(vt256, z256);
			a3 = _mm256_add_epi32(_mm256_madd_epi16(q_lo256, t_lo256),
								  _mm256_madd_epi16(q_hi256, t_hi256));

			/* Reduce AVX2 accumulators */
			{
				__m128i sum;
				sum = _mm_add_epi32(_mm256_castsi256_si128(a0), _mm256_extracti128_si256(a0, 1));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
				results[0] += (int64)(uint32)_mm_cvtsi128_si32(sum);

				sum = _mm_add_epi32(_mm256_castsi256_si128(a1), _mm256_extracti128_si256(a1, 1));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
				results[1] += (int64)(uint32)_mm_cvtsi128_si32(sum);

				sum = _mm_add_epi32(_mm256_castsi256_si128(a2), _mm256_extracti128_si256(a2, 1));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
				results[2] += (int64)(uint32)_mm_cvtsi128_si32(sum);

				sum = _mm_add_epi32(_mm256_castsi256_si128(a3), _mm256_extracti128_si256(a3, 1));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));
				sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));
				results[3] += (int64)(uint32)_mm_cvtsi128_si32(sum);
			}
			i += 32;
		}

		/* Scalar tail */
		for (; i < dim; i++)
		{
			uint32 q = query[i];
			results[0] += (int64) q * (int64) t0[i];
			results[1] += (int64) q * (int64) t1[i];
			results[2] += (int64) q * (int64) t2[i];
			results[3] += (int64) q * (int64) t3[i];
		}
		return;
	}

	/* General case: use persistent accumulators per target */
	{
		__m512i acc[4];
		int j;
		int remaining_targets = count;
		int base = 0;

		while (remaining_targets > 0)
		{
			int batch = (remaining_targets > 4) ? 4 : remaining_targets;

			for (j = 0; j < batch; j++)
				acc[j] = _mm512_setzero_si512();

			i = 0;
			for (; i + 64 <= dim; i += 64)
			{
				__m512i vq = _mm512_loadu_si512((const void *) (query + i));
				__m512i q_lo = _mm512_unpacklo_epi8(vq, zero);
				__m512i q_hi = _mm512_unpackhi_epi8(vq, zero);

				for (j = 0; j < batch; j++)
				{
					__m512i vt = _mm512_loadu_si512((const void *) (targets[base + j] + i));
					__m512i t_lo = _mm512_unpacklo_epi8(vt, zero);
					__m512i t_hi = _mm512_unpackhi_epi8(vt, zero);
					acc[j] = _mm512_add_epi32(acc[j], _mm512_madd_epi16(q_lo, t_lo));
					acc[j] = _mm512_add_epi32(acc[j], _mm512_madd_epi16(q_hi, t_hi));
				}
			}

			/* Reduce and store */
			for (j = 0; j < batch; j++)
				results[base + j] = (int64) _mm512_reduce_add_epi32(acc[j]);

			/* Scalar tail */
			for (; i < dim; i++)
			{
				uint32 q = query[i];
				for (j = 0; j < batch; j++)
					results[base + j] += (int64) q * (int64) targets[base + j][i];
			}

			base += batch;
			remaining_targets -= batch;
		}
	}
}
#endif

#if defined(__x86_64__)
__attribute__((target("avx512f,avx512bw")))
static void
Sq8AccumulateSumsAVX512(int dim,
					  const uint8 *ax,
					  const uint8 *bx,
					  int64 *sa,
					  int64 *sb,
					  int64 *sa2,
					  int64 *sb2,
					  int64 *sab)
{
	int i = 0;
	__m512i zero = _mm512_setzero_si512();
	__m512i ones16 = _mm512_set1_epi16(1);
	uint64 tsa, tsb, tsa2, tsb2, tsab;

	__m512i sum_a2_32 = _mm512_setzero_si512();
	__m512i sum_b2_32 = _mm512_setzero_si512();
	__m512i sum_ab_32 = _mm512_setzero_si512();
	__m512i sum_a_32 = _mm512_setzero_si512();
	__m512i sum_b_32 = _mm512_setzero_si512();

	for (; i + 64 <= dim; i += 64)
	{
		__m512i va = _mm512_loadu_si512((const void *) (ax + i));
		__m512i vb = _mm512_loadu_si512((const void *) (bx + i));

		/* Sum bytes: widen to u16 and reduce with madd by 1 */
		__m512i va_lo = _mm512_unpacklo_epi8(va, zero);
		__m512i va_hi = _mm512_unpackhi_epi8(va, zero);
		__m512i vb_lo = _mm512_unpacklo_epi8(vb, zero);
		__m512i vb_hi = _mm512_unpackhi_epi8(vb, zero);

		sum_a_32 = _mm512_add_epi32(sum_a_32, _mm512_madd_epi16(va_lo, ones16));
		sum_a_32 = _mm512_add_epi32(sum_a_32, _mm512_madd_epi16(va_hi, ones16));
		sum_b_32 = _mm512_add_epi32(sum_b_32, _mm512_madd_epi16(vb_lo, ones16));
		sum_b_32 = _mm512_add_epi32(sum_b_32, _mm512_madd_epi16(vb_hi, ones16));

		sum_a2_32 = _mm512_add_epi32(sum_a2_32, _mm512_madd_epi16(va_lo, va_lo));
		sum_a2_32 = _mm512_add_epi32(sum_a2_32, _mm512_madd_epi16(va_hi, va_hi));
		sum_b2_32 = _mm512_add_epi32(sum_b2_32, _mm512_madd_epi16(vb_lo, vb_lo));
		sum_b2_32 = _mm512_add_epi32(sum_b2_32, _mm512_madd_epi16(vb_hi, vb_hi));
		sum_ab_32 = _mm512_add_epi32(sum_ab_32, _mm512_madd_epi16(va_lo, vb_lo));
		sum_ab_32 = _mm512_add_epi32(sum_ab_32, _mm512_madd_epi16(va_hi, vb_hi));
	}

	tsa = (uint64) (uint32) _mm512_reduce_add_epi32(sum_a_32);
	tsb = (uint64) (uint32) _mm512_reduce_add_epi32(sum_b_32);
	tsa2 = (uint64) (uint32) _mm512_reduce_add_epi32(sum_a2_32);
	tsb2 = (uint64) (uint32) _mm512_reduce_add_epi32(sum_b2_32);
	tsab = (uint64) (uint32) _mm512_reduce_add_epi32(sum_ab_32);

	for (; i < dim; i++)
	{
		uint32 a = ax[i];
		uint32 b = bx[i];
		tsa += a;
		tsb += b;
		tsa2 += (uint64) a * (uint64) a;
		tsb2 += (uint64) b * (uint64) b;
		tsab += (uint64) a * (uint64) b;
	}

	*sa = (int64) tsa;
	*sb = (int64) tsb;
	*sa2 = (int64) tsa2;
	*sb2 = (int64) tsb2;
	*sab = (int64) tsab;
}
#endif

float
Sq8L2SquaredDistanceGeneric(int dim, const uint8 *ax, const uint8 *bx,
							 float lower_bound_a, float diff_a,
							 float lower_bound_b, float diff_b)
{
	int64 sa, sb, sa2, sb2, sab;
	double scale_a = (double) diff_a / 255.0;
	double scale_b = (double) diff_b / 255.0;
	double delta = (double) lower_bound_a - (double) lower_bound_b;
	double result = 0.0;

	Sq8AccumulateSums(dim, ax, bx, &sa, &sb, &sa2, &sb2, &sab);
	result += (scale_a * scale_a) * (double) sa2;
	result += (scale_b * scale_b) * (double) sb2;
	result += (-2.0 * scale_a * scale_b) * (double) sab;
	result += (2.0 * delta * scale_a) * (double) sa;
	result += (-2.0 * delta * scale_b) * (double) sb;
	result += (double) dim * delta * delta;

	return (float) result;
}

float
Sq8InnerProductGeneric(int dim, const uint8 *ax, const uint8 *bx,
						float lower_bound_a, float diff_a,
						float lower_bound_b, float diff_b)
{
	int64 sa, sb, sa2, sb2, sab;
	double scale_a = (double) diff_a / 255.0;
	double scale_b = (double) diff_b / 255.0;
	double result = 0.0;

	Sq8AccumulateSums(dim, ax, bx, &sa, &sb, &sa2, &sb2, &sab);
	result += (scale_a * scale_b) * (double) sab;
	result += (scale_a * (double) lower_bound_b) * (double) sa;
	result += (scale_b * (double) lower_bound_a) * (double) sb;
	result += (double) dim * (double) lower_bound_a * (double) lower_bound_b;

	return (float) result;
}

double
Sq8CosineSimilarityGeneric(int dim, const uint8 *ax, const uint8 *bx,
							float lower_bound_a, float diff_a,
							float lower_bound_b, float diff_b)
{
	int64 sa, sb, sa2, sb2, sab;
	double scale_a = (double) diff_a / 255.0;
	double scale_b = (double) diff_b / 255.0;
	double dot = 0.0;
	double norm_a = 0.0;
	double norm_b = 0.0;
	double denom;

	Sq8AccumulateSums(dim, ax, bx, &sa, &sb, &sa2, &sb2, &sab);

	dot += (scale_a * scale_b) * (double) sab;
	dot += (scale_a * (double) lower_bound_b) * (double) sa;
	dot += (scale_b * (double) lower_bound_a) * (double) sb;
	dot += (double) dim * (double) lower_bound_a * (double) lower_bound_b;

	norm_a += (scale_a * scale_a) * (double) sa2;
	norm_a += (2.0 * scale_a * (double) lower_bound_a) * (double) sa;
	norm_a += (double) dim * (double) lower_bound_a * (double) lower_bound_a;

	norm_b += (scale_b * scale_b) * (double) sb2;
	norm_b += (2.0 * scale_b * (double) lower_bound_b) * (double) sb;
	norm_b += (double) dim * (double) lower_bound_b * (double) lower_bound_b;

	/* Match halfvec: sqrt(a*b) is slightly better behaved */
	denom = sqrt(norm_a * norm_b);
	if (denom == 0.0)
		return 0.0;

	return dot / denom;
}

float
Sq8L1DistanceGeneric(int dim, const uint8 *ax, const uint8 *bx,
					  float lower_bound_a, float diff_a,
					  float lower_bound_b, float diff_b)
{
	float result = 0.0f;
	float scale_a = diff_a / 255.0f;
	float scale_b = diff_b / 255.0f;
	
	for (int i = 0; i < dim; i++)
	{
		float va = (float) ax[i] * scale_a + lower_bound_a;
		float vb = (float) bx[i] * scale_b + lower_bound_b;
		result += fabsf(va - vb);
	}
	
	return result;
}

/*
 * Same quantization parameter versions - much faster for index internal use
 * Since both vectors use the same lower_bound and diff, we can work in code space:
 * 
 * For L2: |va - vb|^2 = |decode(ca) - decode(cb)|^2
 *                     = |(ca - cb) * diff/255|^2
 *                     = (ca - cb)^2 * (diff/255)^2
 * 
 * We compute sum((ca - cb)^2) in integer space and scale at the end.
 */
static int64
Sq8L2SquaredDistanceSameQuantGeneric(int dim, const uint8 *ax, const uint8 *bx)
{
	int64 result = 0;
	
	for (int i = 0; i < dim; i++)
	{
		int d = (int)ax[i] - (int)bx[i];
		result += (int64)d * d;
	}
	
	return result;
}

/*
 * Inner product in code space:
 * va * vb = (ca/255*diff + lb) * (cb/255*diff + lb)
 * 
 * For index searches where we only need ranking (not exact values),
 * we can just compute sum(ca * cb) since the transformation is monotonic
 * when diff > 0.
 */
static int64
Sq8InnerProductSameQuantGeneric(int dim, const uint8 *ax, const uint8 *bx)
{
	int64 result = 0;
	
	for (int i = 0; i < dim; i++)
	{
		result += (int64)ax[i] * (int64)bx[i];
	}
	
	return result;
}

/*
 * AVX2 optimized versions
 * Optimization 2: Use algebraic form L2² = Σa² + Σb² - 2Σab with maddubs
 * Optimization 6: Unified with batch path design
 * Optimization 7: SIMD tail processing for remaining elements
 * Optimization 8: Use hadd for horizontal reduction
 */
#if defined(__x86_64__) && defined(__AVX2__)

/*
 * Helper: Process remaining 1-31 bytes using partial SIMD
 * Uses a stack buffer to safely handle unaligned tail
 */
__attribute__((target("avx2")))
static inline int64
Sq8L2TailAVX2(int remaining, const uint8 *ax, const uint8 *bx)
{
	if (remaining <= 0)
		return 0;
	
	if (remaining >= 16)
	{
		/* Process 16 bytes with SSE */
		__m128i va = _mm_loadu_si128((const __m128i *)ax);
		__m128i vb = _mm_loadu_si128((const __m128i *)bx);
		__m128i zero = _mm_setzero_si128();
		
		__m128i va_lo = _mm_unpacklo_epi8(va, zero);
		__m128i va_hi = _mm_unpackhi_epi8(va, zero);
		__m128i vb_lo = _mm_unpacklo_epi8(vb, zero);
		__m128i vb_hi = _mm_unpackhi_epi8(vb, zero);
		
		__m128i d_lo = _mm_sub_epi16(va_lo, vb_lo);
		__m128i d_hi = _mm_sub_epi16(va_hi, vb_hi);
		
		__m128i sq_lo = _mm_madd_epi16(d_lo, d_lo);
		__m128i sq_hi = _mm_madd_epi16(d_hi, d_hi);
		__m128i sum128 = _mm_add_epi32(sq_lo, sq_hi);
		int64 result;
		int j;
		
		sum128 = _mm_hadd_epi32(sum128, sum128);
		sum128 = _mm_hadd_epi32(sum128, sum128);
		result = (int64)(uint32)_mm_cvtsi128_si32(sum128);
		
		/* Handle remaining after 16 bytes */
		for (j = 16; j < remaining; j++)
		{
			int d = (int)ax[j] - (int)bx[j];
			result += (int64)d * d;
		}
		return result;
	}
	else
	{
		/* Less than 16 bytes - use scalar */
		int64 result = 0;
		int j;
		for (j = 0; j < remaining; j++)
		{
			int d = (int)ax[j] - (int)bx[j];
			result += (int64)d * d;
		}
		return result;
	}
}

__attribute__((target("avx2")))
static int64
Sq8L2SquaredDistanceSameQuantAVX2(int dim, const uint8 *ax, const uint8 *bx)
{
	int i = 0;
	__m256i zero = _mm256_setzero_si256();
	__m256i sum = _mm256_setzero_si256();
	int64 result;
	
	for (; i + 32 <= dim; i += 32)
	{
		__m256i va = _mm256_loadu_si256((const __m256i *)(ax + i));
		__m256i vb = _mm256_loadu_si256((const __m256i *)(bx + i));
		
		/* Unpack to 16-bit for subtraction */
		__m256i va_lo = _mm256_unpacklo_epi8(va, zero);
		__m256i va_hi = _mm256_unpackhi_epi8(va, zero);
		__m256i vb_lo = _mm256_unpacklo_epi8(vb, zero);
		__m256i vb_hi = _mm256_unpackhi_epi8(vb, zero);
		
		/* Subtract */
		__m256i diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
		__m256i diff_hi = _mm256_sub_epi16(va_hi, vb_hi);
		
		/* Square using madd (multiply and add pairs) */
		__m256i sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
		__m256i sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);
		
		/* Accumulate */
		sum = _mm256_add_epi32(sum, sq_lo);
		sum = _mm256_add_epi32(sum, sq_hi);
	}
	
	/* Horizontal sum using hadd */
	result = (int64) Sq8ReduceAddEpi32AVX2(sum);
	
	/* Optimization 7: Handle remaining elements with SIMD helper */
	if (i < dim)
		result += Sq8L2TailAVX2(dim - i, ax + i, bx + i);
	
	return result;
}

/*
 * Helper: Process remaining 1-31 bytes for Inner Product using partial SIMD
 */
__attribute__((target("avx2")))
static inline int64
Sq8IPTailAVX2(int remaining, const uint8 *ax, const uint8 *bx)
{
	if (remaining <= 0)
		return 0;
	
	if (remaining >= 16)
	{
		/* Process 16 bytes with SSE */
		__m128i va = _mm_loadu_si128((const __m128i *)ax);
		__m128i vb = _mm_loadu_si128((const __m128i *)bx);
		__m128i zero = _mm_setzero_si128();
		
		__m128i va_lo = _mm_unpacklo_epi8(va, zero);
		__m128i va_hi = _mm_unpackhi_epi8(va, zero);
		__m128i vb_lo = _mm_unpacklo_epi8(vb, zero);
		__m128i vb_hi = _mm_unpackhi_epi8(vb, zero);
		
		__m128i prod_lo = _mm_madd_epi16(va_lo, vb_lo);
		__m128i prod_hi = _mm_madd_epi16(va_hi, vb_hi);
		__m128i sum128 = _mm_add_epi32(prod_lo, prod_hi);
		int64 result;
		int j;
		
		sum128 = _mm_hadd_epi32(sum128, sum128);
		sum128 = _mm_hadd_epi32(sum128, sum128);
		result = (int64)(uint32)_mm_cvtsi128_si32(sum128);
		
		/* Handle remaining after 16 bytes */
		for (j = 16; j < remaining; j++)
		{
			result += (int64)ax[j] * (int64)bx[j];
		}
		return result;
	}
	else
	{
		/* Less than 16 bytes - use scalar */
		int64 result = 0;
		int j;
		for (j = 0; j < remaining; j++)
		{
			result += (int64)ax[j] * (int64)bx[j];
		}
		return result;
	}
}

__attribute__((target("avx2")))
static int64
Sq8InnerProductSameQuantAVX2(int dim, const uint8 *ax, const uint8 *bx)
{
	int i = 0;
	__m256i zero = _mm256_setzero_si256();
	__m256i sum = _mm256_setzero_si256();
	int64 result;
	
	for (; i + 32 <= dim; i += 32)
	{
		__m256i va = _mm256_loadu_si256((const __m256i *)(ax + i));
		__m256i vb = _mm256_loadu_si256((const __m256i *)(bx + i));
		
		/* Unpack to 16-bit */
		__m256i va_lo = _mm256_unpacklo_epi8(va, zero);
		__m256i va_hi = _mm256_unpackhi_epi8(va, zero);
		__m256i vb_lo = _mm256_unpacklo_epi8(vb, zero);
		__m256i vb_hi = _mm256_unpackhi_epi8(vb, zero);
		
		/* Multiply using madd */
		__m256i prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
		__m256i prod_hi = _mm256_madd_epi16(va_hi, vb_hi);
		
		/* Accumulate */
		sum = _mm256_add_epi32(sum, prod_lo);
		sum = _mm256_add_epi32(sum, prod_hi);
	}
	
	/* Horizontal sum using hadd via shared function */
	result = (int64) Sq8ReduceAddEpi32AVX2(sum);
	
	/* Optimization 7: Handle remaining elements with SIMD helper */
	if (i < dim)
		result += Sq8IPTailAVX2(dim - i, ax + i, bx + i);
	
	return result;
}

#endif /* AVX2 */

/*
 * AVX-512 optimized versions
 */
#if defined(__x86_64__)

__attribute__((target("avx512f,avx512bw")))
static int64
Sq8L2SquaredDistanceSameQuantAVX512(int dim, const uint8 *ax, const uint8 *bx)
{
	int64 result = 0;
	int i = 0;
	
	__m512i zero = _mm512_setzero_si512();
	__m512i sum = _mm512_setzero_si512();
	
	for (; i + 64 <= dim; i += 64)
	{
		__m512i va = _mm512_loadu_si512((const void *) (ax + i));
		__m512i vb = _mm512_loadu_si512((const void *) (bx + i));
		
		/* Unpack to 16-bit */
		__m512i va_lo = _mm512_unpacklo_epi8(va, zero);
		__m512i va_hi = _mm512_unpackhi_epi8(va, zero);
		__m512i vb_lo = _mm512_unpacklo_epi8(vb, zero);
		__m512i vb_hi = _mm512_unpackhi_epi8(vb, zero);
		
		/* Subtract */
		__m512i diff_lo = _mm512_sub_epi16(va_lo, vb_lo);
		__m512i diff_hi = _mm512_sub_epi16(va_hi, vb_hi);
		
		/* Square using madd */
		__m512i sq_lo = _mm512_madd_epi16(diff_lo, diff_lo);
		__m512i sq_hi = _mm512_madd_epi16(diff_hi, diff_hi);
		
		/* Accumulate */
		sum = _mm512_add_epi32(sum, sq_lo);
		sum = _mm512_add_epi32(sum, sq_hi);
	}
	
	/* Reduce to scalar */
	result = (int64)_mm512_reduce_add_epi32(sum);
	
	/* Handle remaining elements */
	for (; i < dim; i++)
	{
		int d = (int)ax[i] - (int)bx[i];
		result += (int64)d * d;
	}
	
	return result;
}

__attribute__((target("avx512f,avx512bw")))
static int64
Sq8InnerProductSameQuantAVX512(int dim, const uint8 *ax, const uint8 *bx)
{
	int64 result = 0;
	int i = 0;
	
	__m512i zero = _mm512_setzero_si512();
	__m512i sum = _mm512_setzero_si512();
	
	for (; i + 64 <= dim; i += 64)
	{
		__m512i va = _mm512_loadu_si512((const void *) (ax + i));
		__m512i vb = _mm512_loadu_si512((const void *) (bx + i));
		
		/* Unpack to 16-bit */
		__m512i va_lo = _mm512_unpacklo_epi8(va, zero);
		__m512i va_hi = _mm512_unpackhi_epi8(va, zero);
		__m512i vb_lo = _mm512_unpacklo_epi8(vb, zero);
		__m512i vb_hi = _mm512_unpackhi_epi8(vb, zero);
		
		/* Multiply using madd */
		__m512i prod_lo = _mm512_madd_epi16(va_lo, vb_lo);
		__m512i prod_hi = _mm512_madd_epi16(va_hi, vb_hi);
		
		/* Accumulate */
		sum = _mm512_add_epi32(sum, prod_lo);
		sum = _mm512_add_epi32(sum, prod_hi);
	}
	
	/* Reduce to scalar */
	result = (int64)_mm512_reduce_add_epi32(sum);
	
	/* Handle remaining elements */
	for (; i < dim; i++)
	{
		result += (int64)ax[i] * (int64)bx[i];
	}
	
	return result;
}

#endif /* __x86_64__ */

/*
 * CPU feature detection and function pointer initialization
 */
#if defined(__x86_64__) && defined(HAVE__GET_CPUID)
/* Robust OSXSAVE + XCR0 checks (similar to halfvec dispatch) */
#define CPU_FEATURE_OSXSAVE (1u << 27)
#define CPU_FEATURE_AVX     (1u << 28)

#ifdef _MSC_VER
#define TARGET_XSAVE
#else
#define TARGET_XSAVE __attribute__((target("xsave")))
#endif

TARGET_XSAVE static bool
SupportsAvxState(uint64 xcr0_mask)
{
	unsigned int exx[4] = {0, 0, 0, 0};

	__get_cpuid(1, &exx[0], &exx[1], &exx[2], &exx[3]);

	if ((exx[2] & CPU_FEATURE_OSXSAVE) != CPU_FEATURE_OSXSAVE)
		return false;
	if ((exx[2] & CPU_FEATURE_AVX) != CPU_FEATURE_AVX)
		return false;

	return ((_xgetbv(0) & xcr0_mask) == xcr0_mask);
}

static bool
CheckAVX2Support(void)
{
	unsigned int eax, ebx, ecx, edx;

	/* Need XMM+YMM state enabled */
	if (!SupportsAvxState(6))
		return false;

	if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
		return false;

	(void) eax;
	(void) ecx;
	(void) edx;
	return (ebx & (1u << 5)) != 0; /* AVX2 */
}

static bool __attribute__((unused))
CheckAVX512Support(void)
{
	unsigned int eax, ebx, ecx, edx;

	/* Need XMM+YMM + opmask + ZMM state */
	if (!SupportsAvxState(0xE6))
		return false;

	if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
		return false;

	(void) eax;
	(void) ecx;
	(void) edx;
	/* AVX-512F (16) + AVX-512BW (30) */
	return ((ebx & (1u << 16)) != 0) && ((ebx & (1u << 30)) != 0);
}

static bool __attribute__((unused))
CheckAVXVNNISupport(void)
{
	unsigned int eax, ebx, ecx, edx;
	unsigned int max_subleaf;

	/* Need AVX2 first */
	if (!CheckAVX2Support())
		return false;

	if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
		return false;

	/* CPUID.(EAX=7,ECX=0):EAX contains the maximum supported subleaf */
	max_subleaf = eax;
	if (max_subleaf < 1)
		return false;

	/* Check CPUID.(EAX=7,ECX=1):EAX[4] for AVX-VNNI */
	if (!__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx))
		return false;

	return (eax & (1u << 4)) != 0;
}

static bool __attribute__((unused))
CheckAVX512VNNISupport(void)
{
	unsigned int eax, ebx, ecx, edx;

	/*
	 * Need AVX-512F+BW and OS support for ZMM state.
	 * AVX512VNNI flag: CPUID.(EAX=7,ECX=0):ECX[11].
	 */
	if (!CheckAVX512Support())
		return false;

	if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
		return false;

	(void) eax;
	(void) ebx;
	(void) edx;
	return (ecx & (1u << 11)) != 0;
}
		#endif

/* Forward declarations for optimized L2 with sum2 functions */
static void Sq8L2SquaredDistanceWithSum2BatchGenericImpl(int dim, const uint8 *query,
                                                         int32 query_sum, int32 query_sum2,
                                                         const uint8 **targets, const int32 *target_sum2s,
                                                         int count, int64 *results);
static int64 Sq8L2SquaredDistanceWithSum2SingleFallbackImpl(int dim, const uint8 *ax,
															int32 sum_a, int32 sum2_a,
															const uint8 *bx, int32 sum2_b);
#if defined(__x86_64__) && defined(__AVX2__)
static void Sq8L2SquaredDistanceWithSum2BatchAVX2Impl(int dim, const uint8 *query,
                                                      int32 query_sum, int32 query_sum2,
                                                      const uint8 **targets, const int32 *target_sum2s,
                                                      int count, int64 *results);
static void Sq8L2SquaredDistanceWithSum2BatchVNNIImpl(int dim, const uint8 *query,
                                                      int32 query_sum, int32 query_sum2,
                                                      const uint8 **targets, const int32 *target_sum2s,
                                                      int count, int64 *results);
static int64 Sq8L2SquaredDistanceWithSum2SingleVNNIImpl(int dim, const uint8 *ax,
														int32 sum_a, int32 sum2_a,
														const uint8 *bx, int32 sum2_b);
#endif
#if defined(__x86_64__)
static void Sq8L2SquaredDistanceWithSum2BatchAVX512Impl(int dim, const uint8 *query,
                                                        int32 query_sum, int32 query_sum2,
                                                        const uint8 **targets, const int32 *target_sum2s,
                                                        int count, int64 *results);
static void Sq8L2SquaredDistanceWithSum2BatchAVX512VNNIImpl(int dim, const uint8 *query,
                                                            int32 query_sum, int32 query_sum2,
                                                            const uint8 **targets, const int32 *target_sum2s,
                                                            int count, int64 *results);
static int64 Sq8L2SquaredDistanceWithSum2SingleAVX512VNNIImpl(int dim, const uint8 *ax,
															  int32 sum_a, int32 sum2_a,
															  const uint8 *bx, int32 sum2_b);
#endif

void
Sq8utilsInit(void)
{
	/* Set generic implementations as default */
	Sq8L2SquaredDistance = Sq8L2SquaredDistanceGeneric;
	Sq8InnerProduct = Sq8InnerProductGeneric;
	Sq8CosineSimilarity = Sq8CosineSimilarityGeneric;
	Sq8L1Distance = Sq8L1DistanceGeneric;
	Sq8L2SquaredDistanceSameQuant = Sq8L2SquaredDistanceSameQuantGeneric;
	Sq8InnerProductSameQuant = Sq8InnerProductSameQuantGeneric;
	Sq8AccumulateSums = Sq8AccumulateSumsGeneric;
	Sq8L2SquaredDistanceSameQuantBatchImpl = Sq8L2SquaredDistanceSameQuantBatchGenericImpl;
	Sq8InnerProductSameQuantBatchImpl = Sq8InnerProductSameQuantBatchGenericImpl;
	Sq8L2SquaredDistanceWithSum2BatchImpl = Sq8L2SquaredDistanceWithSum2BatchGenericImpl;
	Sq8L2SquaredDistanceWithSum2SingleImpl = Sq8L2SquaredDistanceWithSum2SingleFallbackImpl;
		

#if defined(__x86_64__) && defined(HAVE__GET_CPUID)
	if (CheckAVX512Support())
	{
		Sq8L2SquaredDistanceSameQuant = Sq8L2SquaredDistanceSameQuantAVX512;
		Sq8InnerProductSameQuant = Sq8InnerProductSameQuantAVX512;
		Sq8AccumulateSums = Sq8AccumulateSumsAVX512;
		Sq8L2SquaredDistanceSameQuantBatchImpl = Sq8L2SquaredDistanceSameQuantBatchAVX512Impl;
		Sq8InnerProductSameQuantBatchImpl = Sq8InnerProductSameQuantBatchAVX512Impl;
		if (CheckAVX512VNNISupport())
		{
			Sq8L2SquaredDistanceWithSum2BatchImpl = Sq8L2SquaredDistanceWithSum2BatchAVX512VNNIImpl;
			Sq8L2SquaredDistanceWithSum2SingleImpl = Sq8L2SquaredDistanceWithSum2SingleAVX512VNNIImpl;
		}
		else
		{
			Sq8L2SquaredDistanceWithSum2BatchImpl = Sq8L2SquaredDistanceWithSum2BatchAVX512Impl;
			Sq8L2SquaredDistanceWithSum2SingleImpl = Sq8L2SquaredDistanceWithSum2SingleFallbackImpl;
		}
		return;
	}
	
	#if defined(__AVX2__)
	/* Check for AVX-VNNI first (better performance) */
	if (CheckAVXVNNISupport())
	{
		Sq8L2SquaredDistanceSameQuant = Sq8L2SquaredDistanceSameQuantAVX2;
		Sq8InnerProductSameQuant = Sq8InnerProductSameQuantAVX2;
		Sq8AccumulateSums = Sq8AccumulateSumsAVX2;
		Sq8L2SquaredDistanceSameQuantBatchImpl = Sq8L2SquaredDistanceSameQuantBatchAVX2Impl;
		Sq8InnerProductSameQuantBatchImpl = Sq8InnerProductSameQuantBatchAVX2Impl;
		Sq8L2SquaredDistanceWithSum2BatchImpl = Sq8L2SquaredDistanceWithSum2BatchVNNIImpl;
		Sq8L2SquaredDistanceWithSum2SingleImpl = Sq8L2SquaredDistanceWithSum2SingleVNNIImpl;
		return;
	}
	
	if (CheckAVX2Support())
	{
		Sq8L2SquaredDistanceSameQuant = Sq8L2SquaredDistanceSameQuantAVX2;
		Sq8InnerProductSameQuant = Sq8InnerProductSameQuantAVX2;
		Sq8AccumulateSums = Sq8AccumulateSumsAVX2;
		Sq8L2SquaredDistanceSameQuantBatchImpl = Sq8L2SquaredDistanceSameQuantBatchAVX2Impl;
		Sq8InnerProductSameQuantBatchImpl = Sq8InnerProductSameQuantBatchAVX2Impl;
		Sq8L2SquaredDistanceWithSum2BatchImpl = Sq8L2SquaredDistanceWithSum2BatchAVX2Impl;
		Sq8L2SquaredDistanceWithSum2SingleImpl = Sq8L2SquaredDistanceWithSum2SingleFallbackImpl;
		return;
	}
	#endif
#endif
}

/*
 * Global SQ8 Quantization Functions
 * 
 * These functions support computing and using global quantization parameters
 * for HNSW indexes. Instead of per-vector lower_bound/diff, we use a single
 * set of parameters computed from a sample of vectors during index build.
 */

/*
 * Compute global quantization parameters from sampled float vectors.
 * Scans all samples to find the global min and max values across all dimensions.
 */
void
Sq8ComputeGlobalQuantParams(float **samples, int nsamples, int dim,
                             float *lower_bound, float *diff)
{
    float global_min = samples[0][0];
    float global_max = samples[0][0];
    
    /* Find global min/max across all samples and all dimensions */
    for (int i = 0; i < nsamples; i++)
    {
        const float *vec = samples[i];
        for (int j = 0; j < dim; j++)
        {
            if (vec[j] < global_min)
                global_min = vec[j];
            if (vec[j] > global_max)
                global_max = vec[j];
        }
    }
    
    *lower_bound = global_min;
    *diff = global_max - global_min;
    
    /* Avoid division by zero - if all values are same, use small diff */
    if (*diff < 1e-10f)
        *diff = 1e-10f;
}

/*
 * Encode a float vector using global quantization parameters.
 * This is similar to Sq8EncodeVector but takes explicit global params.
 */
void
Sq8EncodeWithGlobalParams(const float *values, int dim,
                           float lower_bound, float diff, uint8 *codes)
{
    float scale = 255.0f / diff;
    for (int i = 0; i < dim; i++)
    {
        float codef = (values[i] - lower_bound) * scale;
        /* Clamp to valid range */
        if (codef < 0.0f)
            codef = 0.0f;
        else if (codef > 255.0f)
            codef = 255.0f;
        codes[i] = (uint8)(codef + 0.5f);
    }
}

/*
 * Batch distance computation for SQ8 vectors (same quantization).
 * Computes L2 squared distance from query to multiple targets in one call.
 * This is more efficient than calling Sq8L2SquaredDistanceSameQuant repeatedly.
 */
void
Sq8L2SquaredDistanceSameQuantBatch(int dim, const uint8 *query,
                                    const uint8 **targets, int count,
                                    int64 *results)
{
	if (count <= 0)
		return;
	if (Sq8L2SquaredDistanceSameQuantBatchImpl)
		Sq8L2SquaredDistanceSameQuantBatchImpl(dim, query, targets, count, results);
	else
		Sq8L2SquaredDistanceSameQuantBatchGenericImpl(dim, query, targets, count, results);
}

/*
 * Batch inner product computation for SQ8 vectors (same quantization).
 * Computes inner product from query to multiple targets in one call.
 */
void
Sq8InnerProductSameQuantBatch(int dim, const uint8 *query,
                               const uint8 **targets, int count,
                               int64 *results)
{
	if (count <= 0)
		return;
	if (Sq8InnerProductSameQuantBatchImpl)
		Sq8InnerProductSameQuantBatchImpl(dim, query, targets, count, results);
	else
		Sq8InnerProductSameQuantBatchGenericImpl(dim, query, targets, count, results);
}

/*
 * Optimized IP distance computation using precomputed sums.
 * Formula: dot = scale^2 * dot_ab + scale*lb*(sum_a + sum_b) + dim*lb^2
 */
float
Sq8InnerProductWithSums(int64 dot_product, int32 sum_q, int32 sum_t,
                        int dim, float lower_bound, float scale)
{
	double scale_d = (double)scale;
	double lb_d = (double)lower_bound;
	
	double result = scale_d * scale_d * (double)dot_product;
	result += scale_d * lb_d * ((double)sum_q + (double)sum_t);
	result += (double)dim * lb_d * lb_d;
	
	return (float)result;
}

/*
 * Optimized Cosine distance computation using precomputed sums.
 * Formula: cosine_sim = dot / (norm_a * norm_b)
 * where norm^2 = scale^2 * sum2 + 2*scale*lb*sum + dim*lb^2
 */
float
Sq8CosineDistanceWithSums(int64 dot_product, int32 sum_q, int32 sum2_q,
                          int32 sum_t, int32 sum2_t, int dim,
                          float lower_bound, float scale)
{
	double scale_d = (double)scale;
	double lb_d = (double)lower_bound;
	double scale2 = scale_d * scale_d;
	double dot;
	double norm_q2;
	double norm_t2;
	double denom;
	double cosine_sim;
	
	/* Compute dot product */
	dot = scale2 * (double)dot_product;
	dot += scale_d * lb_d * ((double)sum_q + (double)sum_t);
	dot += (double)dim * lb_d * lb_d;
	
	/* Compute norms */
	norm_q2 = scale2 * (double)sum2_q;
	norm_q2 += 2.0 * scale_d * lb_d * (double)sum_q;
	norm_q2 += (double)dim * lb_d * lb_d;
	
	norm_t2 = scale2 * (double)sum2_t;
	norm_t2 += 2.0 * scale_d * lb_d * (double)sum_t;
	norm_t2 += (double)dim * lb_d * lb_d;
	
	/* Compute cosine similarity */
	denom = sqrt(norm_q2 * norm_t2);
	if (denom == 0.0)
		return 1.0f;  /* Maximum distance for zero vectors */
	
	cosine_sim = dot / denom;
	
	/* Return cosine distance: 1 - cosine_similarity */
	return (float)(1.0 - cosine_sim);
}

/*
 * ============================================================================
 * Optimized L2 Distance using Sum2
 * ============================================================================
 * 
 * Formula: L2² = Σ(a-b)² = Σa² + Σb² - 2×Σab
 * 
 * Since sum2 (Σcodes²) is precomputed for both query and targets,
 * we only need to compute the inner product Σab.
 * 
 * This is more efficient because:
 * 1. Query's sum2 is computed once during query encoding
 * 2. Target's sum2 is stored in the index (Sq8IndexTuple->sum2)
 * 3. Inner product can use efficient VNNI instructions
 */

/*
 * Fallback single-point implementation.
 *
 * Prefer the direct (a-b)^2 kernel for non-VNNI paths since it is usually faster
 * than dot-based reconstruction at small dimensions.
 */
static int64
Sq8L2SquaredDistanceWithSum2SingleFallbackImpl(int dim, const uint8 *ax,
											   int32 sum_a, int32 sum2_a,
											   const uint8 *bx, int32 sum2_b)
{
	(void) sum_a;
	(void) sum2_a;
	(void) sum2_b;

	if (Sq8L2SquaredDistanceSameQuant != NULL)
		return Sq8L2SquaredDistanceSameQuant(dim, ax, bx);

	/* Defensive fallback - should not happen */
	{
		int64 dist = 0;
		for (int i = 0; i < dim; i++)
		{
			int d = (int) ax[i] - (int) bx[i];
			dist += (int64) d * (int64) d;
		}
		return dist;
	}
}

#if defined(__x86_64__) && defined(__AVX2__)
__attribute__((target("avx2,avxvnni")))
static int64
Sq8L2SquaredDistanceWithSum2SingleVNNIImpl(int dim, const uint8 *ax,
										   int32 sum_a, int32 sum2_a,
										   const uint8 *bx, int32 sum2_b)
{
	int i = 0;
	int64 sum_q = (int64) sum_a;
	__m256i bias = _mm256_set1_epi8((char) 0x80);
	__m256i acc = _mm256_setzero_si256();

	for (; i + 32 <= dim; i += 32)
	{
		__m256i va = _mm256_loadu_si256((const __m256i *) (ax + i));
		__m256i vb = _mm256_loadu_si256((const __m256i *) (bx + i));
		vb = _mm256_xor_si256(vb, bias); /* b - 128 as signed byte */
		acc = _mm256_dpbusd_epi32(acc, va, vb);
	}

	{
		int64 dp = Sq8ReduceAddEpi32SignedAVX2(acc);

		for (; i < dim; i++)
		{
			int q = (int) ax[i];
			dp += (int64) q * ((int64) (int) bx[i] - 128);
		}

		/* dot = dp + 128*sum(q) */
		{
			int64 dot = dp + 128 * sum_q;
			return (int64) sum2_a + (int64) sum2_b - 2 * dot;
		}
	}
}
#endif /* __AVX2__ */

#if defined(__x86_64__)
__attribute__((target("avx512f,avx512bw,avx512vnni")))
static int64
Sq8L2SquaredDistanceWithSum2SingleAVX512VNNIImpl(int dim, const uint8 *ax,
												 int32 sum_a, int32 sum2_a,
												 const uint8 *bx, int32 sum2_b)
{
	int i = 0;
	int64 sum_q = (int64) sum_a;
	__m512i bias = _mm512_set1_epi8((char) 0x80);
	__m512i acc = _mm512_setzero_si512();

	for (; i + 64 <= dim; i += 64)
	{
		__m512i va = _mm512_loadu_si512((const void *) (ax + i));
		__m512i vb = _mm512_loadu_si512((const void *) (bx + i));
		vb = _mm512_xor_si512(vb, bias); /* b - 128 as signed byte */
		acc = _mm512_dpbusd_epi32(acc, va, vb);
	}

	{
		int64 dp = Sq8ReduceAddEpi32SignedAVX512(acc);

		for (; i < dim; i++)
		{
			int q = (int) ax[i];
			dp += (int64) q * ((int64) (int) bx[i] - 128);
		}

		{
			int64 dot = dp + 128 * sum_q;
			return (int64) sum2_a + (int64) sum2_b - 2 * dot;
		}
	}
}
#endif /* __x86_64__ */

/*
 * Single-point L2 squared distance using sum2 optimization.
 * Uses the best available single-point kernel (VNNI when available).
 */
int64
Sq8L2SquaredDistanceWithSum2(int dim, const uint8 *ax,
							  int32 sum_a, int32 sum2_a,
							  const uint8 *bx, int32 sum2_b)
{
	if (Sq8L2SquaredDistanceWithSum2SingleImpl != NULL)
		return Sq8L2SquaredDistanceWithSum2SingleImpl(dim, ax, sum_a, sum2_a, bx, sum2_b);

	return Sq8L2SquaredDistanceWithSum2SingleFallbackImpl(dim, ax, sum_a, sum2_a, bx, sum2_b);
}

/* Generic implementation - query_sum unused in generic, but kept for API consistency */
static void
Sq8L2SquaredDistanceWithSum2BatchGenericImpl(int dim, const uint8 *query,
											 int32 query_sum, int32 query_sum2,
											 const uint8 **targets, const int32 *target_sum2s,
											 int count, int64 *results)
{
	(void) query_sum;  /* Unused in generic impl */
	for (int j = 0; j < count; j++)
	{
		/* Compute inner product */
		int64 dot = 0;
		for (int i = 0; i < dim; i++)
		{
			dot += (int64)query[i] * (int64)targets[j][i];
		}
		/* L2² = sum2_q + sum2_t - 2*dot */
		results[j] = (int64)query_sum2 + (int64)target_sum2s[j] - 2 * dot;
	}
}

#if defined(__x86_64__) && defined(__AVX2__)
	/*
	 * AVX-VNNI implementation using _mm256_dpbusd_epi32
	 * 
	 * VNNI instruction: vpdpbusd
	 * Computes: dst = src + Σ(u8(a[i]) * s8(b[i])) for 4 byte pairs per lane.
	 * 
	 * P2 optimization: query_sum is now passed in to avoid recomputing sum(q).
	 */
__attribute__((target("avx2,avxvnni")))
static void
Sq8L2SquaredDistanceWithSum2BatchVNNIImpl(int dim, const uint8 *query,
										  int32 query_sum, int32 query_sum2,
										  const uint8 **targets, const int32 *target_sum2s,
										  int count, int64 *results)
{
	int i;
	int64 sum_q = (int64) query_sum;  /* Use precomputed sum instead of computing */
	
	if (count <= 0)
		return;
	
	/*
	 * Use vpdpbusd (AVX-VNNI) to compute dot products efficiently.
	 *
	 * _mm256_dpbusd_epi32 computes sum(u8(a[i]) * s8(b[i])) in dword lanes.
	 * We need u8*u8, so we bias the target by 128:
	 *   dot = sum(q * t)
	 *   dp  = sum(q * (t - 128)) = sum(q*t) - 128*sum(q)
	 *   dot = dp + 128*sum(q)
	 *
	 * sum(q) is now passed in as query_sum, avoiding O(dim) recomputation.
	 */

	/* Process 4 targets at a time for best efficiency */
	if (count == 4)
	{
		const uint8 *t0 = targets[0];
		const uint8 *t1 = targets[1];
		const uint8 *t2 = targets[2];
		const uint8 *t3 = targets[3];
		__m256i bias = _mm256_set1_epi8((char) 0x80);
		__m256i acc0 = _mm256_setzero_si256();
		__m256i acc1 = _mm256_setzero_si256();
		__m256i acc2 = _mm256_setzero_si256();
		__m256i acc3 = _mm256_setzero_si256();
		int64 dp0, dp1, dp2, dp3;
		int64 dot0, dot1, dot2, dot3;

		i = 0;
		for (; i + 32 <= dim; i += 32)
		{
			__m256i vq = _mm256_loadu_si256((const __m256i *)(query + i));

			/* Avoid over-prefetching for small dimensions (e.g. dim=200) */
			if (i + 256 < dim)
			{
				_mm_prefetch((const char *)(t0 + i + 256), _MM_HINT_T0);
				_mm_prefetch((const char *)(t1 + i + 256), _MM_HINT_T0);
				_mm_prefetch((const char *)(t2 + i + 256), _MM_HINT_T0);
				_mm_prefetch((const char *)(t3 + i + 256), _MM_HINT_T0);
			}

			/* Target 0 */
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *)(t0 + i));
				vt = _mm256_xor_si256(vt, bias); /* t - 128 as signed byte */
				acc0 = _mm256_dpbusd_epi32(acc0, vq, vt);
			}
			/* Target 1 */
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *)(t1 + i));
				vt = _mm256_xor_si256(vt, bias);
				acc1 = _mm256_dpbusd_epi32(acc1, vq, vt);
			}
			/* Target 2 */
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *)(t2 + i));
				vt = _mm256_xor_si256(vt, bias);
				acc2 = _mm256_dpbusd_epi32(acc2, vq, vt);
			}
			/* Target 3 */
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *)(t3 + i));
				vt = _mm256_xor_si256(vt, bias);
				acc3 = _mm256_dpbusd_epi32(acc3, vq, vt);
			}
		}

		/* Reduce accumulators */
		dp0 = Sq8ReduceAddEpi32SignedAVX2(acc0);
		dp1 = Sq8ReduceAddEpi32SignedAVX2(acc1);
		dp2 = Sq8ReduceAddEpi32SignedAVX2(acc2);
		dp3 = Sq8ReduceAddEpi32SignedAVX2(acc3);
		
		/* Handle tail */
		for (; i < dim; i++)
		{
			int q = (int) query[i];
			dp0 += (int64) q * ((int64) (int) t0[i] - 128);
			dp1 += (int64) q * ((int64) (int) t1[i] - 128);
			dp2 += (int64) q * ((int64) (int) t2[i] - 128);
			dp3 += (int64) q * ((int64) (int) t3[i] - 128);
		}

		dot0 = dp0 + 128 * sum_q;
		dot1 = dp1 + 128 * sum_q;
		dot2 = dp2 + 128 * sum_q;
		dot3 = dp3 + 128 * sum_q;
		
		/* Compute L2² = sum2_q + sum2_t - 2*dot */
		results[0] = (int64)query_sum2 + (int64)target_sum2s[0] - 2 * dot0;
		results[1] = (int64)query_sum2 + (int64)target_sum2s[1] - 2 * dot1;
		results[2] = (int64)query_sum2 + (int64)target_sum2s[2] - 2 * dot2;
		results[3] = (int64)query_sum2 + (int64)target_sum2s[3] - 2 * dot3;
		return;
	}

	/* General case - use precomputed sum_q */
	{
		__m256i bias = _mm256_set1_epi8((char) 0x80);

		for (int j = 0; j < count; j++)
		{
			const uint8 *target = targets[j];
			__m256i acc = _mm256_setzero_si256();
			int64 dp;

			i = 0;
			for (; i + 32 <= dim; i += 32)
			{
				__m256i vq = _mm256_loadu_si256((const __m256i *)(query + i));
				__m256i vt = _mm256_loadu_si256((const __m256i *)(target + i));
				vt = _mm256_xor_si256(vt, bias);
				acc = _mm256_dpbusd_epi32(acc, vq, vt);
			}

			dp = Sq8ReduceAddEpi32SignedAVX2(acc);

			for (; i < dim; i++)
			{
				int q = (int) query[i];
				dp += (int64) q * ((int64) (int) target[i] - 128);
			}

			/* dot = dp + 128*sum(q) */
			{
				int64 dot = dp + 128 * sum_q;
				results[j] = (int64) query_sum2 + (int64) target_sum2s[j] - 2 * dot;
			}
		}
		return;
	}
}

/*
 * AVX2 implementation (fallback when VNNI not available)
 * Uses the same approach but without VNNI instructions
 */
__attribute__((target("avx2")))
static void
Sq8L2SquaredDistanceWithSum2BatchAVX2Impl(int dim, const uint8 *query,
										  int32 query_sum, int32 query_sum2,
										  const uint8 **targets, const int32 *target_sum2s,
										  int count, int64 *results)
{
	int i;
	(void) query_sum;  /* AVX2 non-VNNI doesn't need bias correction */
	
	if (count <= 0)
		return;
	
	/* Process 4 targets at a time for best efficiency */
	if (count == 4)
	{
		const uint8 *t0 = targets[0];
		const uint8 *t1 = targets[1];
		const uint8 *t2 = targets[2];
		const uint8 *t3 = targets[3];
		__m256i zero = _mm256_setzero_si256();
		__m256i acc0 = _mm256_setzero_si256();
		__m256i acc1 = _mm256_setzero_si256();
		__m256i acc2 = _mm256_setzero_si256();
		__m256i acc3 = _mm256_setzero_si256();
		int64 dot0 , dot1, dot2, dot3;
		
		i = 0;
		for (; i + 32 <= dim; i += 32)
		{
			__m256i vq = _mm256_loadu_si256((const __m256i *)(query + i));
			__m256i q_lo = _mm256_unpacklo_epi8(vq, zero);
			__m256i q_hi = _mm256_unpackhi_epi8(vq, zero);
			
			/* Target 0 */
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *)(t0 + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(q_lo, t_lo));
				acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(q_hi, t_hi));
			}
			/* Target 1 */
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *)(t1 + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(q_lo, t_lo));
				acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(q_hi, t_hi));
			}
			/* Target 2 */
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *)(t2 + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(q_lo, t_lo));
				acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(q_hi, t_hi));
			}
			/* Target 3 */
			{
				__m256i vt = _mm256_loadu_si256((const __m256i *)(t3 + i));
				__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
				__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
				acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(q_lo, t_lo));
				acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(q_hi, t_hi));
			}
		}
		
		/* Reduce accumulators */
		dot0 = (int64) Sq8ReduceAddEpi32AVX2(acc0);
		dot1 = (int64) Sq8ReduceAddEpi32AVX2(acc1);
		dot2 = (int64) Sq8ReduceAddEpi32AVX2(acc2);
		dot3 = (int64) Sq8ReduceAddEpi32AVX2(acc3);
		
		/* Handle tail */
		for (; i < dim; i++)
		{
			int q = (int)query[i];
			dot0 += (int64)q * (int64)t0[i];
			dot1 += (int64)q * (int64)t1[i];
			dot2 += (int64)q * (int64)t2[i];
			dot3 += (int64)q * (int64)t3[i];
		}
		
		/* Compute L2² = sum2_q + sum2_t - 2*dot */
		results[0] = (int64)query_sum2 + (int64)target_sum2s[0] - 2 * dot0;
		results[1] = (int64)query_sum2 + (int64)target_sum2s[1] - 2 * dot1;
		results[2] = (int64)query_sum2 + (int64)target_sum2s[2] - 2 * dot2;
		results[3] = (int64)query_sum2 + (int64)target_sum2s[3] - 2 * dot3;
		return;
	}
	
	/* General case */
	for (int j = 0; j < count; j++)
	{
		const uint8 *target = targets[j];
		__m256i zero = _mm256_setzero_si256();
		__m256i acc = _mm256_setzero_si256();
		int64 dot;
		
		i = 0;
		for (; i + 32 <= dim; i += 32)
		{
			__m256i vq = _mm256_loadu_si256((const __m256i *)(query + i));
			__m256i vt = _mm256_loadu_si256((const __m256i *)(target + i));
			__m256i q_lo = _mm256_unpacklo_epi8(vq, zero);
			__m256i q_hi = _mm256_unpackhi_epi8(vq, zero);
			__m256i t_lo = _mm256_unpacklo_epi8(vt, zero);
			__m256i t_hi = _mm256_unpackhi_epi8(vt, zero);
			acc = _mm256_add_epi32(acc, _mm256_madd_epi16(q_lo, t_lo));
			acc = _mm256_add_epi32(acc, _mm256_madd_epi16(q_hi, t_hi));
		}
		
		dot = (int64) Sq8ReduceAddEpi32AVX2(acc);
		
		/* Handle tail */
		for (; i < dim; i++)
		{
			dot += (int64)query[i] * (int64)target[i];
		}
		
		/* Compute L2² = sum2_q + sum2_t - 2*dot */
		results[j] = (int64)query_sum2 + (int64)target_sum2s[j] - 2 * dot;
	}
}
#endif /* AVX2 */

#if defined(__x86_64__)
__attribute__((target("avx512f,avx512bw,avx512vnni")))
static void
Sq8L2SquaredDistanceWithSum2BatchAVX512VNNIImpl(int dim, const uint8 *query,
												int32 query_sum, int32 query_sum2,
												const uint8 **targets, const int32 *target_sum2s,
												int count, int64 *results)
{
	int i;
	int64 sum_q = (int64) query_sum;  /* Use precomputed sum instead of computing */

	if (count <= 0)
		return;

	/*
	 * Use vpdpbusd (AVX512VNNI) to compute dot products efficiently.
	 *
	 * _mm512_dpbusd_epi32 computes sum(u8(a[i]) * s8(b[i])) in dword lanes.
	 * We need u8*u8, so we bias the target by 128:
	 *   dot = sum(q * t)
	 *   dp  = sum(q * (t - 128)) = sum(q*t) - 128*sum(q)
	 *   dot = dp + 128*sum(q)
	 *
	 * P2 optimization: sum(q) is now passed in as query_sum.
	 */

	/* Fast path for the common batch size */
	if (count == 4)
	{
		const uint8 *t0 = targets[0];
		const uint8 *t1 = targets[1];
		const uint8 *t2 = targets[2];
		const uint8 *t3 = targets[3];
		__m512i bias = _mm512_set1_epi8((char) 0x80);
		__m512i acc0 = _mm512_setzero_si512();
		__m512i acc1 = _mm512_setzero_si512();
		__m512i acc2 = _mm512_setzero_si512();
		__m512i acc3 = _mm512_setzero_si512();
		int64 dp0, dp1, dp2, dp3;

		i = 0;
		for (; i + 64 <= dim; i += 64)
		{
			__m512i vq = _mm512_loadu_si512((const void *) (query + i));

			/* Avoid over-prefetching for small dimensions (e.g. dim=200) */
			if (i + 512 < dim)
			{
				_mm_prefetch((const char *) (t0 + i + 512), _MM_HINT_T0);
				_mm_prefetch((const char *) (t1 + i + 512), _MM_HINT_T0);
				_mm_prefetch((const char *) (t2 + i + 512), _MM_HINT_T0);
				_mm_prefetch((const char *) (t3 + i + 512), _MM_HINT_T0);
			}

			{
				__m512i vt = _mm512_loadu_si512((const void *) (t0 + i));
				vt = _mm512_xor_si512(vt, bias);
				acc0 = _mm512_dpbusd_epi32(acc0, vq, vt);
			}
			{
				__m512i vt = _mm512_loadu_si512((const void *) (t1 + i));
				vt = _mm512_xor_si512(vt, bias);
				acc1 = _mm512_dpbusd_epi32(acc1, vq, vt);
			}
			{
				__m512i vt = _mm512_loadu_si512((const void *) (t2 + i));
				vt = _mm512_xor_si512(vt, bias);
				acc2 = _mm512_dpbusd_epi32(acc2, vq, vt);
			}
			{
				__m512i vt = _mm512_loadu_si512((const void *) (t3 + i));
				vt = _mm512_xor_si512(vt, bias);
				acc3 = _mm512_dpbusd_epi32(acc3, vq, vt);
			}
		}

		dp0 = Sq8ReduceAddEpi32SignedAVX512(acc0);
		dp1 = Sq8ReduceAddEpi32SignedAVX512(acc1);
		dp2 = Sq8ReduceAddEpi32SignedAVX512(acc2);
		dp3 = Sq8ReduceAddEpi32SignedAVX512(acc3);

		for (; i < dim; i++)
		{
			int q = (int) query[i];
			dp0 += (int64) q * ((int64) (int) t0[i] - 128);
			dp1 += (int64) q * ((int64) (int) t1[i] - 128);
			dp2 += (int64) q * ((int64) (int) t2[i] - 128);
			dp3 += (int64) q * ((int64) (int) t3[i] - 128);
		}

		{
			int64 dot0 = dp0 + 128 * sum_q;
			int64 dot1 = dp1 + 128 * sum_q;
			int64 dot2 = dp2 + 128 * sum_q;
			int64 dot3 = dp3 + 128 * sum_q;
			results[0] = (int64) query_sum2 + (int64) target_sum2s[0] - 2 * dot0;
			results[1] = (int64) query_sum2 + (int64) target_sum2s[1] - 2 * dot1;
			results[2] = (int64) query_sum2 + (int64) target_sum2s[2] - 2 * dot2;
			results[3] = (int64) query_sum2 + (int64) target_sum2s[3] - 2 * dot3;
		}
		return;
	}

	/* General case - use precomputed sum_q */
	{
		__m512i bias = _mm512_set1_epi8((char) 0x80);
		int j;

		for (j = 0; j < count; j++)
		{
			const uint8 *target = targets[j];
			__m512i acc = _mm512_setzero_si512();
			int64 dp;

			i = 0;
			for (; i + 64 <= dim; i += 64)
			{
				__m512i vq = _mm512_loadu_si512((const void *) (query + i));
				__m512i vt = _mm512_loadu_si512((const void *) (target + i));
				vt = _mm512_xor_si512(vt, bias);
				acc = _mm512_dpbusd_epi32(acc, vq, vt);
			}

			dp = Sq8ReduceAddEpi32SignedAVX512(acc);

			for (; i < dim; i++)
			{
				int q = (int) query[i];
				dp += (int64) q * ((int64) (int) target[i] - 128);
			}

			{
				int64 dot = dp + 128 * sum_q;
				results[j] = (int64) query_sum2 + (int64) target_sum2s[j] - 2 * dot;
			}
		}
		return;
	}
}

__attribute__((target("avx512f,avx512bw")))
static void
Sq8L2SquaredDistanceWithSum2BatchAVX512Impl(int dim, const uint8 *query,
											int32 query_sum, int32 query_sum2,
											const uint8 **targets, const int32 *target_sum2s,
											int count, int64 *results)
{
	int i;
	int j;
	__m512i zero = _mm512_setzero_si512();
	(void) query_sum;  /* AVX512 non-VNNI uses direct dot product */

	if (count <= 0)
		return;

	if (count == 4)
	{
		const uint8 *t0 = targets[0];
		const uint8 *t1 = targets[1];
		const uint8 *t2 = targets[2];
		const uint8 *t3 = targets[3];
		__m512i acc0 = _mm512_setzero_si512();
		__m512i acc1 = _mm512_setzero_si512();
		__m512i acc2 = _mm512_setzero_si512();
		__m512i acc3 = _mm512_setzero_si512();
		int64 dot0;
		int64 dot1;
		int64 dot2;
		int64 dot3;

		i = 0;
		for (; i + 64 <= dim; i += 64)
		{
			__m512i vq = _mm512_loadu_si512((const void *)(query + i));
			__m512i q_lo = _mm512_unpacklo_epi8(vq, zero);
			__m512i q_hi = _mm512_unpackhi_epi8(vq, zero);

			/* Avoid over-prefetching for small dimensions (e.g. dim=200) */
			if (i + 256 < dim)
			{
				_mm_prefetch((const char *)(t0 + i + 256), _MM_HINT_T0);
				_mm_prefetch((const char *)(t1 + i + 256), _MM_HINT_T0);
				_mm_prefetch((const char *)(t2 + i + 256), _MM_HINT_T0);
				_mm_prefetch((const char *)(t3 + i + 256), _MM_HINT_T0);
			}

			{
				__m512i vt = _mm512_loadu_si512((const void *)(t0 + i));
				__m512i t_lo = _mm512_unpacklo_epi8(vt, zero);
				__m512i t_hi = _mm512_unpackhi_epi8(vt, zero);
				acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(q_lo, t_lo));
				acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(q_hi, t_hi));
			}
			{
				__m512i vt = _mm512_loadu_si512((const void *)(t1 + i));
				__m512i t_lo = _mm512_unpacklo_epi8(vt, zero);
				__m512i t_hi = _mm512_unpackhi_epi8(vt, zero);
				acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(q_lo, t_lo));
				acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(q_hi, t_hi));
			}
			{
				__m512i vt = _mm512_loadu_si512((const void *)(t2 + i));
				__m512i t_lo = _mm512_unpacklo_epi8(vt, zero);
				__m512i t_hi = _mm512_unpackhi_epi8(vt, zero);
				acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(q_lo, t_lo));
				acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(q_hi, t_hi));
			}
			{
				__m512i vt = _mm512_loadu_si512((const void *)(t3 + i));
				__m512i t_lo = _mm512_unpacklo_epi8(vt, zero);
				__m512i t_hi = _mm512_unpackhi_epi8(vt, zero);
				acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(q_lo, t_lo));
				acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(q_hi, t_hi));
			}
		}

		dot0 = (int64) _mm512_reduce_add_epi32(acc0);
		dot1 = (int64) _mm512_reduce_add_epi32(acc1);
		dot2 = (int64) _mm512_reduce_add_epi32(acc2);
		dot3 = (int64) _mm512_reduce_add_epi32(acc3);

		for (; i < dim; i++)
		{
			int q = (int) query[i];
			dot0 += (int64) q * (int64) t0[i];
			dot1 += (int64) q * (int64) t1[i];
			dot2 += (int64) q * (int64) t2[i];
			dot3 += (int64) q * (int64) t3[i];
		}

		results[0] = (int64)query_sum2 + (int64)target_sum2s[0] - 2 * dot0;
		results[1] = (int64)query_sum2 + (int64)target_sum2s[1] - 2 * dot1;
		results[2] = (int64)query_sum2 + (int64)target_sum2s[2] - 2 * dot2;
		results[3] = (int64)query_sum2 + (int64)target_sum2s[3] - 2 * dot3;
		return;
	}

	for (j = 0; j < count; j++)
	{
		const uint8 *target = targets[j];
		__m512i acc = _mm512_setzero_si512();
		int64 dot;

		i = 0;
		for (; i + 64 <= dim; i += 64)
		{
			__m512i vq = _mm512_loadu_si512((const void *)(query + i));
			__m512i vt = _mm512_loadu_si512((const void *)(target + i));
			__m512i q_lo = _mm512_unpacklo_epi8(vq, zero);
			__m512i q_hi = _mm512_unpackhi_epi8(vq, zero);
			__m512i t_lo = _mm512_unpacklo_epi8(vt, zero);
			__m512i t_hi = _mm512_unpackhi_epi8(vt, zero);
			/* Avoid over-prefetching for small dimensions (e.g. dim=200) */
			if (i + 256 < dim)
				_mm_prefetch((const char *)(target + i + 256), _MM_HINT_T0);
			acc = _mm512_add_epi32(acc, _mm512_madd_epi16(q_lo, t_lo));
			acc = _mm512_add_epi32(acc, _mm512_madd_epi16(q_hi, t_hi));
		}

		dot = (int64) _mm512_reduce_add_epi32(acc);

		for (; i < dim; i++)
			dot += (int64)query[i] * (int64)target[i];

		results[j] = (int64)query_sum2 + (int64)target_sum2s[j] - 2 * dot;
	}
}
#endif /* __x86_64__ */

/*
 * Public API for L2 squared distance with precomputed sum2
 */
void
Sq8L2SquaredDistanceWithSum2Batch(int dim, const uint8 *query, int32 query_sum,
								  int32 query_sum2, const uint8 **targets,
								  const int32 *target_sum2s, int count, int64 *results)
{
	if (count <= 0)
		return;
	if (Sq8L2SquaredDistanceWithSum2BatchImpl)
		Sq8L2SquaredDistanceWithSum2BatchImpl(dim, query, query_sum, query_sum2, targets, target_sum2s, count, results);
	else
		Sq8L2SquaredDistanceWithSum2BatchGenericImpl(dim, query, query_sum, query_sum2, targets, target_sum2s, count, results);
}
