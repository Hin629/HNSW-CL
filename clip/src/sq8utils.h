#ifndef SQ8UTILS_H
#define SQ8UTILS_H

#include <stdint.h>

/*
 * SQ8 SIMD-accelerated distance functions
 * 
 * These functions compute distances between SQ8-quantized vectors.
 * Implementation uses AVX2/AVX512 when available, with generic fallback.
 */

/* Function pointers - set at initialization based on CPU features */
extern float (*Sq8L2SquaredDistance)(int dim, const uint8 *ax, const uint8 *bx,
									  float lower_bound_a, float diff_a,
									  float lower_bound_b, float diff_b);

extern float (*Sq8InnerProduct)(int dim, const uint8 *ax, const uint8 *bx,
								 float lower_bound_a, float diff_a,
								 float lower_bound_b, float diff_b);

extern double (*Sq8CosineSimilarity)(int dim, const uint8 *ax, const uint8 *bx,
									  float lower_bound_a, float diff_a,
									  float lower_bound_b, float diff_b);

extern float (*Sq8L1Distance)(int dim, const uint8 *ax, const uint8 *bx,
							   float lower_bound_a, float diff_a,
							   float lower_bound_b, float diff_b);

/* Fast distance for same quantization parameters (used within index) */
extern int64 (*Sq8L2SquaredDistanceSameQuant)(int dim, const uint8 *ax, const uint8 *bx);
extern int64 (*Sq8InnerProductSameQuant)(int dim, const uint8 *ax, const uint8 *bx);

/* Initialize the function pointers based on CPU features */
void Sq8utilsInit(void);

/* Generic implementations */
float Sq8L2SquaredDistanceGeneric(int dim, const uint8 *ax, const uint8 *bx,
								   float lower_bound_a, float diff_a,
								   float lower_bound_b, float diff_b);

float Sq8InnerProductGeneric(int dim, const uint8 *ax, const uint8 *bx,
							  float lower_bound_a, float diff_a,
							  float lower_bound_b, float diff_b);

double Sq8CosineSimilarityGeneric(int dim, const uint8 *ax, const uint8 *bx,
								   float lower_bound_a, float diff_a,
								   float lower_bound_b, float diff_b);

float Sq8L1DistanceGeneric(int dim, const uint8 *ax, const uint8 *bx,
							float lower_bound_a, float diff_a,
							float lower_bound_b, float diff_b);

/*
 * Global SQ8 Quantization Support
 * 
 * These functions support using a single global set of quantization parameters
 * (lower_bound, diff) for all vectors in an HNSW index, rather than per-vector
 * parameters. This reduces storage (no per-vector header) and enables faster
 * distance computation using integer-only SameQuant functions.
 * 
 * The global parameters are computed from a sample of vectors during index build
 * and stored in the HNSW meta page for persistence.
 */

/* 
 * Compute global quantization parameters from sampled float vectors.
 * Returns the computed lower_bound and diff that cover the min/max range.
 * 
 * @param samples     Array of pointers to float arrays (each is a vector)
 * @param nsamples    Number of sample vectors
 * @param dim         Dimension of each vector
 * @param lower_bound Output: computed global lower bound
 * @param diff        Output: computed global diff (max - min)
 */
void Sq8ComputeGlobalQuantParams(float **samples, int nsamples, int dim,
                                  float *lower_bound, float *diff);

/*
 * Encode a float vector using global quantization parameters.
 * The result codes can be used directly with SameQuant distance functions.
 * 
 * @param values       Input float vector
 * @param dim          Dimension
 * @param lower_bound  Global lower bound
 * @param diff         Global diff
 * @param codes        Output: encoded uint8 codes
 */
void Sq8EncodeWithGlobalParams(const float *values, int dim,
                                float lower_bound, float diff, uint8 *codes);

/*
 * Re-encode an SQ8 vector from per-vector params to global params without
 * decoding to floats.
 *
 * local decode:  v = lower_bound_local + (diff_local/255) * code_local
 * global encode: code_global = round((v - lower_bound_global) * 255 / diff_global)
 *
 * Optionally returns sum / sum2 of output codes (for IP/Cosine fast paths).
 */
void Sq8ReencodeToGlobalParams(const uint8 *local_codes, int dim,
                               float lower_bound_local, float diff_local,
                               float lower_bound_global, float diff_global,
                               uint8 *out_codes,
                               int32 *out_sum, int32 *out_sum2);

/*
 * Convert distance from SameQuant integer result to proper float distance.
 * For L2 squared distance: result = int_dist * (diff/255)^2
 * 
 * @param int_dist    Integer distance from SameQuant function
 * @param diff        Global diff parameter
 * @return            Actual float distance
 */
static inline float Sq8ConvertSameQuantL2Dist(int64 int_dist, float diff)
{
    float scale = diff / 255.0f;
    return (float)int_dist * scale * scale;
}

/*
 * Convert inner product from SameQuant integer result to proper float value.
 * 
 * @param int_ip      Integer inner product from SameQuant function
 * @param dim         Dimension
 * @param lower_bound Global lower bound  
 * @param diff        Global diff parameter
 * @return            Actual float inner product
 */
static inline float Sq8ConvertSameQuantIP(int64 int_ip, int dim,
                                           float lower_bound, float diff)
{
    float scale = diff / 255.0f;
    /* 
     * Full formula: dot = scale^2 * Sab + scale*lb*(Sa + Sb) + dim*lb^2
     * For SameQuant, we use simplified: scale^2 * int_ip
     * (This is an approximation when lower_bound != 0)
     */
    return (float)int_ip * scale * scale;
}

/*
 * Batch distance computation functions for SQ8 vectors
 * These are optimized for computing distances between one query and multiple targets
 */

/* Batch L2 squared distance (same quantization) */
void Sq8L2SquaredDistanceSameQuantBatch(int dim, const uint8 *query, 
                                          const uint8 **targets, int count, 
                                          int64 *results);

/* Batch inner product (same quantization) */
void Sq8InnerProductSameQuantBatch(int dim, const uint8 *query,
                                    const uint8 **targets, int count,
                                    int64 *results);

/*
 * Optimized L2 squared distance batch using precomputed sums.
 * Uses the formula: L2² = sum2_q + sum2_t - 2*dot_product
 * where sum2 = Σ(code²) is precomputed for both query and targets.
 * 
 * This is faster than computing (a-b)² directly because:
 * 1. Query's sum2 is computed once and reused for all targets
 * 2. Target's sum2 is precomputed and stored in the index
 * 3. Only the inner product needs to be computed per target
 * 
 * @param dim           Dimension of vectors
 * @param query         Query vector codes
 * @param query_sum     Precomputed sum(query_codes) - used for VNNI bias correction
 * @param query_sum2    Precomputed sum(query_codes^2)
 * @param targets       Array of target vector code pointers
 * @param target_sum2s  Array of precomputed sum(target_codes^2)
 * @param count         Number of targets
 * @param results       Output array for L2² results (integer space)
 */
void Sq8L2SquaredDistanceWithSum2Batch(int dim, const uint8 *query, int32 query_sum,
                                        int32 query_sum2, const uint8 **targets,
                                        const int32 *target_sum2s, int count, int64 *results);

/*
 * Single-point L2 squared distance using sum2 optimization.
 * Uses formula: L2² = sum2_a + sum2_b - 2*dot(a,b)
 * This avoids computing (a-b)² directly and can use VNNI for dot product.
 * 
 * @param dim       Dimension of vectors
 * @param ax        First vector codes
 * @param sum_a     Precomputed sum(ax) - used for VNNI bias correction
 * @param sum2_a    Precomputed sum(ax^2)
 * @param bx        Second vector codes
 * @param sum2_b    Precomputed sum(bx^2)
 * @return          L2² distance in integer code space
 */
int64 Sq8L2SquaredDistanceWithSum2(int dim, const uint8 *ax,
                                    int32 sum_a, int32 sum2_a,
                                    const uint8 *bx, int32 sum2_b);

/* Optimized IP distance with precomputed sums */
float Sq8InnerProductWithSums(int64 dot_product, int32 sum_q, int32 sum_t,
                               int dim, float lower_bound, float scale);

/* Optimized Cosine distance with precomputed sums */
float Sq8CosineDistanceWithSums(int64 dot_product, int32 sum_q, int32 sum2_q,
                                 int32 sum_t, int32 sum2_t, int dim,
                                 float lower_bound, float scale);

#endif /* SQ8UTILS_H */
