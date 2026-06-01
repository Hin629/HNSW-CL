/*
 * sq8vec.c - SQ8 (Scalar Quantization 8-bit) Vector Type
 *
 * This implements a PostgreSQL type for vectors quantized to 8-bit codes.
 * Similar to halfvec but uses SQ8 quantization for better compression.
 * Each vector stores its own quantization parameters (lower_bound, diff).
 */
#include "postgres.h"

#include <math.h>

#include "bitvec.h"
#include "catalog/pg_type.h"
#include "common/shortest_dec.h"
#include "fmgr.h"
#include "hnsw.h"
#include "ivfflat.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "port.h"
#include "sparsevec.h"
#include "sq8utils.h"
#include "sq8vec.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/float.h"
#include "utils/lsyscache.h"
#include "utils/numeric.h"
#include "vector.h"

#define STATE_DIMS(x) (ARR_DIMS(x)[0] - 1)
#define CreateStateDatums(dim) palloc(sizeof(Datum) * (dim + 1))

static void Sq8ComputeQuantizationParams(const float *values, int dim, float *lower_bound, float *diff);
static void Sq8EncodeVector(const float *values, int dim, float lower_bound, float diff, uint8 *codes);

static Size
Sq8vecItemSize(int dimensions)
{
	return SQ8VEC_SIZE(dimensions);
}

static void
Sq8vecUpdateCenter(Pointer v, int dimensions, float *x)
{
	Sq8Vector   *vec = (Sq8Vector *) v;

	SET_VARSIZE(vec, SQ8VEC_SIZE(dimensions));
	vec->dim = dimensions;
	vec->unused = 0;

	Sq8ComputeQuantizationParams(x, dimensions, &vec->lower_bound, &vec->diff);
	Sq8EncodeVector(x, dimensions, vec->lower_bound, vec->diff, vec->x);
}

static void
Sq8vecSumCenter(Pointer v, float *x)
{
	Sq8Vector   *vec = (Sq8Vector *) v;
	int			dim = vec->dim;
	float		scale = vec->diff / 255.0f;

	for (int i = 0; i < dim; i++)
		x[i] += (float) vec->x[i] * scale + vec->lower_bound;
}

/*
 * Ensure same dimensions
 */
static inline void
CheckDims(Sq8Vector *a, Sq8Vector *b)
{
	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different sq8vec dimensions %d and %d", a->dim, b->dim)));
}

/*
 * Ensure expected dimensions
 */
static inline void
CheckExpectedDim(int32 typmod, int dim)
{
	if (typmod != -1 && typmod != dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("expected %d dimensions, not %d", typmod, dim)));
}

/*
 * Ensure valid dimensions
 */
static inline void
CheckDim(int dim)
{
	if (dim < 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("sq8vec must have at least 1 dimension")));

	if (dim > SQ8VEC_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("sq8vec cannot have more than %d dimensions", SQ8VEC_MAX_DIM)));
}

/*
 * Allocate and initialize a new SQ8 vector
 */
static Sq8Vector *
InitSq8Vector(int dim)
{
	Sq8Vector *result;
	int size;

	size = SQ8VEC_SIZE(dim);
	result = (Sq8Vector *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = dim;
	result->unused = 0;
	result->lower_bound = 0.0f;
	result->diff = 1.0f;

	return result;
}

/*
 * Allocate and initialize a new SQ8 index tuple (codes-only, no lb/diff)
 * Used for storing vectors in HNSW index with global quantization.
 */
Sq8IndexTuple *
InitSq8IndexTuple(int dim)
{
	Sq8IndexTuple *result;
	int size;

	size = SQ8_INDEX_TUPLE_SIZE(dim);
	result = (Sq8IndexTuple *) palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = dim;
	result->unused = 0;
	result->sum = 0;
	result->sum2 = 0;

	return result;
}

/*
 * Compute quantization parameters from float values
 * Uses min/max to determine lower_bound and diff (range)
 */
static void
Sq8ComputeQuantizationParams(const float *values, int dim, float *lower_bound, float *diff)
{
	float min_val = values[0];
	float max_val = values[0];

	for (int i = 1; i < dim; i++)
	{
		if (values[i] < min_val)
			min_val = values[i];
		if (values[i] > max_val)
			max_val = values[i];
	}

	*lower_bound = min_val;
	*diff = max_val - min_val;

	/* Avoid division by zero - if all values are same, use small diff */
	if (*diff < 1e-10f)
		*diff = 1e-10f;
}

/*
 * Encode float values to SQ8 codes
 */
static void
Sq8EncodeVector(const float *values, int dim, float lower_bound, float diff, uint8 *codes)
{
	float scale = 255.0f / diff;
	for (int i = 0; i < dim; i++)
	{
		float codef = (values[i] - lower_bound) * scale;
		if (codef < 0.0f)
			codef = 0.0f;
		else if (codef > 255.0f)
			codef = 255.0f;
		codes[i] = (uint8) (codef + 0.5f);
	}
}

/*
 * Decode SQ8 codes to float values
 */
static void
Sq8DecodeVector(const uint8 *codes, int dim, float lower_bound, float diff, float *values)
{
	float scale = diff / 255.0f;
	for (int i = 0; i < dim; i++)
	{
		values[i] = (float) codes[i] * scale + lower_bound;
	}
}

/*
 * Check for whitespace
 */
static inline bool
sq8vec_isspace(char ch)
{
	if (ch == ' ' ||
		ch == '\t' ||
		ch == '\n' ||
		ch == '\r' ||
		ch == '\v' ||
		ch == '\f')
		return true;
	return false;
}

/*
 * Convert textual representation to internal representation
 * Input format: [v1,v2,...,vn]
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_in);
Datum
sq8vec_in(PG_FUNCTION_ARGS)
{
	char *lit = PG_GETARG_CSTRING(0);
	int32 typmod = PG_GETARG_INT32(2);
	float x[SQ8VEC_MAX_DIM];
	int dim = 0;
	char *pt = lit;
	Sq8Vector *result;

	while (sq8vec_isspace(*pt))
		pt++;

	if (*pt != '[')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid input syntax for type sq8vec: \"%s\"", lit),
				 errdetail("Vector contents must start with \"[\".")));

	pt++;

	while (sq8vec_isspace(*pt))
		pt++;

	if (*pt == ']')
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("sq8vec must have at least 1 dimension")));

	for (;;)
	{
		float val;
		char *stringEnd;

		if (dim == SQ8VEC_MAX_DIM)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("sq8vec cannot have more than %d dimensions", SQ8VEC_MAX_DIM)));

		while (sq8vec_isspace(*pt))
			pt++;

		if (*pt == '\0')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type sq8vec: \"%s\"", lit)));

		errno = 0;
		val = strtof(pt, &stringEnd);

		if (stringEnd == pt)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type sq8vec: \"%s\"", lit)));

		if (isnan(val))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("NaN not allowed in sq8vec")));

		if (isinf(val))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("infinite value not allowed in sq8vec")));

		x[dim++] = val;

		pt = stringEnd;

		while (sq8vec_isspace(*pt))
			pt++;

		if (*pt == ',')
			pt++;
		else if (*pt == ']')
		{
			pt++;
			break;
		}
		else
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("invalid input syntax for type sq8vec: \"%s\"", lit)));
	}

	while (sq8vec_isspace(*pt))
		pt++;

	if (*pt != '\0')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid input syntax for type sq8vec: \"%s\"", lit),
				 errdetail("Junk after closing right brace.")));

	CheckDim(dim);
	CheckExpectedDim(typmod, dim);

	result = InitSq8Vector(dim);
	Sq8ComputeQuantizationParams(x, dim, &result->lower_bound, &result->diff);
	Sq8EncodeVector(x, dim, result->lower_bound, result->diff, result->x);

	PG_RETURN_POINTER(result);
}

#define AppendChar(ptr, c) (*(ptr)++ = (c))
#define AppendFloat(ptr, f) ((ptr) += float_to_shortest_decimal_bufn((f), (ptr)))

/*
 * Convert internal representation to textual representation
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_out);
Datum
sq8vec_out(PG_FUNCTION_ARGS)
{
	Sq8Vector *vector = PG_GETARG_SQ8VEC_P(0);
	int dim = vector->dim;
	char *buf;
	char *ptr;
	float scale = vector->diff / 255.0f;

	buf = (char *) palloc(FLOAT_SHORTEST_DECIMAL_LEN * dim + 2);
	ptr = buf;

	AppendChar(ptr, '[');

	for (int i = 0; i < dim; i++)
	{
		float val;
		
		if (i > 0)
			AppendChar(ptr, ',');

		/* Decode and output as float */
		val = (float) vector->x[i] * scale + vector->lower_bound;
		AppendFloat(ptr, val);
	}

	AppendChar(ptr, ']');
	*ptr = '\0';

	PG_FREE_IF_COPY(vector, 0);
	PG_RETURN_CSTRING(buf);
}

/*
 * Convert type modifier
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_typmod_in);
Datum
sq8vec_typmod_in(PG_FUNCTION_ARGS)
{
	ArrayType *ta = PG_GETARG_ARRAYTYPE_P(0);
	int32 *tl;
	int n;

	tl = ArrayGetIntegerTypmods(ta, &n);

	if (n != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid type modifier")));

	if (*tl < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dimensions for type sq8vec must be at least 1")));

	if (*tl > SQ8VEC_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dimensions for type sq8vec cannot exceed %d", SQ8VEC_MAX_DIM)));

	PG_RETURN_INT32(*tl);
}

/*
 * Convert external binary representation to internal representation
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_recv);
Datum
sq8vec_recv(PG_FUNCTION_ARGS)
{
	StringInfo buf = (StringInfo) PG_GETARG_POINTER(0);
	int32 typmod = PG_GETARG_INT32(2);
	Sq8Vector *result;
	int16 dim;
	int16 unused;
	float lower_bound;
	float diff;

	dim = pq_getmsgint(buf, sizeof(int16));
	unused = pq_getmsgint(buf, sizeof(int16));
	lower_bound = pq_getmsgfloat4(buf);
	diff = pq_getmsgfloat4(buf);

	CheckDim(dim);
	CheckExpectedDim(typmod, dim);

	if (unused != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("expected unused to be 0, not %d", unused)));

	result = InitSq8Vector(dim);
	result->lower_bound = lower_bound;
	result->diff = diff;

	for (int i = 0; i < dim; i++)
	{
		result->x[i] = pq_getmsgint(buf, sizeof(uint8));
	}

	PG_RETURN_POINTER(result);
}

/*
 * Convert internal representation to the external binary representation
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_send);
Datum
sq8vec_send(PG_FUNCTION_ARGS)
{
	Sq8Vector *vec = PG_GETARG_SQ8VEC_P(0);
	StringInfoData buf;

	pq_begintypsend(&buf);
	pq_sendint(&buf, vec->dim, sizeof(int16));
	pq_sendint(&buf, vec->unused, sizeof(int16));
	pq_sendfloat4(&buf, vec->lower_bound);
	pq_sendfloat4(&buf, vec->diff);
	for (int i = 0; i < vec->dim; i++)
		pq_sendint(&buf, vec->x[i], sizeof(uint8));

	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

/*
 * Convert sq8vec to sq8vec (for type modifier checking)
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec);
Datum
sq8vec(PG_FUNCTION_ARGS)
{
	Sq8Vector *vec = PG_GETARG_SQ8VEC_P(0);
	int32 typmod = PG_GETARG_INT32(1);

	CheckExpectedDim(typmod, vec->dim);

	PG_RETURN_POINTER(vec);
}

/*
 * Convert array to sq8vec
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(array_to_sq8vec);
Datum
array_to_sq8vec(PG_FUNCTION_ARGS)
{
	ArrayType *array = PG_GETARG_ARRAYTYPE_P(0);
	int32 typmod = PG_GETARG_INT32(1);
	Sq8Vector *result;
	int16 typlen;
	bool typbyval;
	char typalign;
	Datum *elemsp;
	int nelemsp;
	float *values;

	if (ARR_NDIM(array) > 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("array must be 1-D")));

	if (ARR_HASNULL(array) && array_contains_nulls(array))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("array must not contain nulls")));

	get_typlenbyvalalign(ARR_ELEMTYPE(array), &typlen, &typbyval, &typalign);
	deconstruct_array(array, ARR_ELEMTYPE(array), typlen, typbyval, typalign, &elemsp, NULL, &nelemsp);

	CheckDim(nelemsp);
	CheckExpectedDim(typmod, nelemsp);

	values = palloc(sizeof(float) * nelemsp);

	if (ARR_ELEMTYPE(array) == INT4OID)
	{
		for (int i = 0; i < nelemsp; i++)
			values[i] = (float) DatumGetInt32(elemsp[i]);
	}
	else if (ARR_ELEMTYPE(array) == FLOAT8OID)
	{
		for (int i = 0; i < nelemsp; i++)
			values[i] = (float) DatumGetFloat8(elemsp[i]);
	}
	else if (ARR_ELEMTYPE(array) == FLOAT4OID)
	{
		for (int i = 0; i < nelemsp; i++)
			values[i] = DatumGetFloat4(elemsp[i]);
	}
	else if (ARR_ELEMTYPE(array) == NUMERICOID)
	{
		for (int i = 0; i < nelemsp; i++)
			values[i] = DatumGetFloat4(DirectFunctionCall1(numeric_float4, elemsp[i]));
	}
	else
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("unsupported array type")));
	}

	pfree(elemsp);

	result = InitSq8Vector(nelemsp);
	Sq8ComputeQuantizationParams(values, nelemsp, &result->lower_bound, &result->diff);
	Sq8EncodeVector(values, nelemsp, result->lower_bound, result->diff, result->x);

	pfree(values);

	PG_RETURN_POINTER(result);
}

/*
 * Convert sq8vec to float4[]
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_to_float4);
Datum
sq8vec_to_float4(PG_FUNCTION_ARGS)
{
	Sq8Vector *vec = PG_GETARG_SQ8VEC_P(0);
	Datum *datums;
	ArrayType *result;

	datums = (Datum *) palloc(sizeof(Datum) * vec->dim);

	for (int i = 0; i < vec->dim; i++)
	{
		float val = (vec->x[i] / 255.0f) * vec->diff + vec->lower_bound;
		datums[i] = Float4GetDatum(val);
	}

	result = construct_array(datums, vec->dim, FLOAT4OID, sizeof(float4), true, TYPALIGN_INT);

	pfree(datums);

	PG_RETURN_POINTER(result);
}

/*
 * Convert vector to sq8vec
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(vector_to_sq8vec);
Datum
vector_to_sq8vec(PG_FUNCTION_ARGS)
{
	Vector *vec = PG_GETARG_VECTOR_P(0);
	int32 typmod = PG_GETARG_INT32(1);
	Sq8Vector *result;

	CheckDim(vec->dim);
	CheckExpectedDim(typmod, vec->dim);

	result = InitSq8Vector(vec->dim);
	Sq8ComputeQuantizationParams(vec->x, vec->dim, &result->lower_bound, &result->diff);
	Sq8EncodeVector(vec->x, vec->dim, result->lower_bound, result->diff, result->x);

	PG_RETURN_POINTER(result);
}

/*
 * Convert sq8vec to vector
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_to_vector);
Datum
sq8vec_to_vector(PG_FUNCTION_ARGS)
{
	Sq8Vector *vec = PG_GETARG_SQ8VEC_P(0);
	int32 typmod = PG_GETARG_INT32(1);
	Vector *result;

	CheckExpectedDim(typmod, vec->dim);

	result = InitVector(vec->dim);
	Sq8DecodeVector(vec->x, vec->dim, vec->lower_bound, vec->diff, result->x);

	PG_RETURN_POINTER(result);
}

/*
 * Get the L2 distance between sq8 vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_l2_distance);
Datum
sq8vec_l2_distance(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);
	float dist_sq;

	CheckDims(a, b);

	dist_sq = Sq8L2SquaredDistance(a->dim, a->x, b->x,
										  a->lower_bound, a->diff,
										  b->lower_bound, b->diff);

	PG_RETURN_FLOAT8(sqrt((double) dist_sq));
}

/*
 * Get the L2 squared distance between sq8 vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_l2_squared_distance);
Datum
sq8vec_l2_squared_distance(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);
	float dist_sq;

	CheckDims(a, b);

	dist_sq = Sq8L2SquaredDistance(a->dim, a->x, b->x,
										  a->lower_bound, a->diff,
										  b->lower_bound, b->diff);

	PG_RETURN_FLOAT8((double) dist_sq);
}

/*
 * Get the inner product of two sq8 vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_inner_product);
Datum
sq8vec_inner_product(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);
	float ip;

	CheckDims(a, b);

	ip = Sq8InnerProduct(a->dim, a->x, b->x,
								a->lower_bound, a->diff,
								b->lower_bound, b->diff);

	PG_RETURN_FLOAT8((double) ip);
}

/*
 * Get the negative inner product of two sq8 vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_negative_inner_product);
Datum
sq8vec_negative_inner_product(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);
	float ip;

	CheckDims(a, b);

	ip = Sq8InnerProduct(a->dim, a->x, b->x,
								a->lower_bound, a->diff,
								b->lower_bound, b->diff);

	PG_RETURN_FLOAT8((double) -ip);
}

/*
 * Get the cosine distance between two sq8 vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_cosine_distance);
Datum
sq8vec_cosine_distance(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);
	double similarity;

	CheckDims(a, b);

	similarity = Sq8CosineSimilarity(a->dim, a->x, b->x,
									  a->lower_bound, a->diff,
									  b->lower_bound, b->diff);

	if (similarity > 1)
		similarity = 1;
	else if (similarity < -1)
		similarity = -1;

	PG_RETURN_FLOAT8(1 - similarity);
}

/*
 * Get the L1 distance between two sq8 vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_l1_distance);
Datum
sq8vec_l1_distance(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);
	float dist;

	CheckDims(a, b);

	dist = Sq8L1Distance(a->dim, a->x, b->x,
								a->lower_bound, a->diff,
								b->lower_bound, b->diff);

	PG_RETURN_FLOAT8((double) dist);
}

/*
 * Get the dimensions of a sq8 vector
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_vector_dims);
Datum
sq8vec_vector_dims(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);

	PG_RETURN_INT32(a->dim);
}

/*
 * Get the L2 norm of a sq8 vector
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_l2_norm);
Datum
sq8vec_l2_norm(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	int64 sum = 0;
	int64 sum2 = 0;
	double scale = (double) a->diff / 255.0;
	double lb = (double) a->lower_bound;

	for (int i = 0; i < a->dim; i++)
	{
		int c = (int) a->x[i];
		sum += c;
		sum2 += (int64) c * (int64) c;
	}

	/* ||a||^2 = scale^2*sum(code^2) + 2*scale*lb*sum(code) + dim*lb^2 */
	{
		double norm2 = (scale * scale) * (double) sum2 + (2.0 * scale * lb) * (double) sum + (double) a->dim * lb * lb;
		PG_RETURN_FLOAT8(sqrt(norm2));
	}
}

/*
 * Normalize a sq8 vector with the L2 norm
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_l2_normalize);
Datum
sq8vec_l2_normalize(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *result;
	float *values;
	double norm = 0.0;

	/* Decode to float values */
	values = palloc(sizeof(float) * a->dim);
	Sq8DecodeVector(a->x, a->dim, a->lower_bound, a->diff, values);

	/* Compute norm */
	for (int i = 0; i < a->dim; i++)
		norm += (double)values[i] * (double)values[i];
	norm = sqrt(norm);

	/* Normalize */
	if (norm > 0)
	{
		for (int i = 0; i < a->dim; i++)
			values[i] = (float)(values[i] / norm);
	}

	/* Re-encode */
	result = InitSq8Vector(a->dim);
	Sq8ComputeQuantizationParams(values, a->dim, &result->lower_bound, &result->diff);
	Sq8EncodeVector(values, a->dim, result->lower_bound, result->diff, result->x);

	pfree(values);

	PG_RETURN_POINTER(result);
}

/*
 * Add sq8 vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_add);
Datum
sq8vec_add(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);
	Sq8Vector *result;
	float *values;
	float scale_a;
	float scale_b;

	CheckDims(a, b);

	values = palloc(sizeof(float) * a->dim);
	scale_a = a->diff / 255.0f;
	scale_b = b->diff / 255.0f;

	for (int i = 0; i < a->dim; i++)
	{
		float va = (float) a->x[i] * scale_a + a->lower_bound;
		float vb = (float) b->x[i] * scale_b + b->lower_bound;
		values[i] = va + vb;
	}

	result = InitSq8Vector(a->dim);
	Sq8ComputeQuantizationParams(values, a->dim, &result->lower_bound, &result->diff);
	Sq8EncodeVector(values, a->dim, result->lower_bound, result->diff, result->x);

	pfree(values);

	PG_RETURN_POINTER(result);
}

/*
 * Subtract sq8 vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_sub);
Datum
sq8vec_sub(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);
	Sq8Vector *result;
	float *values;
	float scale_a;
	float scale_b;

	CheckDims(a, b);

	values = palloc(sizeof(float) * a->dim);
	scale_a = a->diff / 255.0f;
	scale_b = b->diff / 255.0f;

	for (int i = 0; i < a->dim; i++)
	{
		float va = (float) a->x[i] * scale_a + a->lower_bound;
		float vb = (float) b->x[i] * scale_b + b->lower_bound;
		values[i] = va - vb;
	}

	result = InitSq8Vector(a->dim);
	Sq8ComputeQuantizationParams(values, a->dim, &result->lower_bound, &result->diff);
	Sq8EncodeVector(values, a->dim, result->lower_bound, result->diff, result->x);

	pfree(values);

	PG_RETURN_POINTER(result);
}

/*
 * Multiply sq8 vectors element-wise
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_mul);
Datum
sq8vec_mul(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);
	Sq8Vector *result;
	float *values;
	float scale_a;
	float scale_b;

	CheckDims(a, b);

	values = palloc(sizeof(float) * a->dim);
	scale_a = a->diff / 255.0f;
	scale_b = b->diff / 255.0f;

	for (int i = 0; i < a->dim; i++)
	{
		float va = (float) a->x[i] * scale_a + a->lower_bound;
		float vb = (float) b->x[i] * scale_b + b->lower_bound;
		values[i] = va * vb;
	}

	result = InitSq8Vector(a->dim);
	Sq8ComputeQuantizationParams(values, a->dim, &result->lower_bound, &result->diff);
	Sq8EncodeVector(values, a->dim, result->lower_bound, result->diff, result->x);

	pfree(values);

	PG_RETURN_POINTER(result);
}

/*
 * Concatenate sq8 vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_concat);
Datum
sq8vec_concat(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);
	Sq8Vector *result;
	float *values;
	int dim = a->dim + b->dim;
	float scale_a;
	float scale_b;

	CheckDim(dim);

	values = palloc(sizeof(float) * dim);
	scale_a = a->diff / 255.0f;
	scale_b = b->diff / 255.0f;

	for (int i = 0; i < a->dim; i++)
		values[i] = (float) a->x[i] * scale_a + a->lower_bound;

	for (int i = 0; i < b->dim; i++)
		values[a->dim + i] = (float) b->x[i] * scale_b + b->lower_bound;

	result = InitSq8Vector(dim);
	Sq8ComputeQuantizationParams(values, dim, &result->lower_bound, &result->diff);
	Sq8EncodeVector(values, dim, result->lower_bound, result->diff, result->x);

	pfree(values);

	PG_RETURN_POINTER(result);
}

/*
 * Binary quantize sq8 vector to bit
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_binary_quantize);
Datum
sq8vec_binary_quantize(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	VarBit *result = InitBitVector(a->dim);
	unsigned char *rx = VARBITS(result);
	int i = 0;
	int count = (a->dim / 8) * 8;

	for (; i < count; i += 8)
	{
		unsigned char result_byte = 0;

		for (int j = 0; j < 8; j++)
		{
			float val = (a->x[i + j] / 255.0f) * a->diff + a->lower_bound;
			result_byte |= (val > 0) << (7 - j);
		}

		rx[i / 8] = result_byte;
	}

	for (; i < a->dim; i++)
	{
		float val = (a->x[i] / 255.0f) * a->diff + a->lower_bound;
		rx[i / 8] |= (val > 0) << (7 - (i % 8));
	}

	PG_RETURN_VARBIT_P(result);
}

/*
 * Get a subvector
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_subvector);
Datum
sq8vec_subvector(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	int32 start = PG_GETARG_INT32(1);
	int32 count = PG_GETARG_INT32(2);
	int32 end;
	Sq8Vector *result;
	float *values;
	int32 dim;

	if (count < 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("sq8vec must have at least 1 dimension")));

	if (start > a->dim - count)
		end = a->dim + 1;
	else
		end = start + count;

	if (start < 1)
		start = 1;
	else if (start > a->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("sq8vec must have at least 1 dimension")));

	dim = end - start;
	CheckDim(dim);

	values = palloc(sizeof(float) * dim);
	for (int i = 0; i < dim; i++)
	{
		values[i] = (a->x[start - 1 + i] / 255.0f) * a->diff + a->lower_bound;
	}

	result = InitSq8Vector(dim);
	Sq8ComputeQuantizationParams(values, dim, &result->lower_bound, &result->diff);
	Sq8EncodeVector(values, dim, result->lower_bound, result->diff, result->x);

	pfree(values);

	PG_RETURN_POINTER(result);
}

/*
 * Internal helper to compare sq8 vectors
 */
static int
sq8vec_cmp_internal(Sq8Vector *a, Sq8Vector *b)
{
	int dim = Min(a->dim, b->dim);

	for (int i = 0; i < dim; i++)
	{
		float va = (a->x[i] / 255.0f) * a->diff + a->lower_bound;
		float vb = (b->x[i] / 255.0f) * b->diff + b->lower_bound;

		if (va < vb)
			return -1;
		if (va > vb)
			return 1;
	}

	if (a->dim < b->dim)
		return -1;
	if (a->dim > b->dim)
		return 1;

	return 0;
}

/*
 * Less than
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_lt);
Datum
sq8vec_lt(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);

	PG_RETURN_BOOL(sq8vec_cmp_internal(a, b) < 0);
}

/*
 * Less than or equal
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_le);
Datum
sq8vec_le(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);

	PG_RETURN_BOOL(sq8vec_cmp_internal(a, b) <= 0);
}

/*
 * Equal
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_eq);
Datum
sq8vec_eq(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);

	PG_RETURN_BOOL(sq8vec_cmp_internal(a, b) == 0);
}

/*
 * Not equal
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_ne);
Datum
sq8vec_ne(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);

	PG_RETURN_BOOL(sq8vec_cmp_internal(a, b) != 0);
}

/*
 * Greater than or equal
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_ge);
Datum
sq8vec_ge(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);

	PG_RETURN_BOOL(sq8vec_cmp_internal(a, b) >= 0);
}

/*
 * Greater than
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_gt);
Datum
sq8vec_gt(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);

	PG_RETURN_BOOL(sq8vec_cmp_internal(a, b) > 0);
}

/*
 * Compare sq8 vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_cmp);
Datum
sq8vec_cmp(PG_FUNCTION_ARGS)
{
	Sq8Vector *a = PG_GETARG_SQ8VEC_P(0);
	Sq8Vector *b = PG_GETARG_SQ8VEC_P(1);

	PG_RETURN_INT32(sq8vec_cmp_internal(a, b));
}

/*
 * Check state array
 */
static float8 *
CheckStateArray(ArrayType *statearray, const char *caller)
{
	if (ARR_NDIM(statearray) != 1 ||
		ARR_DIMS(statearray)[0] < 1 ||
		ARR_HASNULL(statearray) ||
		ARR_ELEMTYPE(statearray) != FLOAT8OID)
		elog(ERROR, "%s: expected state array", caller);
	return (float8 *) ARR_DATA_PTR(statearray);
}

/*
 * Accumulate sq8 vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_accum);
Datum
sq8vec_accum(PG_FUNCTION_ARGS)
{
	ArrayType *statearray = PG_GETARG_ARRAYTYPE_P(0);
	Sq8Vector *newval = PG_GETARG_SQ8VEC_P(1);
	float8 *statevalues;
	int16 dim;
	bool newarr;
	float8 n;
	Datum *statedatums;
	ArrayType *result;

	statevalues = CheckStateArray(statearray, "sq8vec_accum");
	dim = STATE_DIMS(statearray);
	newarr = dim == 0;

	if (newarr)
		dim = newval->dim;
	else
		CheckExpectedDim(dim, newval->dim);

	n = statevalues[0] + 1.0;

	statedatums = CreateStateDatums(dim);
	statedatums[0] = Float8GetDatum(n);

	if (newarr)
	{
		for (int i = 0; i < dim; i++)
		{
			float val = (newval->x[i] / 255.0f) * newval->diff + newval->lower_bound;
			statedatums[i + 1] = Float8GetDatum((double) val);
		}
	}
	else
	{
		for (int i = 0; i < dim; i++)
		{
			float val = (newval->x[i] / 255.0f) * newval->diff + newval->lower_bound;
			double v = statevalues[i + 1] + (double) val;

			if (isinf(v))
				float_overflow_error();

			statedatums[i + 1] = Float8GetDatum(v);
		}
	}

	result = construct_array(statedatums, dim + 1,
							 FLOAT8OID,
							 sizeof(float8), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE);

	pfree(statedatums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * Average sq8 vectors
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(sq8vec_avg);
Datum
sq8vec_avg(PG_FUNCTION_ARGS)
{
	ArrayType *statearray = PG_GETARG_ARRAYTYPE_P(0);
	float8 *statevalues;
	float8 n;
	uint16 dim;
	Sq8Vector *result;
	float *values;

	statevalues = CheckStateArray(statearray, "sq8vec_avg");
	n = statevalues[0];

	if (n == 0.0)
		PG_RETURN_NULL();

	dim = STATE_DIMS(statearray);
	CheckDim(dim);

	values = palloc(sizeof(float) * dim);
	for (int i = 0; i < dim; i++)
		values[i] = (float)(statevalues[i + 1] / n);

	result = InitSq8Vector(dim);
	Sq8ComputeQuantizationParams(values, dim, &result->lower_bound, &result->diff);
	Sq8EncodeVector(values, dim, result->lower_bound, result->diff, result->x);

	pfree(values);

	PG_RETURN_POINTER(result);
}

/*
 * HNSW support function for sq8vec
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnsw_sq8vec_support);
Datum
hnsw_sq8vec_support(PG_FUNCTION_ARGS)
{
	static const HnswTypeInfo typeInfo = {
		.maxDimensions = SQ8VEC_MAX_DIM,
		.normalize = sq8vec_l2_normalize,
		.checkValue = NULL
	};

	PG_RETURN_POINTER(&typeInfo);
}

/*
 * IVFFlat support function for sq8vec
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ivfflat_sq8vec_support);
Datum
ivfflat_sq8vec_support(PG_FUNCTION_ARGS)
{
	static const IvfflatTypeInfo typeInfo = {
		.maxDimensions = IVFFLAT_MAX_DIM,
		.normalize = sq8vec_l2_normalize,
		.itemSize = Sq8vecItemSize,
		.updateCenter = Sq8vecUpdateCenter,
		.sumCenter = Sq8vecSumCenter
	};

	PG_RETURN_POINTER(&typeInfo);
}
