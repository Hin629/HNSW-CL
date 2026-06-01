#ifndef SQ8VEC_H
#define SQ8VEC_H

#include "postgres.h"
#include <stdint.h>

/*
 * SQ8 Vector Type
 * 
 * Uses Scalar Quantization with 8-bit codes.
 * Each float is mapped to uint8 using: code = (value - lower_bound) / diff * 255
 * Where diff = upper_bound - lower_bound
 * 
 * The quantization parameters (lower_bound, diff) are stored in the vector header.
 */

#define SQ8VEC_MAX_DIM 16000

/* 
 * SQ8 Vector structure:
 * - vl_len_: varlena header
 * - dim: number of dimensions  
 * - lower_bound: minimum value used for quantization
 * - diff: range (upper_bound - lower_bound) used for quantization
 * - x[]: quantized codes (uint8)
 */
typedef struct Sq8Vector
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int16		dim;			/* number of dimensions */
	int16		unused;			/* reserved for alignment/future use */
	float		lower_bound;	/* quantization lower bound */
	float		diff;			/* quantization range (upper - lower) */
	uint8		x[FLEXIBLE_ARRAY_MEMBER];	/* quantized codes */
}			Sq8Vector;

/* Size calculation includes header + quantization params + codes */
#define SQ8VEC_SIZE(_dim)		(offsetof(Sq8Vector, x) + sizeof(uint8)*(_dim))
#define DatumGetSq8Vector(x)	((Sq8Vector *) PG_DETOAST_DATUM(x))
#define PG_GETARG_SQ8VEC_P(x)	DatumGetSq8Vector(PG_GETARG_DATUM(x))
#define PG_RETURN_SQ8VEC_P(x)	PG_RETURN_POINTER(x)

/*
 * SQ8 Index Tuple - Compact representation for HNSW index internal storage.
 * 
 * This structure stores ONLY the quantized codes without per-vector quantization
 * parameters. The global quantization parameters (lower_bound, diff) are stored
 * in the HNSW meta page and shared by all vectors in the index.
 * 
 * Benefits:
 * - Smaller index storage: saves 8 bytes per vector (no per-vector lb/diff)
 * - Faster distance computation: uses integer-only SameQuant functions
 * - Better cache efficiency: more compact data
 * 
 * Optimization: Precomputed sum and sum2 fields for IP/Cosine distance:
 * - sum = sum(codes[i])  
 * - sum2 = sum(codes[i]^2)
 * These are computed once during index build and used to accelerate distance 
 * calculations, avoiding O(dim) summation for each distance computation.
 * 
 * The structure looks like a varlena so it can be stored as a Datum in the
 * existing HNSW element tuple infrastructure.
 */
typedef struct Sq8IndexTuple
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int16		dim;			/* number of dimensions */
	int16		unused;			/* reserved for alignment */
	int32		sum;			/* precomputed sum(codes) for IP/Cosine */
	int32		sum2;			/* precomputed sum(codes^2) for Cosine */
	uint8		codes[FLEXIBLE_ARRAY_MEMBER];	/* quantized codes only */
}			Sq8IndexTuple;

/* Size calculation for index tuple: header + dim info + sum fields + codes */
#define SQ8_INDEX_TUPLE_SIZE(_dim)		(offsetof(Sq8IndexTuple, codes) + sizeof(uint8)*(_dim))
#define DatumGetSq8IndexTuple(x)		((Sq8IndexTuple *) DatumGetPointer(x))

/* Check if a datum is an Sq8IndexTuple (by size comparison) */
#define IsSq8IndexTuple(ptr, dim) \
	(VARSIZE_ANY(ptr) == SQ8_INDEX_TUPLE_SIZE(dim))

/* Check if a datum is an Sq8Vector (has lower_bound and diff fields) */
#define IsSq8Vector(ptr) \
	(VARSIZE_ANY(ptr) >= offsetof(Sq8Vector, x) && \
	 ((Sq8Vector *)(ptr))->dim > 0 && \
	 VARSIZE_ANY(ptr) == SQ8VEC_SIZE(((Sq8Vector *)(ptr))->dim))

/* Function declarations */
Sq8IndexTuple *InitSq8IndexTuple(int dim);

#endif /* SQ8VEC_H */
