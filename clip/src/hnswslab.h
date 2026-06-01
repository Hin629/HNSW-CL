/*
 * Shared memory slab for HNSW index
 */
#ifndef HNSWSLAB_H
#define HNSWSLAB_H

#include "postgres.h"
#include "hnsw.h"
#include "storage/dsm.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "utils/hsearch.h"

/*
 * Slab state machine:
 * - INVALID: slab does not exist or was corrupted
 * - BUILDING: index build in progress, slab being populated
 * - READY: build complete, slab available for queries
 */
typedef enum HnswSlabState
{
	HNSW_SLAB_INVALID = 0,
	HNSW_SLAB_BUILDING = 1,
	HNSW_SLAB_READY = 2
} HnswSlabState;

/*
 * Directory entry for one index's slab in shared memory hash table.
 * Key is (dbid, relid) to uniquely identify the index.
 */
typedef struct HnswSlabDirKey
{
	Oid			dbid;			/* MyDatabaseId */
	Oid			relid;			/* index relation OID */
} HnswSlabDirKey;

typedef struct HnswSlabDirEntry
{
	HnswSlabDirKey key;
	
	/* State and DSM handle */
	HnswSlabState state;
	dsm_handle	handle;			/* DSM segment handle, 0 if none */
	
	/* Validation fields */
	uint32		magic;			/* HNSW_SLAB_MAGIC for validation */
	uint32		version;		/* slab format version */
	Size		totalBytes;		/* total size of DSM segment */
	uint64		buildXid;		/* XID of the build transaction */
	
	/* Index parameters (for validation) */
	int			m;
	int			dimensions;
	uint32		numElements;
	int			maxLevel;		/* max level of entry point */
	
	/* Entry point info */
	uint32		entryPointId;	/* internal ID of entry point */
	int			entryPointLevel;
	
	/* Waiting mechanism for BUILDING -> READY transition */
	int			waitersCount;
} HnswSlabDirEntry;

#define HNSW_SLAB_MAGIC 0x484E5357
#define HNSW_SLAB_VERSION 1	

#define HNSW_SLAB_MAX_NODE_LOCKS 65536

/* DELETE_MARK */
#define HNSW_SLAB_DELETE_MARK 0x01

/* Maximum level for HNSW graph */
#define HNSW_SLAB_MAX_LEVEL 32

/* Random value constant */
#define HNSW_SLAB_MAX_RANDOM_VALUE ((1U << 31) - 1)

/*
 * Label lookup entry.
 * Maps heaptid (encoded as uint64) to internalId.
 */
typedef struct HnswSlabLabelEntry
{
	uint64		label;			/* Encoded heaptid: (blkno << 16) | off */
	uint32		internalId;		/* Internal slab node ID */
} HnswSlabLabelEntry;

/*
 * Label lookup sentinels for open-addressing table.
 *
 * Our label encoding is always < 2^48 (BlockNumber is 32-bit, OffsetNumber is 16-bit),
 * so UINT64_MAX is safe as a tombstone marker.
 */
#define HNSW_SLAB_LABEL_EMPTY UINT64CONST(0)
#define HNSW_SLAB_LABEL_TOMBSTONE UINT64CONST(0xFFFFFFFFFFFFFFFF)

/* Label lookup key for hash table */
typedef struct HnswSlabLabelKey
{
	Oid			dbid;
	Oid			relid;
	uint64		label;			/* Encoded heaptid */
} HnswSlabLabelKey;

/* Label lookup directory entry (per-index) */
typedef struct HnswSlabLabelDirEntry
{
	HnswSlabLabelKey key;
	uint32		internalId;
} HnswSlabLabelDirEntry;

/* Encode heaptid to uint64 label (stable, comparable, hashable) */
static inline uint64
HnswSlabEncodeHeaptid(ItemPointer heaptid)
{
	if (!ItemPointerIsValid(heaptid))
		return 0;
	return ((uint64)ItemPointerGetBlockNumber(heaptid) << 16) |
		   (uint64)ItemPointerGetOffsetNumber(heaptid);
}

/* Decode uint64 label back to heaptid */
static inline void
HnswSlabDecodeHeaptid(uint64 label, ItemPointer heaptid)
{
	if (label == 0)
	{
		ItemPointerSetInvalid(heaptid);
		return;
	}
	ItemPointerSet(heaptid, (BlockNumber)(label >> 16), (OffsetNumber)(label & 0xFFFF));
}

/*
 * Slab header stored at the beginning of the DSM segment.
 */
typedef struct HnswSlabHeader
{
	uint32		magic;
	uint32		version;
	
	/* Index parameters */
	int			m;
	int			maxM0;			/* 2*m for layer 0 */
	int			dimensions;
	int			maxLevel;
	
	/* Legacy element count for build-time use (use curElementCount for runtime) */
	uint32		numElements;
	
	/* Dynamic element count - use atomic for concurrent access */
	pg_atomic_uint32 curElementCount;	/* Current number of elements */
	pg_atomic_uint32 numDeleted;		/* Number of deleted elements */
	uint32		maxElements;			/* Maximum capacity */
	
	/* Legacy entry point fields for build-time use */
	uint32		entryPointId;
	int			entryPointLevel;
	
	/* Entry point - atomic access for runtime */
	pg_atomic_uint32 entryPointIdAtomic;
	pg_atomic_uint32 entryPointLevelAtomic;	/* -1 stored as 0xFFFFFFFF */
	
	/* Node locks offset */
	Size		nodeLockOffset;
	uint32		nodeLockCount;
	
	/* Random level generator (mult = 1/log(m)) */
	double		mult;
	
	/* Global spinlock for entry point updates*/
	slock_t		globalLock;
	
	
	/* Deleted elements pool offset and count */
	Size		deletedPoolOffset;		/* Offset to deleted ID pool (ring buffer) */
	pg_atomic_uint32 deletedPoolHead;	/* Next slot to pop from (read position) */
	pg_atomic_uint32 deletedPoolTail;	/* Next slot to push to (write position) */
	uint32		deletedPoolCapacity;	/* Max entries in pool */
	slock_t		deletedPoolLock;		/* Lock for deleted_elements operations */
	
	/* Label lookup hash table offset */
	Size		labelLookupOffset;		/* Offset to label->internalId hash table */
	uint32		labelLookupCapacity;	/* Number of hash buckets */
	slock_t		labelLookupLock;		/* Lock for label_lookup_ operations */
	

	Size		dataLevel0Offset;		/* Offset to level 0 AOS data */
	Size		dataLevel0PerElement;	/* Total bytes per element in level 0 */
	Size		offsetDataInElement;	/* Offset to vector within element */
	Size		offsetMetaInElement;	/* Offset to metadata within element */
	Size		vectorStride;			/* bytes per vector slot */
	Size		adj0Stride;				/* bytes per layer 0 adjacency */
	Size		nodeMetaStride;			/* bytes per node metadata */
	
	/* Upper layers adjacency (separate pool, as level varies per node) */
	Size		adjUpperOffset;		/* upper layers adjacency offset */
	Size		adjUpperStride;		/* bytes per upper layer chunk (single chunk size) */
	
	/* Compact upper adjacency layout */
	uint64		totalUpperChunks;		/* Total upper layer chunks allocated */
	pg_atomic_uint64 usedUpperChunks;	/* Used upper layer chunks (for dynamic insert) */
	
	/* Sizes */
	Size		totalSize;
	
	/* SQ8 quantization parameters */
	bool		isSq8Index;
	float		sq8LowerBound;
	float		sq8Diff;
} HnswSlabHeader;

/*
 * Per-node metadata stored in slab.
 * Equivalent to HnswElementData fields needed for query.
 */
typedef struct HnswSlabNodeMeta
{
	ItemPointerData heaptids[HNSW_HEAPTIDS];
	uint8		heaptidsLength;
	uint8		level;
	uint8		deleted;
	uint8		version;
	uint32		adjUpperOffset;		/* Offset into compact upper adjacency pool */
} HnswSlabNodeMeta;

/*
 * Per-backend private visited array with tag-based reset.
 */
typedef uint16 SlabVisitedTag;

typedef struct HnswSlabVisited
{
	SlabVisitedTag currentTag;
	uint32		capacity;
	SlabVisitedTag *tags;		/* array indexed by internal_id */
	MemoryContext ctx;
} HnswSlabVisited;

/*
 * Attached slab handle for a backend.
 * Holds DSM mapping and cached pointers.
 */
typedef struct HnswSlabHandle
{
	/* Cache identity and lifetime (per-backend) */
	Oid			cacheDbid;
	Oid			cacheRelid;
	dsm_handle	cacheDsmHandle;
	int			refcount;
	uint64		lastUsed;
	bool		cached;

	dsm_segment *segment;
	HnswSlabHeader *header;
	
	/* Cached pointers into DSM - AOS layout */
	LWLockPadded *nodeLocks;		/* Array of per-node locks */
	char	   *dataLevel0;			/* AOS data: [adj0 + vector + meta] per element */
	char	   *adjUpperPool;		/* Upper layer adjacency (separate) */
	
	/* Deleted elements pool (ring buffer) */
	uint32	   *deletedPool;		/* Array of deleted internal IDs */
	
	/* Label lookup hash table pointer */
	HnswSlabLabelEntry *labelLookup;
	uint32		labelLookupCapacity;
	
	/* Per-backend visited array (private, not in DSM) */
	HnswSlabVisited *visited;
	
	/* Cached parameters */
	int			m;
	int			maxM0;
	int			dimensions;
	uint32		maxElements;		/* Maximum capacity */
	uint32		nodeLockCount;
	
	/* AOS layout parameters (cached from header) */
	Size		dataLevel0PerElement;	/* Total bytes per element */
	Size		offsetDataInElement;	/* Offset to vector within element */
	Size		offsetMetaInElement;	/* Offset to metadata within element */
	Size		vectorStride;
	Size		adj0Stride;
	Size		adjUpperStride;
	
	/* Entry point - refreshed from atomics */
	uint32		entryPointId;
	int			entryPointLevel;
	
	/* Random level generator state (per-backend) */
	double		mult;
	unsigned int levelGenState;
	
	
	/* Deleted pool parameters */
	uint32		deletedPoolCapacity;
	
	/* SQ8 parameters */
	bool		isSq8Index;
	float		sq8LowerBound;
	float		sq8Diff;
	float		sq8Scale;
	float		sq8Scale2;
} HnswSlabHandle;

/*
 * Build context for populating slab during index build.
 */
typedef struct HnswSlabBuildContext
{
	dsm_segment *segment;
	HnswSlabHeader *header;
	
	/* AOS layout pointers */
	char	   *dataLevel0;			/* Level 0 AOS data */
	char	   *adjUpperPool;		/* Upper layer adjacency */
	
	uint32		nextInternalId;
	int			m;
	int			dimensions;
	
	/* Compact upper adjacency tracking */
	uint64		currentUpperOffset;		/* Next available offset in upper adjacency pool */

	/* DSM handle (segment may be detached before commit callback) */
	dsm_handle	dsmHandle;
	Size		totalBytes;

	/* Publish fields copied from header when finished */
	uint32		publishNumElements;
	int			publishMaxLevel;
	uint32		publishEntryPointId;
	int			publishEntryPointLevel;
	Size		publishTotalSize;

	/* Registered for transaction callback */
	bool		registered;
	bool		finished;
	Oid			dbid;
	Oid			relid;
} HnswSlabBuildContext;

/* --- Shared memory initialization --- */
extern Size HnswSlabShmemSize(void);
extern void HnswSlabShmemInit(void);

/* --- Directory operations (protected by LWLock) --- */
extern HnswSlabDirEntry *HnswSlabDirLookup(Oid dbid, Oid relid, bool *found);
extern HnswSlabDirEntry *HnswSlabDirCreate(Oid dbid, Oid relid, bool *found);
extern void HnswSlabDirRemove(Oid dbid, Oid relid);
extern void HnswSlabDirSetReady(Oid dbid, Oid relid, dsm_handle handle,
								uint32 numElements, int m, int dimensions,
								int maxLevel, uint32 entryPointId, int entryPointLevel,
								Size totalBytes);
extern void HnswSlabDirSetInvalid(Oid dbid, Oid relid);

/* --- Build operations --- */
extern HnswSlabBuildContext *HnswSlabBuildBegin(Oid dbid, Oid relid,
												int m, int dimensions,
												Size vectorStride,
												uint32 estimatedElements,
												int maxLevel,
												uint64 totalUpperChunks,
												bool isSq8Index,
												float sq8LowerBound,
												float sq8Diff);
extern uint32 HnswSlabBuildAddVector(HnswSlabBuildContext *ctx,
									 Pointer vectorData, Size vectorSize);
extern void HnswSlabBuildSetNodeMeta(HnswSlabBuildContext *ctx,
									 uint32 internalId,
									 ItemPointerData *heaptids,
									 uint8 heaptidsLength,
									 uint8 level, uint8 deleted, uint8 version);
extern void HnswSlabBuildSetAdjacency(HnswSlabBuildContext *ctx,
									  uint32 internalId, int layer,
									  uint32 *neighborIds, int neighborCount);
extern void HnswSlabBuildSetEntryPoint(HnswSlabBuildContext *ctx,
									   uint32 entryPointId, int entryPointLevel);
extern void HnswSlabBuildFinish(HnswSlabBuildContext *ctx);
extern void HnswSlabBuildAbort(HnswSlabBuildContext *ctx);

/* --- Query operations --- */
extern HnswSlabHandle *HnswSlabAttach(Oid dbid, Oid relid);
extern void HnswSlabDetach(HnswSlabHandle *handle);

/* --- Slab data access --- */

/*
 * Get pointer to the start of an element's data in level 0.
 * Element layout: [adj0] + [vector] + [nodeMeta]
 */
static inline char *
HnswSlabGetElementBase(HnswSlabHandle *handle, uint32 internalId)
{
	return handle->dataLevel0 + (Size)internalId * handle->dataLevel0PerElement;
}

/* Get vector data by internal ID - equivalent to getDataByInternalId() */
static inline char *
HnswSlabGetVector(HnswSlabHandle *handle, uint32 internalId)
{
	return HnswSlabGetElementBase(handle, internalId) + handle->offsetDataInElement;
}

/* Get layer 0 adjacency list - returns pointer to [count][id1][id2]... */
static inline uint32 *
HnswSlabGetAdj0(HnswSlabHandle *handle, uint32 internalId)
{
	/* adj0 is at offset 0 within each element */
	return (uint32 *)HnswSlabGetElementBase(handle, internalId);
}

/* Get node metadata - now inline within each element */
static inline HnswSlabNodeMeta *
HnswSlabGetNodeMeta(HnswSlabHandle *handle, uint32 internalId)
{
	return (HnswSlabNodeMeta *)(HnswSlabGetElementBase(handle, internalId) + handle->offsetMetaInElement);
}

/* Get upper layer adjacency list for given layer */
static inline uint32 *
HnswSlabGetAdjUpper(HnswSlabHandle *handle, uint32 internalId, int layer)
{
	Size offset;
	HnswSlabNodeMeta *meta;
	int layerIndex;
	Size upperChunkSize;
	
	if (layer == 0)
		return HnswSlabGetAdj0(handle, internalId);
	
	/* Upper layers are stored consecutively per node in separate pool */
	meta = HnswSlabGetNodeMeta(handle, internalId);
	if (layer > meta->level)
		return NULL;
	
	/* Use compact layout via adjUpperOffset */
	/* layerIndex 0 = layer 1, layerIndex 1 = layer 2, etc */
	layerIndex = layer - 1;
	upperChunkSize = handle->adjUpperStride;
	offset = (Size)meta->adjUpperOffset * upperChunkSize + 
			 (Size)layerIndex * upperChunkSize;
	
	return (uint32 *)(handle->adjUpperPool + offset);
}

/* Get adjacency list count (first element) */
static inline uint32
HnswSlabGetAdjCount(uint32 *adj)
{
	return adj[0];
}

/* Set adjacency list count */
static inline void
HnswSlabSetAdjCount(uint32 *adj, uint32 count)
{
	adj[0] = count;
}

/* Get adjacency list neighbor ID at index */
static inline uint32
HnswSlabGetAdjNeighbor(uint32 *adj, int idx)
{
	return adj[idx + 1];
}

/* Set adjacency list neighbor ID at index */
static inline void
HnswSlabSetAdjNeighbor(uint32 *adj, int idx, uint32 neighborId)
{
	adj[idx + 1] = neighborId;
}

/* Get adjacency list at any level */
static inline uint32 *
HnswSlabGetAdjAtLevel(HnswSlabHandle *handle, uint32 internalId, int layer)
{
	if (layer == 0)
		return HnswSlabGetAdj0(handle, internalId);
	else
		return HnswSlabGetAdjUpper(handle, internalId, layer);
}

/* Set adjacency list count at any level (convenience wrapper) */
static inline void
HnswSlabSetAdjCountAtLevel(HnswSlabHandle *handle, uint32 internalId, int layer, uint32 count)
{
	uint32 *adj = HnswSlabGetAdjAtLevel(handle, internalId, layer);
	if (adj != NULL)
		adj[0] = count;
}

/* Set adjacency list neighbor at any level (convenience wrapper) */
static inline void
HnswSlabSetAdjNeighborAtLevel(HnswSlabHandle *handle, uint32 internalId, int layer, int idx, uint32 neighborId)
{
	uint32 *adj = HnswSlabGetAdjAtLevel(handle, internalId, layer);
	if (adj != NULL)
		adj[idx + 1] = neighborId;
}

/* Get node lock for given internal ID (uses modulo for scalability) */
static inline LWLock *
HnswSlabGetNodeLock(HnswSlabHandle *handle, uint32 internalId)
{
	uint32 lockIdx;

	if (handle->nodeLocks == NULL || handle->nodeLockCount == 0)
		return NULL;

	lockIdx = internalId % handle->nodeLockCount;
	return &handle->nodeLocks[lockIdx].lock;
}

/* Refresh entry point from header atomics */
static inline void
HnswSlabRefreshEntryPoint(HnswSlabHandle *handle)
{
	uint32 epLevel;
	handle->entryPointId = pg_atomic_read_u32(&handle->header->entryPointIdAtomic);
	epLevel = pg_atomic_read_u32(&handle->header->entryPointLevelAtomic);
	handle->entryPointLevel = (epLevel == 0xFFFFFFFF) ? -1 : (int)epLevel;
}

/* Get current element count from header */
static inline uint32
HnswSlabGetElementCount(HnswSlabHandle *handle)
{
	return pg_atomic_read_u32(&handle->header->curElementCount);
}

/* Fast validity check for an internal ID */
static inline bool
HnswSlabIsValidInternalId(HnswSlabHandle *handle, uint32 internalId)
{
	/* maxElements is the allocated capacity in the DSM layout */
	if (internalId >= handle->maxElements)
		return false;
	/* curElementCount is the published element count */
	if (internalId >= HnswSlabGetElementCount(handle))
		return false;
	return true;
}

/* --- Visited array operations (per-backend private) --- */
extern HnswSlabVisited *HnswSlabVisitedCreate(MemoryContext ctx, uint32 capacity);
extern void HnswSlabVisitedDestroy(HnswSlabVisited *visited);

/* Reset visited array - O(1) via tag increment */
static inline void
HnswSlabVisitedReset(HnswSlabVisited *visited)
{
	visited->currentTag++;
	if (visited->currentTag == 0)
	{
		/* Tag wrapped around, clear array */
		MemSet(visited->tags, 0, sizeof(SlabVisitedTag) * visited->capacity);
		visited->currentTag = 1;
	}
}

/* Check if visited and mark as visited - returns true if already visited */
static inline bool
HnswSlabVisitedCheckAndMark(HnswSlabVisited *visited, uint32 internalId)
{
	if (internalId >= visited->capacity)
		return true;  /* Out of range, treat as already visited (skip) */
	
	if (visited->tags[internalId] == visited->currentTag)
		return true;  /* Already visited */
	
	visited->tags[internalId] = visited->currentTag;
	return false;  /* Now marked as visited */
}

/* --- Search operations --- */
extern List *HnswSlabSearchBaseLayer(HnswSlabHandle *handle,
									 HnswSlabVisited *visited,
									 Datum queryValue,
									 HnswSupport *support,
									 uint32 epId,
									 int ef,
									 pairingheap **discarded,
									 bool resetVisited);

extern uint32 HnswSlabSearchUpperLayers(HnswSlabHandle *handle,
										Datum queryValue,
										HnswSupport *support,
										uint32 epId,
										int epLevel,
										int targetLayer);

/* --- Dynamic insert operations --- */
extern uint32 HnswSlabAddPoint(HnswSlabHandle *handle,
							   Pointer vectorData, Size vectorSize,
							   ItemPointerData *heaptids, uint8 heaptidsLength,
							   Datum queryValue, HnswSupport *support,
							   int efConstruction);

/* --- Dynamic delete operations --- */
extern void HnswSlabMarkDeleted(HnswSlabHandle *handle, uint32 internalId);
extern bool HnswSlabIsDeleted(HnswSlabHandle *handle, uint32 internalId);
extern void HnswSlabUnmarkDeleted(HnswSlabHandle *handle, uint32 internalId);

/* --- Label lookup operations --- */
extern bool HnswSlabLabelLookup(HnswSlabHandle *handle, uint64 label, uint32 *internalId);
extern void HnswSlabLabelInsert(HnswSlabHandle *handle, uint64 label, uint32 internalId);
extern void HnswSlabLabelRemove(HnswSlabHandle *handle, uint64 label);
extern uint32 HnswSlabLabelLookupByHeaptid(HnswSlabHandle *handle, ItemPointer heaptid, bool *found);

/* --- Deleted elements pool operations --- */
extern bool HnswSlabPopDeletedId(HnswSlabHandle *handle, uint32 *internalId);
extern void HnswSlabPushDeletedId(HnswSlabHandle *handle, uint32 internalId);

/* --- Update point operation --- */
extern void HnswSlabUpdatePoint(HnswSlabHandle *handle,
								uint32 internalId,
								Pointer vectorData, Size vectorSize,
								Datum queryValue, HnswSupport *support,
								int efConstruction);

/* --- Vacuum sync operations --- */
extern void HnswSlabSyncDeleteFromVacuum(Oid dbid, Oid relid,
										ItemPointer heaptid);
extern void HnswSlabSyncRemoveHeaptidFromVacuum(Oid dbid, Oid relid,
												ItemPointer heaptid);
extern void HnswSlabSyncUpdateHeaptidsFromVacuum(Oid dbid, Oid relid,
												 ItemPointer oldPrimaryHeaptid,
												 ItemPointerData *heaptids,
												 uint8 heaptidsLength);

/* --- Utility functions --- */
extern int HnswSlabGetRandomLevel(HnswSlabHandle *handle);
extern void HnswSlabVisitedResize(HnswSlabVisited *visited, uint32 newCapacity);
extern List *HnswSlabScanAll(HnswSlabHandle *handle);

/* --- Auto-rebuild from disk API --- */
extern HnswSlabHandle *HnswSlabAttachOrBuildFromDisk(Relation index);

#endif /* HNSWSLAB_H */
