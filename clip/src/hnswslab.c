/*
 * Shared memory slab implementation for HNSW index.
 *
 * AOS (Array of Structures) Layout :
 * - Level 0 data stored as [adj0 + vector + metadata] per element (improved cache locality)
 * - Upper layer adjacency in separate compact pool (as level varies per node)
 * - Integer adjacency lists with [count][id1][id2]... format
 * - Per-backend private visited_array with tag-based O(1) reset
 */
#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/xact.h"
#include "hnsw.h"
#include "hnswslab.h"
#include "sq8vec.h"
#include "sq8utils.h"
#include "lib/pairingheap.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "storage/dsm.h"
#include "storage/ipc.h"
#include "storage/lmgr.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#ifdef USE_SSE
#include <xmmintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define HNSW_SLAB_PREFETCH_R(addr) __builtin_prefetch((addr), 0, 3)
#else
#define HNSW_SLAB_PREFETCH_R(addr) ((void) 0)
#endif

#ifdef USE_SSE
#undef HNSW_SLAB_PREFETCH_R
#define HNSW_SLAB_PREFETCH_R(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#endif

#ifndef HNSW_SLAB_PREFETCH_DISTANCE
#define HNSW_SLAB_PREFETCH_DISTANCE 4
#endif

/* Global shared hash table for slab directory */
static HTAB *HnswSlabDirHash = NULL;
static LWLock *HnswSlabDirLock = NULL;

static HTAB *HnswSlabLocalCache = NULL;
static uint64 HnswSlabLocalClock = 0;
static bool HnswSlabLocalCacheExitHookRegistered = false;

#define HNSW_SLAB_LOCAL_CACHE_MAX 8

typedef struct HnswSlabLocalEntry
{
	HnswSlabDirKey key;
	dsm_handle dhandle;
	HnswSlabHandle *handle;
} HnswSlabLocalEntry;

static void HnswSlabLocalCacheInitIfNeeded(void);
static void HnswSlabLocalCacheMaybeEvict(void);
static void HnswSlabHandleFreeInternal(HnswSlabHandle *handle);
static void HnswSlabLocalCacheCleanup(int code, Datum arg);

/* Maximum number of HNSW indexes that can have slabs */
#define HNSW_SLAB_DIR_SIZE 64

/* Per-backend tracking for build context cleanup */
static HnswSlabBuildContext *CurrentBuildContext = NULL;

/* Forward declarations */
static void HnswSlabXactCallback(XactEvent event, void *arg);
static Size HnswSlabEstimateSize(int m, Size vectorStride, uint32 numElements, int maxLevel, uint64 totalUpperChunks);
static dsm_handle HnswSlabDirGetReadyHandle(Oid dbid, Oid relid);
static dsm_handle HnswSlabDirRemoveAndGetHandle(Oid dbid, Oid relid);

static void
HnswSlabLocalCacheInitIfNeeded(void)
{
	HASHCTL ctl;

	if (HnswSlabLocalCache != NULL)
		return;

	MemSet(&ctl, 0, sizeof(ctl));
	ctl.keysize = sizeof(HnswSlabDirKey);
	ctl.entrysize = sizeof(HnswSlabLocalEntry);
	ctl.hcxt = TopMemoryContext;

	HnswSlabLocalCache = hash_create("HNSW Slab Local Cache",
								HNSW_SLAB_DIR_SIZE,
								&ctl,
								HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	if (!HnswSlabLocalCacheExitHookRegistered)
	{
		before_shmem_exit(HnswSlabLocalCacheCleanup, (Datum) 0);
		HnswSlabLocalCacheExitHookRegistered = true;
	}
}

static void
HnswSlabLocalCacheCleanup(int code, Datum arg)
{
	HASH_SEQ_STATUS seq;
	HnswSlabLocalEntry *entry;

	if (HnswSlabLocalCache == NULL)
		return;

	hash_seq_init(&seq, HnswSlabLocalCache);
	while ((entry = (HnswSlabLocalEntry *) hash_seq_search(&seq)) != NULL)
	{
		if (entry->handle != NULL)
		{
			HnswSlabHandle *h = entry->handle;
			entry->handle = NULL;
			HnswSlabHandleFreeInternal(h);
		}
	}
}

static void
HnswSlabHandleFreeInternal(HnswSlabHandle *handle)
{
	if (handle == NULL)
		return;

	/* This frees per-backend visited array */
	if (handle->visited != NULL)
	{
		HnswSlabVisitedDestroy(handle->visited);
		handle->visited = NULL;
	}

	/* Unmap DSM segment */
	if (handle->segment != NULL)
	{
		dsm_detach(handle->segment);
		handle->segment = NULL;
	}

	pfree(handle);
}

static void
HnswSlabLocalCacheMaybeEvict(void)
{
	HASH_SEQ_STATUS seq;
	HnswSlabLocalEntry *entry;
	HnswSlabLocalEntry *victim = NULL;
	uint64 victimLastUsed = UINT64_MAX;

	if (HnswSlabLocalCache == NULL)
		return;

	if (hash_get_num_entries(HnswSlabLocalCache) <= HNSW_SLAB_LOCAL_CACHE_MAX)
		return;

	/* Evict least-recently-used entry with refcount==0 */
	hash_seq_init(&seq, HnswSlabLocalCache);
	while ((entry = (HnswSlabLocalEntry *) hash_seq_search(&seq)) != NULL)
	{
		if (entry->handle == NULL)
			continue;
		if (entry->handle->refcount != 0)
			continue;
		if (entry->handle->lastUsed < victimLastUsed)
		{
			victim = entry;
			victimLastUsed = entry->handle->lastUsed;
		}
	}

	if (victim != NULL)
	{
		HnswSlabHandle *h = victim->handle;
		victim->handle = NULL;
		hash_search(HnswSlabLocalCache, &victim->key, HASH_REMOVE, NULL);
		HnswSlabHandleFreeInternal(h);
	}
}

/*
 * Estimate shared memory size for slab directory.
 */
Size
HnswSlabShmemSize(void)
{
	Size size = 0;
	
	/* Hash table for directory */
	size = add_size(size, hash_estimate_size(HNSW_SLAB_DIR_SIZE, sizeof(HnswSlabDirEntry)));
	
	/* LWLock */
	size = add_size(size, sizeof(LWLockPadded));
	
	return size;
}

/*
 * Initialize shared memory for slab directory.
 */
void
HnswSlabShmemInit(void)
{
	HASHCTL		hashCtl;
	bool		found;
	LWLockPadded *lockPad;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);

	/* Initialize hash table */
	MemSet(&hashCtl, 0, sizeof(hashCtl));
	hashCtl.keysize = sizeof(HnswSlabDirKey);
	hashCtl.entrysize = sizeof(HnswSlabDirEntry);
	
	HnswSlabDirHash = ShmemInitHash("HNSW Slab Directory",
									HNSW_SLAB_DIR_SIZE,
									HNSW_SLAB_DIR_SIZE,
									&hashCtl,
									HASH_ELEM | HASH_BLOBS);
	
	/* Initialize LWLock */
	lockPad = (LWLockPadded *) ShmemInitStruct("HNSW Slab Lock",
											   sizeof(LWLockPadded),
											   &found);
	if (!found)
	{
		LWLockInitialize(&lockPad->lock, hnsw_lock_tranche_id);
	}
	HnswSlabDirLock = &lockPad->lock;

	LWLockRelease(AddinShmemInitLock);
}

/*
 * Look up a slab directory entry.
 */
HnswSlabDirEntry *
HnswSlabDirLookup(Oid dbid, Oid relid, bool *found)
{
	HnswSlabDirKey key;
	HnswSlabDirEntry *entry;
	
	/* Check if slab directory is initialized */
	if (HnswSlabDirHash == NULL || HnswSlabDirLock == NULL)
	{
		*found = false;
		return NULL;
	}
	
	key.dbid = dbid;
	key.relid = relid;

	LWLockAcquire(HnswSlabDirLock, LW_SHARED);
	entry = (HnswSlabDirEntry *) hash_search(HnswSlabDirHash, &key, HASH_FIND, found);
	LWLockRelease(HnswSlabDirLock);

	return entry;
}

/*
 * Get DSM handle for a READY slab.
 */
static dsm_handle
HnswSlabDirGetReadyHandle(Oid dbid, Oid relid)
{
	HnswSlabDirKey key;
	HnswSlabDirEntry *entry;
	bool found;
	dsm_handle handle;

	if (HnswSlabDirHash == NULL || HnswSlabDirLock == NULL)
		elog(ERROR, "HNSW slab directory not initialized");

	key.dbid = dbid;
	key.relid = relid;

	LWLockAcquire(HnswSlabDirLock, LW_SHARED);
	entry = (HnswSlabDirEntry *) hash_search(HnswSlabDirHash, &key, HASH_FIND, &found);
	if (!found)
	{
		LWLockRelease(HnswSlabDirLock);
		elog(ERROR, "HNSW slab not found for index %u.%u - index must be rebuilt", dbid, relid);
	}
	if (entry->state != HNSW_SLAB_READY)
	{
		HnswSlabState state = entry->state;
		LWLockRelease(HnswSlabDirLock);
		elog(ERROR, "HNSW slab not ready for index %u.%u (state=%d) - index must be rebuilt",
			 dbid, relid, state);
	}
	handle = entry->handle;
	LWLockRelease(HnswSlabDirLock);

	if (handle == 0)
		elog(ERROR, "HNSW slab has invalid handle for index %u.%u - index must be rebuilt",
			 dbid, relid);

	return handle;
}

/*
 * Create or look up a slab directory entry.
 */
HnswSlabDirEntry *
HnswSlabDirCreate(Oid dbid, Oid relid, bool *found)
{
	HnswSlabDirKey key;
	HnswSlabDirEntry *entry;
	
	/* Check if slab directory is initialized */
	if (HnswSlabDirHash == NULL || HnswSlabDirLock == NULL)
	{
		*found = false;
		elog(ERROR, "HNSW slab directory not initialized - extension may need to be preloaded");
		return NULL;
	}
	
	key.dbid = dbid;
	key.relid = relid;
	
	LWLockAcquire(HnswSlabDirLock, LW_EXCLUSIVE);
	entry = (HnswSlabDirEntry *) hash_search(HnswSlabDirHash, &key, HASH_ENTER, found);
	
	if (!*found)
	{
		/* Initialize new entry */
		entry->state = HNSW_SLAB_BUILDING;
		entry->handle = 0;
		entry->magic = HNSW_SLAB_MAGIC;
		entry->version = HNSW_SLAB_VERSION;
		entry->totalBytes = 0;
		entry->buildXid = GetCurrentTransactionId();
		entry->m = 0;
		entry->dimensions = 0;
		entry->numElements = 0;
		entry->maxLevel = 0;
		entry->entryPointId = 0;
		entry->entryPointLevel = -1;
		entry->waitersCount = 0;
	}
	
	LWLockRelease(HnswSlabDirLock);
	
	return entry;
}

/*
 * Remove a slab directory entry.
 */
void
HnswSlabDirRemove(Oid dbid, Oid relid)
{
	HnswSlabDirKey key;
	
	/* Check if slab directory is initialized */
	if (HnswSlabDirHash == NULL || HnswSlabDirLock == NULL)
		return;
	
	key.dbid = dbid;
	key.relid = relid;
	
	LWLockAcquire(HnswSlabDirLock, LW_EXCLUSIVE);
	hash_search(HnswSlabDirHash, &key, HASH_REMOVE, NULL);
	LWLockRelease(HnswSlabDirLock);
}

/*
 * Remove an entry and return its DSM handle.
 */
static dsm_handle
HnswSlabDirRemoveAndGetHandle(Oid dbid, Oid relid)
{
	HnswSlabDirKey key;
	HnswSlabDirEntry *entry;
	bool found;
	dsm_handle handle = 0;

	if (HnswSlabDirHash == NULL || HnswSlabDirLock == NULL)
		return 0;

	key.dbid = dbid;
	key.relid = relid;

	LWLockAcquire(HnswSlabDirLock, LW_EXCLUSIVE);
	entry = (HnswSlabDirEntry *) hash_search(HnswSlabDirHash, &key, HASH_FIND, &found);
	if (found)
		handle = entry->handle;
	hash_search(HnswSlabDirHash, &key, HASH_REMOVE, NULL);
	LWLockRelease(HnswSlabDirLock);

	return handle;
}

/*
 * Mark a slab as READY after successful build.
 */
void
HnswSlabDirSetReady(Oid dbid, Oid relid, dsm_handle handle,
					uint32 numElements, int m, int dimensions,
					int maxLevel, uint32 entryPointId, int entryPointLevel,
					Size totalBytes)
{
	HnswSlabDirKey key;
	HnswSlabDirEntry *entry;
	bool found;
	
	/* Check if slab directory is initialized */
	if (HnswSlabDirHash == NULL || HnswSlabDirLock == NULL)
		return;
	
	key.dbid = dbid;
	key.relid = relid;
	
	LWLockAcquire(HnswSlabDirLock, LW_EXCLUSIVE);
	entry = (HnswSlabDirEntry *) hash_search(HnswSlabDirHash, &key, HASH_FIND, &found);
	
	if (found)
	{
		if (entry->state == HNSW_SLAB_BUILDING)
		{
			entry->state = HNSW_SLAB_READY;
			entry->handle = handle;
			entry->numElements = numElements;
			entry->m = m;
			entry->dimensions = dimensions;
			entry->maxLevel = maxLevel;
			entry->entryPointId = entryPointId;
			entry->entryPointLevel = entryPointLevel;
			entry->totalBytes = totalBytes;
		}
	}
	
	LWLockRelease(HnswSlabDirLock);
}

/*
 * Mark a slab as INVALID (after abort or error).
 */
void
HnswSlabDirSetInvalid(Oid dbid, Oid relid)
{
	HnswSlabDirKey key;
	HnswSlabDirEntry *entry;
	bool found;
	
	/* Check if slab directory is initialized */
	if (HnswSlabDirHash == NULL || HnswSlabDirLock == NULL)
		return;
	
	key.dbid = dbid;
	key.relid = relid;
	
	LWLockAcquire(HnswSlabDirLock, LW_EXCLUSIVE);
	entry = (HnswSlabDirEntry *) hash_search(HnswSlabDirHash, &key, HASH_FIND, &found);
	
	if (found)
	{
		entry->state = HNSW_SLAB_INVALID;
		entry->handle = 0;
	}
	
	LWLockRelease(HnswSlabDirLock);
}

/*
 * Estimate DSM size for slab.
 */
static Size
HnswSlabEstimateSize(int m, Size vectorStride, uint32 numElements, int maxLevel, uint64 totalUpperChunks)
{
	Size size = 0;
	Size adj0Size;
	Size adjUpperSize;
	Size nodeMetaSize;
	Size dataLevel0PerElement;
	int maxM0 = m * 2;
	Size upperChunkSize;
	int numLocks;
	Size labelLookupSize;
	Size deletedPoolSize;

	/* Header */
	size = add_size(size, MAXALIGN(sizeof(HnswSlabHeader)));

	/* 
	 * AOS Level 0 data: each element = [adj0] + [vector] + [nodeMeta]
	 * Layout within element:
	 *   offset 0: adj0 list ([count] + maxM0 neighbors + 1 padding)
	 *   offset adj0Size: vector data
	 *   offset adj0Size + vectorStride: node metadata
	 */
	adj0Size = MAXALIGN(sizeof(uint32) * (maxM0 + 2));
	nodeMetaSize = MAXALIGN(sizeof(HnswSlabNodeMeta));
	dataLevel0PerElement = adj0Size + MAXALIGN(vectorStride) + nodeMetaSize;
	size = add_size(size, (Size)numElements * dataLevel0PerElement);

	/* Upper layers adjacency: COMPACT layout using totalUpperChunks */
	/* Each chunk is [count] + m neighbors + 1 padding slot */
	upperChunkSize = MAXALIGN(sizeof(uint32) * (m + 2));
	adjUpperSize = totalUpperChunks * upperChunkSize;
	size = add_size(size, adjUpperSize);

	/* Node locks array for dynamic insert - limited to HNSW_SLAB_MAX_NODE_LOCKS */
	numLocks = Min((int)numElements, HNSW_SLAB_MAX_NODE_LOCKS);
	size = add_size(size, MAXALIGN(sizeof(LWLockPadded) * numLocks));

	/*
	 * Label lookup hash table.
	 * Size = 2 * numElements for ~50% load factor.
	 * Each entry is (uint64 label, uint32 internalId).
	 */
	labelLookupSize = MAXALIGN(sizeof(HnswSlabLabelEntry) * (Size)(numElements * 2 + 1));
	size = add_size(size, labelLookupSize);

	/*
	 * Deleted elements pool (ring buffer).
	 * Size = numElements to accommodate worst case where all elements are deleted.
	 */
	deletedPoolSize = MAXALIGN(sizeof(uint32) * (Size)(numElements + 1));
	size = add_size(size, deletedPoolSize);
	
	/* Add some padding for safety */
	size = add_size(size, 1024 * 1024);  /* 1MB safety margin */
	
	return size;
}

/*
 * Transaction callback for slab build cleanup.
 */
static void
HnswSlabXactCallback(XactEvent event, void *arg)
{
	HnswSlabBuildContext *ctx = (HnswSlabBuildContext *) arg;

	if (ctx == NULL)
		return;

	switch (event)
	{
		case XACT_EVENT_COMMIT:
			/* Publish slab on commit if finished */
			if (ctx->finished)
			{
				/* Already published in HnswSlabBuildFinish */
			}
			else
			{
				dsm_handle old = HnswSlabDirRemoveAndGetHandle(ctx->dbid, ctx->relid);
				if (old != 0)
					dsm_unpin_segment(old);
			}
			break;

		case XACT_EVENT_ABORT:
		case XACT_EVENT_PARALLEL_ABORT:
			/* Abort - clean up slab */
			if (ctx->segment != NULL)
			{
				dsm_detach(ctx->segment);
				ctx->segment = NULL;
			}
			{
				dsm_handle old = HnswSlabDirRemoveAndGetHandle(ctx->dbid, ctx->relid);
				if (old != 0)
					dsm_unpin_segment(old);
			}
			break;

		default:
			break;
	}

	/* Unregister callback */
	UnregisterXactCallback(HnswSlabXactCallback, ctx);
	ctx->registered = false;

	if (CurrentBuildContext == ctx)
		CurrentBuildContext = NULL;

	pfree(ctx);
}

/*
 * Begin building a slab.
 */
HnswSlabBuildContext *
HnswSlabBuildBegin(Oid dbid, Oid relid,
				   int m, int dimensions,
				   Size vectorStride,
				   uint32 estimatedElements,
				   int maxLevel,
				   uint64 totalUpperChunks,
				   bool isSq8Index,
				   float sq8LowerBound,
				   float sq8Diff)
{
	HnswSlabBuildContext *ctx;
	dsm_segment *segment;
	Size totalSize;
	HnswSlabHeader *header;
	Size offset;
	Size adj0Stride;
	Size adjUpperStride;
	Size nodeMetaStride;
	Size dataLevel0PerElement;
	int maxM0 = m * 2;
	bool found;
	Size upperChunkSize;
	dsm_handle oldHandle;
	MemoryContext oldCtx;

	/* Remove any existing entry and unpin its segment (if any) */
	oldHandle = HnswSlabDirRemoveAndGetHandle(dbid, relid);
	if (oldHandle != 0)
		dsm_unpin_segment(oldHandle);

	/* Create directory entry in BUILDING state */
	HnswSlabDirCreate(dbid, relid, &found);

	/* Estimate and allocate DSM segment using compact totalUpperChunks */
	totalSize = HnswSlabEstimateSize(m, vectorStride, estimatedElements, maxLevel, totalUpperChunks);

	segment = dsm_create(totalSize, 0);
	if (segment == NULL)
		elog(ERROR, "could not create DSM segment for HNSW slab (size %zu)", totalSize);
	
	/* Pin the segment so it persists across transactions */
	dsm_pin_segment(segment);
	
	/* Initialize header */
	header = (HnswSlabHeader *) dsm_segment_address(segment);
	MemSet(header, 0, sizeof(HnswSlabHeader));
	
	header->magic = HNSW_SLAB_MAGIC;
	header->version = HNSW_SLAB_VERSION;
	header->m = m;
	header->maxM0 = maxM0;
	header->dimensions = dimensions;
	header->numElements = 0;
	header->maxLevel = maxLevel;
	header->entryPointId = 0;
	header->entryPointLevel = -1;

	/* Initialize atomic counters for dynamic insert support */
	pg_atomic_init_u32(&header->curElementCount, 0);
	pg_atomic_init_u32(&header->numDeleted, 0);
	header->maxElements = estimatedElements;
	pg_atomic_init_u32(&header->entryPointIdAtomic, 0);
	pg_atomic_init_u32(&header->entryPointLevelAtomic, (uint32)-1);

	/* Initialize global spinlock */
	SpinLockInit(&header->globalLock);

	/* Calculate mult for level generation: 1/log(m) */
	header->mult = 1.0 / log((double)m);
	
	/* Calculate layout offsets */
	offset = MAXALIGN(sizeof(HnswSlabHeader));

	/*
	 * AOS Layout for Layer 0:
	 * Each element = [adj0 list] + [vector data] + [node metadata]
	 * 
	 * This layout improves cache locality - when we access a neighbor's
	 * adjacency list, its vector data is in the same or adjacent cache line.
	 */
	adj0Stride = MAXALIGN(sizeof(uint32) * (maxM0 + 2));
	nodeMetaStride = MAXALIGN(sizeof(HnswSlabNodeMeta));
	
	header->adj0Stride = adj0Stride;
	header->vectorStride = MAXALIGN(vectorStride);
	header->nodeMetaStride = nodeMetaStride;
	
	/* Offsets within each element */
	header->offsetDataInElement = adj0Stride;  /* vector comes after adj0 */
	header->offsetMetaInElement = adj0Stride + header->vectorStride;  /* meta comes after vector */
	
	/* Total bytes per element in level 0 */
	dataLevel0PerElement = adj0Stride + header->vectorStride + nodeMetaStride;
	header->dataLevel0PerElement = dataLevel0PerElement;
	
	/* Level 0 data pool (AOS) */
	header->dataLevel0Offset = offset;
	offset = add_size(offset, (Size)estimatedElements * dataLevel0PerElement);

	/* Upper layers adjacency pool - COMPACT layout using totalUpperChunks */
	header->adjUpperOffset = offset;
	upperChunkSize = MAXALIGN(sizeof(uint32) * (m + 2));
	/* adjUpperStride is now the size of one chunk (for per-layer access) */
	adjUpperStride = upperChunkSize;
	header->adjUpperStride = adjUpperStride;
	header->totalUpperChunks = totalUpperChunks;
	pg_atomic_init_u64(&header->usedUpperChunks, 0);  /* Will be set after build */
	offset = add_size(offset, totalUpperChunks * upperChunkSize);

	/* Node locks pool for dynamic insert */
	{
		int numLocks = Min((int)estimatedElements, HNSW_SLAB_MAX_NODE_LOCKS);
		int i;
		LWLockPadded *locks;

		header->nodeLockOffset = offset;
		header->nodeLockCount = (uint32) numLocks;
		offset = add_size(offset, MAXALIGN(sizeof(LWLockPadded) * numLocks));

		/* Initialize locks */
		locks = (LWLockPadded *)((char *)header + header->nodeLockOffset);
		for (i = 0; i < numLocks; i++)
			LWLockInitialize(&locks[i].lock, hnsw_lock_tranche_id);
	}

	/*
	 * Label lookup hash table.
	 * Uses open addressing with linear probing.
	 */
	{
		uint32 labelCapacity = estimatedElements * 2 + 1;
		HnswSlabLabelEntry *labelTable;
		uint32 i;

		header->labelLookupOffset = offset;
		header->labelLookupCapacity = labelCapacity;
		SpinLockInit(&header->labelLookupLock);
		offset = add_size(offset, MAXALIGN(sizeof(HnswSlabLabelEntry) * labelCapacity));

		/* Initialize all entries to empty (label=0 means empty) */
		labelTable = (HnswSlabLabelEntry *)((char *)header + header->labelLookupOffset);
		for (i = 0; i < labelCapacity; i++)
		{
			labelTable[i].label = 0;
			labelTable[i].internalId = 0;
		}
	}

	/*
	 * Deleted elements pool (ring buffer).
	 */
	{
		uint32 poolCapacity = estimatedElements + 1;

		header->deletedPoolOffset = offset;
		header->deletedPoolCapacity = poolCapacity;
		pg_atomic_init_u32(&header->deletedPoolHead, 0);
		pg_atomic_init_u32(&header->deletedPoolTail, 0);
		SpinLockInit(&header->deletedPoolLock);
		offset = add_size(offset, MAXALIGN(sizeof(uint32) * poolCapacity));

		/* Pool will be zero-initialized by MemSet earlier */
	}
	
	header->totalSize = totalSize;
	header->isSq8Index = isSq8Index;
	header->sq8LowerBound = sq8LowerBound;
	header->sq8Diff = sq8Diff;

	/* Create build context */
	oldCtx = MemoryContextSwitchTo(TopMemoryContext);
	ctx = palloc0(sizeof(HnswSlabBuildContext));
	MemoryContextSwitchTo(oldCtx);
	ctx->segment = segment;
	ctx->header = header;
	/* AOS layout - single pointer to level 0 data */
	ctx->dataLevel0 = (char *)header + header->dataLevel0Offset;
	ctx->adjUpperPool = (char *)header + header->adjUpperOffset;
	ctx->nextInternalId = 0;
	ctx->m = m;
	ctx->dimensions = dimensions;
	ctx->currentUpperOffset = 0;	/* Initialize compact upper adjacency offset */
	ctx->dbid = dbid;
	ctx->relid = relid;
	ctx->dsmHandle = dsm_segment_handle(segment);
	ctx->totalBytes = totalSize;
	ctx->finished = false;

	/* Store BUILDING handle in directory so abort can clean up */
	{
		HnswSlabDirKey key;
		HnswSlabDirEntry *entry;

		key.dbid = dbid;
		key.relid = relid;
		LWLockAcquire(HnswSlabDirLock, LW_EXCLUSIVE);
		entry = (HnswSlabDirEntry *) hash_search(HnswSlabDirHash, &key, HASH_FIND, &found);
		if (found)
		{
			entry->state = HNSW_SLAB_BUILDING;
			entry->handle = ctx->dsmHandle;
			entry->totalBytes = totalSize;
			entry->m = m;
			entry->dimensions = dimensions;
			entry->maxLevel = maxLevel;
			entry->numElements = 0;
		}
		LWLockRelease(HnswSlabDirLock);
	}

	/* Register transaction callback */
	RegisterXactCallback(HnswSlabXactCallback, ctx);
	ctx->registered = true;
	CurrentBuildContext = ctx;
	
	return ctx;
}

/*
 * Add a vector to the slab during build.
 * Returns the internal ID assigned to this vector.
 */
uint32
HnswSlabBuildAddVector(HnswSlabBuildContext *ctx, Pointer vectorData, Size vectorSize)
{
	uint32 internalId;
	char *elementBase;
	char *dest;

	internalId = ctx->nextInternalId++;
	ctx->header->numElements = ctx->nextInternalId;

	/* Get base address of this element in AOS layout */
	elementBase = ctx->dataLevel0 + (Size)internalId * ctx->header->dataLevel0PerElement;
	
	/* Vector is at offset offsetDataInElement within the element */
	dest = elementBase + ctx->header->offsetDataInElement;
	
	if (vectorSize > ctx->header->vectorStride)
		elog(ERROR, "HNSW slab build: vector size %zu exceeds stride %zu", vectorSize, ctx->header->vectorStride);
	memcpy(dest, vectorData, vectorSize);
	if (vectorSize < ctx->header->vectorStride)
		MemSet(dest + vectorSize, 0, ctx->header->vectorStride - vectorSize);

	return internalId;
}

/*
 * Get element base address for build context (helper function).
 */
static inline char *
HnswSlabBuildGetElementBase(HnswSlabBuildContext *ctx, uint32 internalId)
{
	return ctx->dataLevel0 + (Size)internalId * ctx->header->dataLevel0PerElement;
}

/*
 * Get node metadata for build context.
 */
static inline HnswSlabNodeMeta *
HnswSlabBuildGetNodeMeta(HnswSlabBuildContext *ctx, uint32 internalId)
{
	char *elementBase = HnswSlabBuildGetElementBase(ctx, internalId);
	return (HnswSlabNodeMeta *)(elementBase + ctx->header->offsetMetaInElement);
}

/*
 * Get adj0 for build context.
 */
static inline uint32 *
HnswSlabBuildGetAdj0(HnswSlabBuildContext *ctx, uint32 internalId)
{
	/* adj0 is at offset 0 within each element */
	return (uint32 *)HnswSlabBuildGetElementBase(ctx, internalId);
}

/*
 * Set node metadata for an internal ID.
 */
void
HnswSlabBuildSetNodeMeta(HnswSlabBuildContext *ctx,
						 uint32 internalId,
						 ItemPointerData *heaptids,
						 uint8 heaptidsLength,
						 uint8 level, uint8 deleted, uint8 version)
{
	HnswSlabNodeMeta *meta;
	int i;
	uint64 label;
	
	/* Get metadata from AOS layout */
	meta = HnswSlabBuildGetNodeMeta(ctx, internalId);
	meta->level = level;
	meta->deleted = deleted;
	meta->version = version;
	
	/* Allocate compact upper adjacency offset for nodes with level > 0 */
	if (level > 0)
	{
		meta->adjUpperOffset = (uint32)ctx->currentUpperOffset;
		ctx->currentUpperOffset += level;  /* Each node needs 'level' chunks for layers 1..level */
	}
	else
	{
		meta->adjUpperOffset = 0;
	}
	
	for (i = 0; i < HNSW_HEAPTIDS; i++)
	{
		if (i < heaptidsLength)
			meta->heaptids[i] = heaptids[i];
		else
			ItemPointerSetInvalid(&meta->heaptids[i]);
	}
	pg_write_barrier();
	meta->heaptidsLength = heaptidsLength;

	/*
	 * Build-time label mapping: insert primary heaptid (heaptids[0]) into label_lookup_.
	 */
	if (heaptidsLength > 0 && ctx->header->labelLookupOffset > 0)
	{
		label = HnswSlabEncodeHeaptid(&heaptids[0]);
		if (label != HNSW_SLAB_LABEL_EMPTY)
		{
			HnswSlabLabelEntry *table;
			uint32 capacity;
			uint32 idx;
			uint32 startIdx;
			bool inserted = false;

			table = (HnswSlabLabelEntry *)((char *)ctx->header + ctx->header->labelLookupOffset);
			capacity = ctx->header->labelLookupCapacity;
			idx = (uint32) (label % capacity);
			startIdx = idx;

			SpinLockAcquire(&ctx->header->labelLookupLock);
			do
			{
				if (table[idx].label == HNSW_SLAB_LABEL_EMPTY ||
					table[idx].label == HNSW_SLAB_LABEL_TOMBSTONE ||
					table[idx].label == label)
				{
					table[idx].label = label;
					table[idx].internalId = internalId;
					inserted = true;
					break;
				}
				idx = (idx + 1) % capacity;
			} while (idx != startIdx);
			SpinLockRelease(&ctx->header->labelLookupLock);

			if (!inserted)
				elog(WARNING, "HNSW slab build: label lookup table full");
		}
	}
}

/*
 * Set adjacency list for a node at a specific layer.
 */
void
HnswSlabBuildSetAdjacency(HnswSlabBuildContext *ctx,
						  uint32 internalId, int layer,
						  uint32 *neighborIds, int neighborCount)
{
	uint32 *adj;
	int i;
	int maxNeighbors;
	int totalSlots;
	Size upperChunkSize;

	if (layer == 0)
	{
		/* Layer 0 adjacency - at offset 0 within each element in AOS layout */
		adj = HnswSlabBuildGetAdj0(ctx, internalId);
		maxNeighbors = ctx->header->maxM0;
		totalSlots = maxNeighbors + 2;
	}
	else
	{
		/* Upper layer adjacency - use compact layout via adjUpperOffset */
		HnswSlabNodeMeta *meta = HnswSlabBuildGetNodeMeta(ctx, internalId);
		int layerIndex = layer - 1;
		Size offset;
		upperChunkSize = ctx->header->adjUpperStride;  /* Size of one chunk */
		offset = (Size)meta->adjUpperOffset * upperChunkSize + 
					  (Size)layerIndex * upperChunkSize;
		adj = (uint32 *)(ctx->adjUpperPool + offset);
		maxNeighbors = ctx->m;
		totalSlots = maxNeighbors + 2;
	}

	/* Format: [count][id1][id2]... */
	if (neighborCount > maxNeighbors)
		neighborCount = maxNeighbors;

	adj[0] = (uint32) neighborCount;
	for (i = 0; i < neighborCount; i++)
		adj[i + 1] = neighborIds[i];
	for (i = neighborCount + 1; i < totalSlots; i++)
		adj[i] = 0;
}

/*
 * Set entry point for the slab.
 */
void
HnswSlabBuildSetEntryPoint(HnswSlabBuildContext *ctx,
						   uint32 entryPointId, int entryPointLevel)
{
	ctx->header->entryPointId = entryPointId;
	ctx->header->entryPointLevel = entryPointLevel;

	/* Also update atomic versions for runtime use */
	pg_atomic_write_u32(&ctx->header->entryPointIdAtomic, entryPointId);
	pg_atomic_write_u32(&ctx->header->entryPointLevelAtomic, (uint32)entryPointLevel);
}

/*
 * Finish building the slab and publish it.
 * Called from transaction commit path.
 */
void
HnswSlabBuildFinish(HnswSlabBuildContext *ctx)
{
	if (ctx == NULL || ctx->segment == NULL)
		return;

	/* Sync atomic counters with build-time values */
	pg_atomic_write_u32(&ctx->header->curElementCount, ctx->header->numElements);
	pg_atomic_write_u32(&ctx->header->entryPointIdAtomic, ctx->header->entryPointId);
	pg_atomic_write_u32(&ctx->header->entryPointLevelAtomic, (uint32)ctx->header->entryPointLevel);
	
	/* Sync usedUpperChunks with build-time value (for dynamic insert) */
	pg_atomic_write_u64(&ctx->header->usedUpperChunks, ctx->currentUpperOffset);

	/* Publish slab immediately */
	HnswSlabDirSetReady(ctx->dbid, ctx->relid, ctx->dsmHandle,
						ctx->header->numElements,
						ctx->m,
						ctx->dimensions,
						ctx->header->maxLevel,
						ctx->header->entryPointId,
						ctx->header->entryPointLevel,
						ctx->header->totalSize);

	ctx->finished = true;

	/* Detach from segment (but segment remains pinned) */
	dsm_detach(ctx->segment);
	ctx->segment = NULL;
}

/*
 * Abort slab build and clean up.
 */
void
HnswSlabBuildAbort(HnswSlabBuildContext *ctx)
{
	if (ctx == NULL)
		return;
	
	if (ctx->segment != NULL)
	{
		dsm_detach(ctx->segment);
		ctx->segment = NULL;
	}

	{
		dsm_handle old = HnswSlabDirRemoveAndGetHandle(ctx->dbid, ctx->relid);
		if (old != 0)
			dsm_unpin_segment(old);
	}

	if (ctx->registered)
	{
		UnregisterXactCallback(HnswSlabXactCallback, ctx);
		ctx->registered = false;
	}
	
	if (CurrentBuildContext == ctx)
		CurrentBuildContext = NULL;

	pfree(ctx);
}

/*
 * Attach to an existing slab for queries.
 * Errors out if slab does not exist or is not READY.
 */
HnswSlabHandle *
HnswSlabAttach(Oid dbid, Oid relid)
{
	HnswSlabHandle *handle = NULL;
	dsm_segment *segment;
	HnswSlabHeader *header;
	dsm_handle dhandle;
	HnswSlabDirKey key;
	HnswSlabLocalEntry *cacheEntry;
	bool found;
	MemoryContext oldCtx;

	/* Fast path: local cache hit */
	HnswSlabLocalCacheInitIfNeeded();
	key.dbid = dbid;
	key.relid = relid;

	dhandle = HnswSlabDirGetReadyHandle(dbid, relid);

	cacheEntry = (HnswSlabLocalEntry *) hash_search(HnswSlabLocalCache, &key, HASH_ENTER, &found);
	if (found && cacheEntry->handle != NULL)
	{
		/* If the slab was rebuilt (DSM handle changed), drop cached mapping when safe */
		if (cacheEntry->dhandle != dhandle)
		{
			if (cacheEntry->handle->refcount != 0)
			{
				elog(ERROR, "HNSW slab handle changed while still in use for index %u.%u",
					 dbid, relid);
			}
			HnswSlabHandleFreeInternal(cacheEntry->handle);
			cacheEntry->handle = NULL;
		}
		else
		{
			cacheEntry->handle->refcount++;
			cacheEntry->handle->lastUsed = ++HnswSlabLocalClock;
			return cacheEntry->handle;
		}
	}

	/* Cache miss: attach to DSM and build a persistent handle in TopMemoryContext */
	segment = dsm_attach(dhandle);
	if (segment == NULL)
		elog(ERROR, "could not attach to HNSW slab DSM segment for index %u.%u - index must be rebuilt",
			 dbid, relid);

	/*
	 * Keep this mapping for the lifetime of the backend (local cache).
	 */
	dsm_pin_mapping(segment);

	header = (HnswSlabHeader *) dsm_segment_address(segment);
	
	/* Validate header */
	if (header->magic != HNSW_SLAB_MAGIC)
		elog(ERROR, "HNSW slab has invalid magic number for index %u.%u - index must be rebuilt",
			 dbid, relid);
	
	if (header->version != HNSW_SLAB_VERSION)
		elog(ERROR, "HNSW slab has incompatible version for index %u.%u - index must be rebuilt",
			 dbid, relid);
	
	oldCtx = MemoryContextSwitchTo(TopMemoryContext);

	/* Create handle */
	handle = palloc0(sizeof(HnswSlabHandle));
	handle->cacheDbid = dbid;
	handle->cacheRelid = relid;
	handle->cacheDsmHandle = dhandle;
	handle->refcount = 1;
	handle->lastUsed = ++HnswSlabLocalClock;
	handle->cached = true;

	handle->segment = segment;
	handle->header = header;
	
	/* AOS layout - single pointer to level 0 data */
	handle->dataLevel0 = (char *)header + header->dataLevel0Offset;
	handle->adjUpperPool = (char *)header + header->adjUpperOffset;
	
	handle->m = header->m;
	handle->maxM0 = header->maxM0;
	handle->dimensions = header->dimensions;
	
	/* AOS layout parameters */
	handle->dataLevel0PerElement = header->dataLevel0PerElement;
	handle->offsetDataInElement = header->offsetDataInElement;
	handle->offsetMetaInElement = header->offsetMetaInElement;
	handle->vectorStride = header->vectorStride;
	handle->adj0Stride = header->adj0Stride;
	handle->adjUpperStride = header->adjUpperStride;
	
	handle->entryPointId = header->entryPointId;
	handle->entryPointLevel = header->entryPointLevel;
	
	handle->isSq8Index = header->isSq8Index;
	handle->sq8LowerBound = header->sq8LowerBound;
	handle->sq8Diff = header->sq8Diff;
	if (header->isSq8Index && header->sq8Diff > 0)
	{
		handle->sq8Scale = header->sq8Diff / 255.0f;
		handle->sq8Scale2 = handle->sq8Scale * handle->sq8Scale;
	}

	/* Initialize dynamic insert support fields */
	handle->maxElements = header->maxElements;
	handle->mult = header->mult;
	if (header->nodeLockOffset > 0)
		handle->nodeLocks = (LWLockPadded *)((char *)header + header->nodeLockOffset);
	else
		handle->nodeLocks = NULL;
	handle->nodeLockCount = header->nodeLockCount;

	/* Initialize label lookup table pointer */
	if (header->labelLookupOffset > 0)
	{
		handle->labelLookup = (HnswSlabLabelEntry *)((char *)header + header->labelLookupOffset);
		handle->labelLookupCapacity = header->labelLookupCapacity;
	}
	else
	{
		handle->labelLookup = NULL;
		handle->labelLookupCapacity = 0;
	}

	/* Initialize deleted elements pool pointer */
	if (header->deletedPoolOffset > 0)
	{
		handle->deletedPool = (uint32 *)((char *)header + header->deletedPoolOffset);
		handle->deletedPoolCapacity = header->deletedPoolCapacity;
	}
	else
	{
		handle->deletedPool = NULL;
		handle->deletedPoolCapacity = 0;
	}
	
	/* Create per-backend visited array */
	handle->visited = HnswSlabVisitedCreate(TopMemoryContext, header->maxElements);

	MemoryContextSwitchTo(oldCtx);

	/* Publish into cache */
	cacheEntry->dhandle = dhandle;
	cacheEntry->handle = handle;

	HnswSlabLocalCacheMaybeEvict();
	
	return handle;
}

/*
 * Detach from a slab.
 */
void
HnswSlabDetach(HnswSlabHandle *handle)
{
	if (handle == NULL)
		return;

	/* Cached handles are retained for reuse; just drop the refcount */
	if (handle->cached)
	{
		if (handle->refcount > 0)
			handle->refcount--;
		return;
	}

	/* Non-cached handle */
	HnswSlabHandleFreeInternal(handle);
}

/*
 * Create per-backend visited array.
 */
HnswSlabVisited *
HnswSlabVisitedCreate(MemoryContext ctx, uint32 capacity)
{
	HnswSlabVisited *visited;
	MemoryContext oldCtx;
	
	oldCtx = MemoryContextSwitchTo(ctx);
	
	visited = palloc0(sizeof(HnswSlabVisited));
	visited->currentTag = 1;
	visited->capacity = capacity;
	visited->tags = palloc0(sizeof(SlabVisitedTag) * capacity);
	visited->ctx = ctx;
	
	MemoryContextSwitchTo(oldCtx);
	
	return visited;
}

/*
 * Destroy per-backend visited array.
 */
void
HnswSlabVisitedDestroy(HnswSlabVisited *visited)
{
	if (visited == NULL)
		return;
	
	if (visited->tags != NULL)
		pfree(visited->tags);
	
	pfree(visited);
}

/*
 * Pairing heap comparison for candidates (min-heap by distance for C).
 */
static int
SlabCompareNearestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	const HnswSearchCandidate *ca = HnswGetSearchCandidateConst(c_node, a);
	const HnswSearchCandidate *cb = HnswGetSearchCandidateConst(c_node, b);
	
	if (ca->distance < cb->distance)
		return 1;
	if (ca->distance > cb->distance)
		return -1;
	return 0;
}

/*
 * Pairing heap comparison for top candidates (max-heap by distance for W).
 */
static int
SlabCompareFurthestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	const HnswSearchCandidate *ca = HnswGetSearchCandidateConst(w_node, a);
	const HnswSearchCandidate *cb = HnswGetSearchCandidateConst(w_node, b);
	
	if (ca->distance > cb->distance)
		return 1;
	if (ca->distance < cb->distance)
		return -1;
	return 0;
}

/*
 * Pairing heap comparison for discarded candidates (min-heap by distance).
 * Uses w_node, matching how discarded is accessed in scan.
 */
static int
SlabCompareNearestDiscardedCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	const HnswSearchCandidate *ca = HnswGetSearchCandidateConst(w_node, a);
	const HnswSearchCandidate *cb = HnswGetSearchCandidateConst(w_node, b);

	if (ca->distance < cb->distance)
		return 1;
	if (ca->distance > cb->distance)
		return -1;
	return 0;
}

/*
 * Prefetch helpers.
 */
static inline void
HnswSlabPrefetchAdjTail(uint32 *adj, uint32 neighborCount)
{
	/* Cache line is 64B = 16x uint32; neighbors start at adj[1] */
	if (neighborCount > 15)
		HNSW_SLAB_PREFETCH_R(adj + 16);
}

static inline void
HnswSlabPrefetchCandidateSetTopAdj(HnswSlabHandle *handle, pairingheap *candidate_set, int layer)
{
	HnswSearchCandidate *top;
	uint32 topId;
	uint32 *adj;

	if (pairingheap_is_empty(candidate_set))
		return;

	top = HnswGetSearchCandidate(c_node, pairingheap_first(candidate_set));
	topId = (uint32) (uintptr_t) top->element.ptr;
	if (topId >= handle->maxElements)
		return;

	adj = HnswSlabGetAdjAtLevel(handle, topId, layer);
	if (adj != NULL)
		HNSW_SLAB_PREFETCH_R(adj);
}

static inline uint32
HnswSlabClampNeighborCount(HnswSlabHandle *handle, int layer, uint32 count)
{
	uint32 maxNeighbors;

	maxNeighbors = (layer == 0) ? (uint32) handle->maxM0 : (uint32) handle->m;
	if (count > maxNeighbors)
		return maxNeighbors;
	return count;
}

/*
 * Compute distance between query and slab vector.
 */
static inline double
HnswSlabComputeDistance(HnswSlabHandle *handle, Datum queryValue,
						uint32 internalId, HnswSupport *support)
{
	/* Defensive: avoid OOB vector access if adjacency contains garbage */
	if (!HnswSlabIsValidInternalId(handle, internalId))
		return DBL_MAX;

	{
		char *vectorData = HnswSlabGetVector(handle, internalId);
		Datum vectorDatum = PointerGetDatum(vectorData);
		
		return HnswComputeDistance(queryValue, vectorDatum, support);
	}
}

/*
 * Search base layer using slab
 */
List *
HnswSlabSearchBaseLayer(HnswSlabHandle *handle,
						HnswSlabVisited *visited,
						Datum queryValue,
						HnswSupport *support,
						uint32 epId,
						int ef,
						pairingheap **discarded,
						bool resetVisited)
{
	pairingheap *candidate_set;    
	pairingheap *top_candidates;    
	double lowerBound;
	List *result = NIL;
	HnswSearchCandidate *epCandidate;
	int topCount = 0;  /* Track size of top_candidates */
	uint32 maxElements = handle->maxElements;
	uint32 curElementCount = pg_atomic_read_u32(&handle->header->curElementCount);
	uint64 loopCounter = 0;
	
	/* Reset visited if requested */
	if (resetVisited)
		HnswSlabVisitedReset(visited);
	
	/* Initialize heaps */
	candidate_set = pairingheap_allocate(SlabCompareNearestCandidates, NULL);
	top_candidates = pairingheap_allocate(SlabCompareFurthestCandidates, NULL);

	/* Initialize discarded heap if needed */
	if (discarded != NULL && *discarded == NULL)
		*discarded = pairingheap_allocate(SlabCompareNearestDiscardedCandidates, NULL);
	
	/* Check entry point validity */
	if (!HnswSlabIsValidInternalId(handle, epId))
		return NIL;
	
	/* Mark entry point as visited */
	HnswSlabVisitedCheckAndMark(visited, epId);
	
	/* Create entry point candidate */
	epCandidate = HnswSearchPoolAlloc(support != NULL ? support->searchPool : NULL);
	epCandidate->distance = HnswSlabComputeDistance(handle, queryValue, epId, support);
	if (!isfinite(epCandidate->distance))
		epCandidate->distance = DBL_MAX;
	/* Store internal ID in element pointer*/
	epCandidate->element.ptr = (HnswElementData *)(uintptr_t)epId;
	
	/* Initialize lowerBound and add to heaps */
	lowerBound = epCandidate->distance;
	pairingheap_add(top_candidates, &epCandidate->w_node);
	topCount = 1;
	pairingheap_add(candidate_set, &epCandidate->c_node);
	
	/* Main search loop */
	while (!pairingheap_is_empty(candidate_set))
	{
		HnswSearchCandidate *current;
		double candidate_dist;
		uint32 current_node_id;
		uint32 *adj;
		uint32 size;
		uint32 j;

		if ((loopCounter++ & 0x3FF) == 0)
			CHECK_FOR_INTERRUPTS();
			
		current = HnswGetSearchCandidate(c_node, pairingheap_remove_first(candidate_set));
		candidate_dist = current->distance;

		/* Stop condition: c->distance > f->distance (Algorithm 2) */
		if (candidate_dist > lowerBound)
			break;

		/* Prefetch next candidate's adjacency (next loop iteration) */
		if (!pairingheap_is_empty(candidate_set))
		{
			HnswSearchCandidate *nextCurrent = HnswGetSearchCandidate(c_node, pairingheap_first(candidate_set));
			uint32 nextId = (uint32)(uintptr_t)nextCurrent->element.ptr;
			if (nextId < maxElements)
				HNSW_SLAB_PREFETCH_R(HnswSlabGetAdj0(handle, nextId));
		}
		
		current_node_id = (uint32)(uintptr_t)current->element.ptr;
		if (!HnswSlabIsValidInternalId(handle, current_node_id))
			continue;
	
		/* Get adjacency list for layer 0 */
		adj = HnswSlabGetAdj0(handle, current_node_id);
		size = HnswSlabClampNeighborCount(handle, 0, HnswSlabGetAdjCount(adj));

		if (size > 0)
		{
			uint32 first_neighbor = HnswSlabGetAdjNeighbor(adj, 0);
			if (first_neighbor < visited->capacity)
			{
				uint32 pf2 = first_neighbor + 32;
				HNSW_SLAB_PREFETCH_R(visited->tags + first_neighbor);
				if (pf2 < visited->capacity)
					HNSW_SLAB_PREFETCH_R(visited->tags + pf2);
			}
			if (first_neighbor < handle->maxElements)
				HNSW_SLAB_PREFETCH_R(HnswSlabGetVector(handle, first_neighbor));
			HnswSlabPrefetchAdjTail(adj, size);
		}
		
		/* Process neighbors */
		if (support->distanceKind == HNSW_DISTANCE_SQ8_L2 && support->sq8EncodedQuery != NULL)
		{
			int batch_idx = 0;
			uint32 batch_ids[4];
			const uint8 *batch_vectors[4];
			int32 batch_sum2s[4];  /* Precomputed sum2 for each target */
			int64 batch_results[4];

			for (j = 0; j < size; j++)
			{
				uint32 candidate_id = HnswSlabGetAdjNeighbor(adj, j);

#if HNSW_SLAB_PREFETCH_DISTANCE > 0
				if (j + HNSW_SLAB_PREFETCH_DISTANCE < size)
				{
					uint32 next_id = HnswSlabGetAdjNeighbor(adj, (int) (j + HNSW_SLAB_PREFETCH_DISTANCE));
					if (next_id < visited->capacity)
						HNSW_SLAB_PREFETCH_R(visited->tags + next_id);
					if (next_id < handle->maxElements)
						HNSW_SLAB_PREFETCH_R(HnswSlabGetVector(handle, next_id));
				}
#endif

				/* Check visited */
				if (HnswSlabVisitedCheckAndMark(visited, candidate_id))
					continue;

				if (candidate_id >= maxElements || candidate_id >= curElementCount)
					continue;

				/* Add to batch - collect both codes and sum2 */
				batch_ids[batch_idx] = candidate_id;
				{
					char *vectorData = HnswSlabGetVector(handle, candidate_id);
					Sq8IndexTuple *qt = (Sq8IndexTuple *) vectorData;
					batch_vectors[batch_idx] = qt->codes;
					batch_sum2s[batch_idx] = qt->sum2;
				}
				batch_idx++;

				if (batch_idx == 4)
				{
					int k;
					/* Use optimized L2 with precomputed sum2: L2² = q_sum2 + t_sum2 - 2*dot */
					Sq8L2SquaredDistanceWithSum2Batch(support->sq8EncodedQueryDim, support->sq8EncodedQuery, 
													  support->sq8QuerySum, support->sq8QuerySum2,
													  batch_vectors, batch_sum2s, 4, batch_results);

					for (k = 0; k < 4; k++)
					{
						double dist = (double) batch_results[k];
						uint32 cid = batch_ids[k];

						/* Check if we should consider this candidate */
						if (topCount < ef ||
							HnswGetSearchCandidate(w_node, pairingheap_first(top_candidates))->distance > dist)
						{
							HnswSearchCandidate *newCandidate = HnswSearchPoolAlloc(support != NULL ? support->searchPool : NULL);
							newCandidate->distance = dist;
							newCandidate->element.ptr = (HnswElementData *)(uintptr_t)cid;

							pairingheap_add(candidate_set, &newCandidate->c_node);
							HnswSlabPrefetchCandidateSetTopAdj(handle, candidate_set, 0);
							pairingheap_add(top_candidates, &newCandidate->w_node);
							topCount++;

							while (topCount > ef)
							{
								HnswSearchCandidate *removed = HnswGetSearchCandidate(w_node,
									pairingheap_remove_first(top_candidates));
								topCount--;
								if (discarded != NULL)
									pairingheap_add(*discarded, &removed->w_node);
							}

							if (!pairingheap_is_empty(top_candidates))
								lowerBound = HnswGetSearchCandidate(w_node, pairingheap_first(top_candidates))->distance;
						}
						else if (discarded != NULL)
						{
							HnswSearchCandidate *wCandidate = HnswSearchPoolAlloc(support != NULL ? support->searchPool : NULL);
							wCandidate->distance = dist;
							wCandidate->element.ptr = (HnswElementData *)(uintptr_t)cid;
							pairingheap_add(*discarded, &wCandidate->w_node);
						}
					}
					batch_idx = 0;
				}
			}

			/* Process remaining */
			if (batch_idx > 0)
			{
				int k;
				/* Use optimized L2 with precomputed sum2: L2² = q_sum2 + t_sum2 - 2*dot */
				Sq8L2SquaredDistanceWithSum2Batch(support->sq8EncodedQueryDim, support->sq8EncodedQuery, 
												  support->sq8QuerySum, support->sq8QuerySum2,
												  batch_vectors, batch_sum2s, batch_idx, batch_results);

				for (k = 0; k < batch_idx; k++)
				{
					double dist = (double) batch_results[k];
					uint32 cid = batch_ids[k];

					if (topCount < ef ||
						HnswGetSearchCandidate(w_node, pairingheap_first(top_candidates))->distance > dist)
						{
							HnswSearchCandidate *newCandidate = HnswSearchPoolAlloc(support != NULL ? support->searchPool : NULL);
							newCandidate->distance = dist;
							newCandidate->element.ptr = (HnswElementData *)(uintptr_t)cid;

							pairingheap_add(candidate_set, &newCandidate->c_node);
							HnswSlabPrefetchCandidateSetTopAdj(handle, candidate_set, 0);
							pairingheap_add(top_candidates, &newCandidate->w_node);
							topCount++;

						while (topCount > ef)
						{
							HnswSearchCandidate *removed = HnswGetSearchCandidate(w_node,
								pairingheap_remove_first(top_candidates));
							topCount--;
							if (discarded != NULL)
								pairingheap_add(*discarded, &removed->w_node);
						}

						if (!pairingheap_is_empty(top_candidates))
							lowerBound = HnswGetSearchCandidate(w_node, pairingheap_first(top_candidates))->distance;
					}
					else if (discarded != NULL)
					{
						HnswSearchCandidate *wCandidate = HnswSearchPoolAlloc(support != NULL ? support->searchPool : NULL);
						wCandidate->distance = dist;
						wCandidate->element.ptr = (HnswElementData *)(uintptr_t)cid;
						pairingheap_add(*discarded, &wCandidate->w_node);
					}
				}
			}
		}
		else
		{
			for (j = 0; j < size; j++)
			{
				uint32 candidate_id = HnswSlabGetAdjNeighbor(adj, j);
				HnswSearchCandidate *newCandidate;
				double dist;
				
#if HNSW_SLAB_PREFETCH_DISTANCE > 0
				if (j + HNSW_SLAB_PREFETCH_DISTANCE < size)
				{
					uint32 next_id = HnswSlabGetAdjNeighbor(adj, (int) (j + HNSW_SLAB_PREFETCH_DISTANCE));
					if (next_id < visited->capacity)
						HNSW_SLAB_PREFETCH_R(visited->tags + next_id);
					if (next_id < handle->maxElements)
						HNSW_SLAB_PREFETCH_R(HnswSlabGetVector(handle, next_id));
				}
#endif
			
				/* Check visited */
				if (HnswSlabVisitedCheckAndMark(visited, candidate_id))
					continue;

				if (candidate_id >= maxElements || candidate_id >= curElementCount)
					continue;

				/* Compute distance */
				{
					char *vectorData = HnswSlabGetVector(handle, candidate_id);
					Datum vectorDatum = PointerGetDatum(vectorData);
					dist = HnswComputeDistance(queryValue, vectorDatum, support);
					if (!isfinite(dist))
						dist = DBL_MAX;
				}
				
				/* Check if we should consider this candidate */
				if (topCount < ef ||
					HnswGetSearchCandidate(w_node, pairingheap_first(top_candidates))->distance > dist)
				{
					/* Create new candidate */
					newCandidate = HnswSearchPoolAlloc(support != NULL ? support->searchPool : NULL);
					newCandidate->distance = dist;
					newCandidate->element.ptr = (HnswElementData *)(uintptr_t)candidate_id;
					
					pairingheap_add(candidate_set, &newCandidate->c_node);
					HnswSlabPrefetchCandidateSetTopAdj(handle, candidate_set, 0);
					pairingheap_add(top_candidates, &newCandidate->w_node);
					topCount++;
					
					/* Prune if exceeds ef */
					while (topCount > ef)
					{
						HnswSearchCandidate *removed = HnswGetSearchCandidate(w_node,
							pairingheap_remove_first(top_candidates));
						topCount--;
						if (discarded != NULL)
							pairingheap_add(*discarded, &removed->w_node);
					}

					/* Update lowerBound */
					if (!pairingheap_is_empty(top_candidates))
						lowerBound = HnswGetSearchCandidate(w_node, pairingheap_first(top_candidates))->distance;
				}
				else if (discarded != NULL)
				{
					HnswSearchCandidate *wCandidate = HnswSearchPoolAlloc(support != NULL ? support->searchPool : NULL);
					wCandidate->distance = dist;
					wCandidate->element.ptr = (HnswElementData *)(uintptr_t)candidate_id;
					pairingheap_add(*discarded, &wCandidate->w_node);
				}
			}
		}
	}
	
	/* Convert top_candidates to result list */
	while (!pairingheap_is_empty(top_candidates))
	{
		HnswSearchCandidate *sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(top_candidates));
		result = lappend(result, sc);
	}
	
	return result;
}

/*
 * Greedy search through upper layers to find best entry point for base layer.
 * Returns the internal ID of the best candidate at targetLayer.
 */
uint32
HnswSlabSearchUpperLayers(HnswSlabHandle *handle,
						  Datum queryValue,
						  HnswSupport *support,
						  uint32 epId,
						  int epLevel,
						  int targetLayer)
{
	uint32 currentId = epId;
	double currentDist;
	int lc;
	uint32 maxElements = handle->maxElements;
	uint32 curElementCount = pg_atomic_read_u32(&handle->header->curElementCount);
	uint64 loopCounter = 0;

	if (currentId >= maxElements || currentId >= curElementCount)
		return epId;

	{
		char *vectorData = HnswSlabGetVector(handle, currentId);
		Datum vectorDatum = PointerGetDatum(vectorData);
		currentDist = HnswComputeDistance(queryValue, vectorDatum, support);
	}
	
	/* Greedy descent from epLevel down to targetLayer */
	for (lc = epLevel; lc > targetLayer; lc--)
	{
		bool improved = true;
		
		while (improved)
		{
			uint32 *adj;
			uint32 size;
			uint32 j;

			if ((loopCounter++ & 0x3FF) == 0)
				CHECK_FOR_INTERRUPTS();
				
			improved = false;
			
			/* Get adjacency for this layer */
			adj = HnswSlabGetAdjUpper(handle, currentId, lc);
			if (adj == NULL)
				break;
			
			size = HnswSlabClampNeighborCount(handle, lc, HnswSlabGetAdjCount(adj));

		if (size > 0)
		{
			uint32 first_neighbor = HnswSlabGetAdjNeighbor(adj, 0);
			if (first_neighbor < maxElements)
			{
					HNSW_SLAB_PREFETCH_R(HnswSlabGetNodeMeta(handle, first_neighbor));
					HNSW_SLAB_PREFETCH_R(HnswSlabGetVector(handle, first_neighbor));
				}
				HnswSlabPrefetchAdjTail(adj, size);
			}
			
			/* Find closest neighbor */
			for (j = 0; j < size; j++)
			{
				uint32 neighborId = HnswSlabGetAdjNeighbor(adj, j);
				double neighborDist;
				HnswSlabNodeMeta *neighborMeta;

#if HNSW_SLAB_PREFETCH_DISTANCE > 0
				if (j + HNSW_SLAB_PREFETCH_DISTANCE < size)
				{
					uint32 pf_id = HnswSlabGetAdjNeighbor(adj, (int) (j + HNSW_SLAB_PREFETCH_DISTANCE));
					if (pf_id < maxElements)
					{
						HNSW_SLAB_PREFETCH_R(HnswSlabGetNodeMeta(handle, pf_id));
						HNSW_SLAB_PREFETCH_R(HnswSlabGetVector(handle, pf_id));
					}
				}
#endif

				if (neighborId >= maxElements || neighborId >= curElementCount)
					continue;
				
				neighborMeta = HnswSlabGetNodeMeta(handle, neighborId);
				if (neighborMeta->level < lc)
					continue;  /* Neighbor doesn't exist at this layer */
				
				{
					char *vectorData = HnswSlabGetVector(handle, neighborId);
					Datum vectorDatum = PointerGetDatum(vectorData);
					neighborDist = HnswComputeDistance(queryValue, vectorDatum, support);
				}
				
				if (neighborDist < currentDist)
				{
					currentId = neighborId;
					currentDist = neighborDist;
					improved = true;
				}
			}
		}
	}
	
	return currentId;
}

/*
 * Resize visited array if capacity is too small.
 */
void
HnswSlabVisitedResize(HnswSlabVisited *visited, uint32 newCapacity)
{
	SlabVisitedTag *newTags;
	MemoryContext oldCtx;

	if (visited->capacity >= newCapacity)
		return;

	oldCtx = MemoryContextSwitchTo(visited->ctx);
	newTags = palloc0(sizeof(SlabVisitedTag) * newCapacity);
	MemoryContextSwitchTo(oldCtx);

	/* Copy old tags */
	if (visited->tags != NULL)
	{
		memcpy(newTags, visited->tags, sizeof(SlabVisitedTag) * visited->capacity);
		pfree(visited->tags);
	}

	visited->tags = newTags;
	visited->capacity = newCapacity;
}

/*
 * Generate random level for new element
 */
int
HnswSlabGetRandomLevel(HnswSlabHandle *handle)
{
	double r = (double)random() / (double)HNSW_SLAB_MAX_RANDOM_VALUE;
	double mult = handle->mult;
	int level;
	int maxLevel = handle->header->maxLevel;

	if (r == 0.0)
		r = 1e-9;

	level = (int)(-log(r) * mult);
	if (level < 0)
		level = 0;
	if (maxLevel < 0)
		maxLevel = 0;
	if (maxLevel > HNSW_SLAB_MAX_LEVEL - 1)
		maxLevel = HNSW_SLAB_MAX_LEVEL - 1;
	if (level > maxLevel)
		level = maxLevel;

	return level;
}

/*
 * Search base layer for insert at a given layer .
 */
static List *
HnswSlabSearchBaseLayerForInsert(HnswSlabHandle *handle,
								 HnswSlabVisited *visited,
								 Datum queryValue,
								 HnswSupport *support,
								 uint32 epId,
								 int layer,
								 int efConstruction)
{
	pairingheap *candidate_set;     /* min-heap of candidates to explore */
	pairingheap *top_candidates;    /* max-heap of best results found */
	double lowerBound;
	List *result = NIL;
	HnswSearchCandidate *epCandidate;
	HnswSlabNodeMeta *epMeta;
	int topCount = 0;

	HnswSlabVisitedReset(visited);

	candidate_set = pairingheap_allocate(SlabCompareNearestCandidates, NULL);
	top_candidates = pairingheap_allocate(SlabCompareFurthestCandidates, NULL);

	epMeta = HnswSlabGetNodeMeta(handle, epId);

	/* Mark entry point as visited */
	HnswSlabVisitedCheckAndMark(visited, epId);

	/* Create entry point candidate */
	epCandidate = palloc(sizeof(HnswSearchCandidate));
	if (!epMeta->deleted)
		epCandidate->distance = HnswSlabComputeDistance(handle, queryValue, epId, support);
	else
		epCandidate->distance = DBL_MAX;
	epCandidate->element.ptr = (HnswElementData *)(uintptr_t)epId;

	/* Initialize lowerBound and add to heaps */
	if (!epMeta->deleted)
	{
		lowerBound = epCandidate->distance;
		pairingheap_add(top_candidates, &epCandidate->w_node);
		topCount = 1;
	}
	else
	{
		lowerBound = DBL_MAX;
	}
	pairingheap_add(candidate_set, &epCandidate->c_node);

	while (!pairingheap_is_empty(candidate_set))
	{
		HnswSearchCandidate *current;
		double candidate_dist;
		uint32 current_node_id;
		uint32 *adj;
		uint32 size;
		uint32 j;
		LWLock *lock;

		current = HnswGetSearchCandidate(c_node, pairingheap_first(candidate_set));
		candidate_dist = current->distance;

		if (candidate_dist > lowerBound && topCount == efConstruction)
			break;

		pairingheap_remove_first(candidate_set);

		/* Prefetch next candidate's adjacency (next loop iteration) */
		if (!pairingheap_is_empty(candidate_set))
		{
			HnswSearchCandidate *nextCurrent = HnswGetSearchCandidate(c_node, pairingheap_first(candidate_set));
			uint32 nextId = (uint32)(uintptr_t)nextCurrent->element.ptr;
			if (nextId < handle->maxElements)
			{
				uint32 *nextAdj = HnswSlabGetAdjAtLevel(handle, nextId, layer);
				if (nextAdj != NULL)
					HNSW_SLAB_PREFETCH_R(nextAdj);
			}
		}

		current_node_id = (uint32)(uintptr_t)current->element.ptr;

		lock = HnswSlabGetNodeLock(handle, current_node_id);
		if (lock != NULL)
			LWLockAcquire(lock, LW_EXCLUSIVE);

		adj = HnswSlabGetAdjAtLevel(handle, current_node_id, layer);
		if (adj == NULL)
		{
			if (lock != NULL)
				LWLockRelease(lock);
			continue;
		}
		size = HnswSlabClampNeighborCount(handle, layer, HnswSlabGetAdjCount(adj));

		if (size > 0)
		{
			uint32 first_neighbor = HnswSlabGetAdjNeighbor(adj, 0);
			if (first_neighbor < visited->capacity)
			{
				uint32 pf2 = first_neighbor + 32;
				HNSW_SLAB_PREFETCH_R(visited->tags + first_neighbor);
				if (pf2 < visited->capacity)
					HNSW_SLAB_PREFETCH_R(visited->tags + pf2);
			}
			if (first_neighbor < handle->maxElements)
				HNSW_SLAB_PREFETCH_R(HnswSlabGetVector(handle, first_neighbor));
			HnswSlabPrefetchAdjTail(adj, size);
		}

		for (j = 0; j < size; j++)
		{
			uint32 candidate_id = HnswSlabGetAdjNeighbor(adj, j);
			HnswSearchCandidate *newCandidate;
			double dist;
			HnswSlabNodeMeta *candidateMeta;

#if HNSW_SLAB_PREFETCH_DISTANCE > 0
			if (j + HNSW_SLAB_PREFETCH_DISTANCE < size)
			{
				uint32 next_id = HnswSlabGetAdjNeighbor(adj, (int) (j + HNSW_SLAB_PREFETCH_DISTANCE));
				if (next_id < visited->capacity)
					HNSW_SLAB_PREFETCH_R(visited->tags + next_id);
				if (next_id < handle->maxElements)
					HNSW_SLAB_PREFETCH_R(HnswSlabGetVector(handle, next_id));
			}
#endif

			if (HnswSlabVisitedCheckAndMark(visited, candidate_id))
				continue;

			dist = HnswSlabComputeDistance(handle, queryValue, candidate_id, support);

			if (topCount < efConstruction ||
				HnswGetSearchCandidate(w_node, pairingheap_first(top_candidates))->distance > dist)
			{
				newCandidate = palloc(sizeof(HnswSearchCandidate));
				newCandidate->distance = dist;
				newCandidate->element.ptr = (HnswElementData *)(uintptr_t)candidate_id;

				pairingheap_add(candidate_set, &newCandidate->c_node);
				HnswSlabPrefetchCandidateSetTopAdj(handle, candidate_set, layer);

				candidateMeta = HnswSlabGetNodeMeta(handle, candidate_id);
				if (!candidateMeta->deleted)
				{
					HnswSearchCandidate *wCandidate = palloc(sizeof(HnswSearchCandidate));
					wCandidate->distance = dist;
					wCandidate->element.ptr = (HnswElementData *)(uintptr_t)candidate_id;
					pairingheap_add(top_candidates, &wCandidate->w_node);
					topCount++;
				}

				while (topCount > efConstruction)
				{
					HnswSearchCandidate *removed = HnswGetSearchCandidate(w_node,
						pairingheap_remove_first(top_candidates));
					topCount--;
					/* Not reused in insertion path; keep allocation in context */
					(void) removed;
				}

				if (!pairingheap_is_empty(top_candidates))
					lowerBound = HnswGetSearchCandidate(w_node, pairingheap_first(top_candidates))->distance;
			}
		}

		if (lock != NULL)
			LWLockRelease(lock);
	}

	while (!pairingheap_is_empty(top_candidates))
	{
		HnswSearchCandidate *sc = HnswGetSearchCandidate(w_node, pairingheap_remove_first(top_candidates));
		result = lappend(result, sc);
	}

	return result;
}

/*
 * Candidate for heuristic pruning (distance to base element + internal ID).
 */
typedef struct HnswSlabHeuristicCandidate
{
	double distance;
	uint32 internalId;
} HnswSlabHeuristicCandidate;

static int
SlabCompareHeuristicCandidateAsc(const void *a, const void *b)
{
	const HnswSlabHeuristicCandidate *ca = (const HnswSlabHeuristicCandidate *) a;
	const HnswSlabHeuristicCandidate *cb = (const HnswSlabHeuristicCandidate *) b;

	if (ca->distance < cb->distance)
		return -1;
	if (ca->distance > cb->distance)
		return 1;
	return 0;
}

/*
 * getNeighborsByHeuristic2
 */
static int
HnswSlabGetNeighborsByHeuristic2(HnswSlabHandle *handle,
								 HnswSlabHeuristicCandidate *candidates,
								 int candidateCount,
								 int M,
								 HnswSupport *support)
{
	HnswSlabHeuristicCandidate selected[HNSW_MAX_M * 2 + 1];
	int selectedCount = 0;
	int i;

	if (candidateCount < M)
		return candidateCount;

	qsort(candidates, candidateCount, sizeof(HnswSlabHeuristicCandidate), SlabCompareHeuristicCandidateAsc);

	for (i = 0; i < candidateCount && selectedCount < M; i++)
	{
		const HnswSlabHeuristicCandidate *current = &candidates[i];
		bool good = true;
		int j;

		for (j = 0; j < selectedCount; j++)
		{
			double curdist;
			Datum a = PointerGetDatum(HnswSlabGetVector(handle, selected[j].internalId));
			Datum b = PointerGetDatum(HnswSlabGetVector(handle, current->internalId));

			curdist = HnswComputeDistance(a, b, support);
			if (curdist < current->distance)
			{
				good = false;
				break;
			}
		}

		if (good)
			selected[selectedCount++] = *current;
	}

	/* Write back in descending order (furthest->closest) */
	for (i = 0; i < selectedCount; i++)
		candidates[i] = selected[selectedCount - 1 - i];

	return selectedCount;
}

/*
 * Mutually connect new element
 */
static uint32
HnswSlabMutuallyConnectNewElement(HnswSlabHandle *handle,
								  uint32 newId,
								  List *topCandidates,
								  int level,
								  HnswSupport *support,
								  bool isUpdate)
{
	HnswSlabHeuristicCandidate *candidates;
	int candidateCount;
	int neighborCount;
	int maxM = handle->m;
	int i;
	uint32 *adj;
	uint32 nextClosest;
	ListCell *lc;
	LWLock *newLock = NULL;

	candidateCount = list_length(topCandidates);
	if (candidateCount <= 0)
		return newId;

	candidates = palloc(sizeof(HnswSlabHeuristicCandidate) * candidateCount);
	candidateCount = 0;

	foreach(lc, topCandidates)
	{
		HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc);
		uint32 id = (uint32)(uintptr_t) sc->element.ptr;

		/* Prevent self-loops */
		if (id == newId)
			continue;

		candidates[candidateCount].internalId = id;
		candidates[candidateCount].distance = sc->distance;
		candidateCount++;
	}

	neighborCount = HnswSlabGetNeighborsByHeuristic2(handle, candidates, candidateCount, maxM, support);
	if (neighborCount <= 0)
		return newId;

	nextClosest = candidates[neighborCount - 1].internalId;

	/*
	 * Set new node's neighbors.
	 */
	if (isUpdate)
	{
		newLock = HnswSlabGetNodeLock(handle, newId);
		if (newLock != NULL)
			LWLockAcquire(newLock, LW_EXCLUSIVE);
	}
	adj = HnswSlabGetAdjAtLevel(handle, newId, level);
	if (adj != NULL)
	{
		if (!isUpdate && HnswSlabGetAdjCount(adj) != 0)
			elog(ERROR, "HNSW slab insert: new element should have blank link list");

		for (i = 0; i < neighborCount; i++)
			adj[i + 1] = candidates[i].internalId;
		/* Keep padding slot deterministic for safe look-ahead prefetch */
		adj[neighborCount + 1] = 0;
		pg_write_barrier();
		adj[0] = neighborCount;
	}
	if (isUpdate && newLock != NULL)
		LWLockRelease(newLock);

	/* Update each neighbor's list to include new node */
	for (i = 0; i < neighborCount; i++)
	{
		uint32 neighborId = candidates[i].internalId;
		LWLock *neighborLock;
		uint32 *neighborAdj;
		int Mcurmax = level ? handle->m : handle->maxM0;
		uint32 curCount;
		bool is_cur_c_present = false;
		int j;

		neighborLock = HnswSlabGetNodeLock(handle, neighborId);
		if (neighborLock != NULL)
			LWLockAcquire(neighborLock, LW_EXCLUSIVE);

		neighborAdj = HnswSlabGetAdjAtLevel(handle, neighborId, level);
		if (neighborAdj == NULL)
		{
			if (neighborLock != NULL)
				LWLockRelease(neighborLock);
			continue;
		}

		curCount = HnswSlabClampNeighborCount(handle, level, HnswSlabGetAdjCount(neighborAdj));
		if (isUpdate)
		{
			for (j = 0; j < (int)curCount; j++)
			{
				if (neighborAdj[j + 1] == newId)
				{
					is_cur_c_present = true;
					break;
				}
			}
		}

		if (!is_cur_c_present)
		{
			if ((int)curCount < Mcurmax)
			{
				neighborAdj[curCount + 1] = newId;
				neighborAdj[curCount + 2] = 0;
				pg_write_barrier();
				neighborAdj[0] = curCount + 1;
			}
			else
			{
				HnswSlabHeuristicCandidate pruned[HNSW_MAX_M * 2 + 1];
				int pruneCount = 0;
				int newCount;

				pruned[pruneCount].internalId = newId;
				pruned[pruneCount].distance = HnswComputeDistance(
					PointerGetDatum(HnswSlabGetVector(handle, newId)),
					PointerGetDatum(HnswSlabGetVector(handle, neighborId)),
					support);
				pruneCount++;

				for (j = 0; j < (int)curCount && pruneCount < lengthof(pruned); j++)
				{
					uint32 existingId = neighborAdj[j + 1];
					pruned[pruneCount].internalId = existingId;
					pruned[pruneCount].distance = HnswComputeDistance(
						PointerGetDatum(HnswSlabGetVector(handle, existingId)),
						PointerGetDatum(HnswSlabGetVector(handle, neighborId)),
						support);
					pruneCount++;
				}

				newCount = HnswSlabGetNeighborsByHeuristic2(handle, pruned, pruneCount, Mcurmax, support);
				for (j = 0; j < newCount; j++)
					neighborAdj[j + 1] = pruned[j].internalId;
				neighborAdj[newCount + 1] = 0;
				pg_write_barrier();
				neighborAdj[0] = (uint32) newCount;
			}
		}

		if (neighborLock != NULL)
			LWLockRelease(neighborLock);
	}

	return nextClosest;
}

/*
 * Add a new point to the slab dynamically.
 */
uint32
HnswSlabAddPoint(HnswSlabHandle *handle,
				 Pointer vectorData, Size vectorSize,
				 ItemPointerData *heaptids, uint8 heaptidsLength,
				 Datum queryValue, HnswSupport *support,
				 int efConstruction)
{
	uint32 internalId;
	uint32 curMax;
	int curlevel;
	uint32 curEpId;
	int curEpLevel;
	int newLevel;
	char *dest;
	HnswSlabNodeMeta *meta;
	HnswSlabHeader *header = handle->header;
	int lc;
	uint32 epId;
	HnswSlabVisited *visited;
	int i;
	uint32 *adj;
	LWLock *newLock;
	bool epDeleted;
	uint64 label;
	uint32 existingId;
	bool labelExists;

	/* Encode primary heaptid as label */
	label = (heaptidsLength > 0) ? HnswSlabEncodeHeaptid(&heaptids[0]) : 0;

	/*
	 * Check if label already exists
	 * "Checking if the element with the same label already exists"
	 */
	if (label != 0)
	{
		labelExists = HnswSlabLabelLookup(handle, label, &existingId);
		if (labelExists && existingId < HnswSlabGetElementCount(handle))
		{
			/*
			 * Element with same label exists - call updatePoint instead.
			 */
			if (HnswSlabIsDeleted(handle, existingId))
			{
				/* If marked deleted, unmark and update */
				HnswSlabUnmarkDeleted(handle, existingId);
			}
			HnswSlabUpdatePoint(handle, existingId, vectorData, vectorSize,
								queryValue, support, efConstruction);
			return existingId;
		}
	}

	/*
	 * reuse a previously deleted internalId and update it in-place via updatePoint.
	 */
	while (HnswSlabPopDeletedId(handle, &internalId))
	{
		LWLock *reuseLock;

		reuseLock = HnswSlabGetNodeLock(handle, internalId);
		if (reuseLock != NULL)
			LWLockAcquire(reuseLock, LW_EXCLUSIVE);

		meta = HnswSlabGetNodeMeta(handle, internalId);
		if (meta->deleted == 0)
		{
			if (reuseLock != NULL)
				LWLockRelease(reuseLock);
			continue;
		}

		/* Remove old primary label (if any) */
		if (meta->heaptidsLength > 0)
			HnswSlabLabelRemove(handle, HnswSlabEncodeHeaptid(&meta->heaptids[0]));

		/* Update heaptids array */
		for (i = 0; i < HNSW_HEAPTIDS && i < heaptidsLength; i++)
			meta->heaptids[i] = heaptids[i];
		for (i = heaptidsLength; i < HNSW_HEAPTIDS; i++)
			ItemPointerSetInvalid(&meta->heaptids[i]);
		pg_write_barrier();
		meta->heaptidsLength = heaptidsLength;

		/* Insert new primary label */
		if (label != 0)
			HnswSlabLabelInsert(handle, label, internalId);

		/* Unmark deleted and decrement deleted count */
		meta->deleted = 0;

		if (reuseLock != NULL)
			LWLockRelease(reuseLock);

		pg_atomic_fetch_sub_u32(&header->numDeleted, 1);

		HnswSlabUpdatePoint(handle, internalId, vectorData, vectorSize,
							queryValue, support, efConstruction);

		return internalId;
	}

	/* Atomically allocate new slot (do not exceed fixed slab capacity) */
	curMax = pg_atomic_read_u32(&header->curElementCount);
	for (;;)
	{
		uint32 desired;

		if (curMax >= header->maxElements)
			elog(ERROR, "HNSW slab is full (cur=%u, max=%u)",
				 curMax, header->maxElements);

		desired = curMax + 1;
		if (pg_atomic_compare_exchange_u32(&header->curElementCount, &curMax, desired))
			break;
	}
	internalId = curMax;

	/* Generate random level */
	newLevel = HnswSlabGetRandomLevel(handle);

	/* Copy vector data to pool */
	dest = HnswSlabGetVector(handle, internalId);
	if (vectorSize > handle->vectorStride)
		elog(ERROR, "HNSW slab insert: vector size %zu exceeds stride %zu", vectorSize, handle->vectorStride);
	memcpy(dest, vectorData, vectorSize);
	if (vectorSize < handle->vectorStride)
		MemSet(dest + vectorSize, 0, handle->vectorStride - vectorSize);

	/* Hold the new element lock for the duration of insertion */
	newLock = HnswSlabGetNodeLock(handle, internalId);
	if (newLock != NULL)
		LWLockAcquire(newLock, LW_EXCLUSIVE);

	/* Initialize node metadata */
	meta = HnswSlabGetNodeMeta(handle, internalId);
	meta->level = newLevel;
	meta->deleted = 0;
	meta->version = 0;
	
	/* Allocate compact upper adjacency offset for nodes with level > 0 */
	if (newLevel > 0)
	{
		uint64 oldUsed;
		uint64 needed = (uint64)newLevel;
		
		/* Atomically allocate space in upper adjacency pool */
		oldUsed = pg_atomic_fetch_add_u64(&header->usedUpperChunks, needed);
		if (oldUsed + needed > header->totalUpperChunks)
		{
			/* Rollback allocation - pool is full */
			pg_atomic_fetch_sub_u64(&header->usedUpperChunks, needed);
			elog(WARNING, "HNSW slab: upper adjacency pool full (used=%lu, total=%lu, needed=%lu)",
				 (unsigned long)oldUsed, (unsigned long)header->totalUpperChunks, (unsigned long)needed);
			meta->adjUpperOffset = 0;
			meta->level = 0;  /* Demote to level 0 */
			newLevel = 0;
		}
		else
		{
			meta->adjUpperOffset = (uint32)oldUsed;
		}
	}
	else
	{
		meta->adjUpperOffset = 0;
	}
	
	for (i = 0; i < HNSW_HEAPTIDS && i < heaptidsLength; i++)
		meta->heaptids[i] = heaptids[i];
	for (i = heaptidsLength; i < HNSW_HEAPTIDS; i++)
		ItemPointerSetInvalid(&meta->heaptids[i]);
	pg_write_barrier();
	meta->heaptidsLength = heaptidsLength;

	/* Initialize adjacency lists to empty */
	for (lc = 0; lc <= newLevel; lc++)
	{
		adj = HnswSlabGetAdjAtLevel(handle, internalId, lc);
		if (adj != NULL)
		{
			adj[0] = 0;
			adj[1] = 0;
		}
	}

	/* Insert into label lookup*/
	if (label != 0)
		HnswSlabLabelInsert(handle, label, internalId);

	/* Resize visited array if needed */
	visited = handle->visited;
	if (visited->capacity <= internalId)
		HnswSlabVisitedResize(visited, (internalId + 1) * 2);

	/* Get current entry point (atomic snapshot) */
	HnswSlabRefreshEntryPoint(handle);
	curEpId = handle->entryPointId;
	curEpLevel = handle->entryPointLevel;

	/* Handle first element or empty index */
	curMax = pg_atomic_read_u32(&header->curElementCount);
	if (curMax == 1 || curEpLevel < 0)
	{
		/* First element - set as entry point */
		SpinLockAcquire(&header->globalLock);
		header->entryPointId = internalId;
		header->entryPointLevel = newLevel;
		pg_atomic_write_u32(&header->entryPointIdAtomic, internalId);
		pg_atomic_write_u32(&header->entryPointLevelAtomic, (uint32) newLevel);
		SpinLockRelease(&header->globalLock);

		/* Update handle cache */
		handle->entryPointId = internalId;
		handle->entryPointLevel = newLevel;

		if (newLock != NULL)
			LWLockRelease(newLock);

		return internalId;
	}

	curlevel = curEpLevel;
	epId = curEpId;

	epDeleted = (HnswSlabGetNodeMeta(handle, epId)->deleted != 0);

	/* Greedy search through upper layers above newLevel */
	if (newLevel < curlevel)
	{
		double curdist = HnswSlabComputeDistance(handle, queryValue, epId, support);

		for (lc = curlevel; lc > newLevel; lc--)
		{
			bool changed = true;

			while (changed)
			{
				uint32 *curAdj;
				uint32 size;
				uint32 j;
				LWLock *lock;

				changed = false;

				lock = HnswSlabGetNodeLock(handle, epId);
				if (lock != NULL)
					LWLockAcquire(lock, LW_SHARED);

				curAdj = HnswSlabGetAdjAtLevel(handle, epId, lc);
				if (curAdj == NULL)
				{
					if (lock != NULL)
						LWLockRelease(lock);
					break;
				}

				size = HnswSlabClampNeighborCount(handle, lc, HnswSlabGetAdjCount(curAdj));
				for (j = 0; j < size; j++)
				{
					uint32 cand = HnswSlabGetAdjNeighbor(curAdj, j);
					HnswSlabNodeMeta *candMeta;
					double d;
					
					#if HNSW_SLAB_PREFETCH_DISTANCE > 0
					if (j + HNSW_SLAB_PREFETCH_DISTANCE < size)
					{
						uint32 next_cand = HnswSlabGetAdjNeighbor(curAdj, (int) (j + HNSW_SLAB_PREFETCH_DISTANCE));
						if (next_cand < handle->maxElements)
						{
							HNSW_SLAB_PREFETCH_R(HnswSlabGetNodeMeta(handle, next_cand));
							HNSW_SLAB_PREFETCH_R(HnswSlabGetVector(handle, next_cand));
						}
					}
#endif

					if (!HnswSlabIsValidInternalId(handle, cand))
						continue;

					candMeta = HnswSlabGetNodeMeta(handle, cand);
					if ((int) candMeta->level < lc)
						continue;

					d = HnswSlabComputeDistance(handle, queryValue, cand, support);
					if (d < curdist)
					{
						curdist = d;
						epId = cand;
						changed = true;
					}
				}

				if (lock != NULL)
					LWLockRelease(lock);
			}
		}
	}
	/* Connect at each layer from min(curlevel, newLevel) down to 0 */
	for (lc = Min(curlevel, newLevel); lc >= 0; lc--)
	{
		List *topCandidates;

		topCandidates = HnswSlabSearchBaseLayerForInsert(handle, visited, queryValue, support,
														 epId, lc, efConstruction);

		/* If entry point is deleted, push it to candidates (bounded by efConstruction) */
		if (epDeleted)
		{
			HnswSearchCandidate *epCand = palloc(sizeof(HnswSearchCandidate));
			epCand->distance = HnswSlabComputeDistance(handle, queryValue, curEpId, support);
			epCand->element.ptr = (HnswElementData *)(uintptr_t) curEpId;
			topCandidates = lappend(topCandidates, epCand);

			if (list_length(topCandidates) > efConstruction)
			{
				/* Keep efConstruction closest candidates by distance */
				int n = list_length(topCandidates);
				HnswSlabHeuristicCandidate tmp[HNSW_MAX_EF_CONSTRUCTION + 1];
				int idx = 0;
				ListCell *lc2;

				/* Fallback: if efConstruction is very large, skip pruning here */
				if (n <= (int) lengthof(tmp))
				{
					foreach(lc2, topCandidates)
					{
						HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc2);
						tmp[idx].distance = sc->distance;
						tmp[idx].internalId = (uint32)(uintptr_t) sc->element.ptr;
						idx++;
					}
					qsort(tmp, idx, sizeof(tmp[0]), SlabCompareHeuristicCandidateAsc);
					list_free_deep(topCandidates);
					topCandidates = NIL;
					for (idx = 0; idx < efConstruction; idx++)
					{
						HnswSearchCandidate *sc = palloc(sizeof(HnswSearchCandidate));
						sc->distance = tmp[idx].distance;
						sc->element.ptr = (HnswElementData *)(uintptr_t) tmp[idx].internalId;
						topCandidates = lappend(topCandidates, sc);
					}
				}
			}
		}

		if (list_length(topCandidates) > 0)
			epId = HnswSlabMutuallyConnectNewElement(handle, internalId, topCandidates, lc, support, false);

		list_free_deep(topCandidates);
	}

	/* Update entry point if new element has higher level */
	if (newLevel > curlevel)
	{
		SpinLockAcquire(&header->globalLock);
		/* Double-check under lock */
		if (newLevel > (int)pg_atomic_read_u32(&header->entryPointLevelAtomic))
		{
			header->entryPointId = internalId;
			header->entryPointLevel = newLevel;
			pg_atomic_write_u32(&header->entryPointIdAtomic, internalId);
			pg_atomic_write_u32(&header->entryPointLevelAtomic, (uint32) newLevel);
		}
		SpinLockRelease(&header->globalLock);

		/* Update handle cache */
		handle->entryPointId = internalId;
		handle->entryPointLevel = newLevel;
	}

	if (newLock != NULL)
		LWLockRelease(newLock);

	return internalId;
}

static inline void
SlabUniqueAddU32(uint32 *arr, int *count, int max, uint32 v)
{
	int i;

	for (i = 0; i < *count; i++)
	{
		if (arr[i] == v)
			return;
	}
	if (*count < max)
		arr[(*count)++] = v;
}

static inline void
SlabMaxHeapSiftUp(HnswSlabHeuristicCandidate *heap, int idx)
{
	while (idx > 0)
	{
		int parent = (idx - 1) / 2;

		if (heap[parent].distance >= heap[idx].distance)
			break;

		{
			HnswSlabHeuristicCandidate tmp = heap[parent];
			heap[parent] = heap[idx];
			heap[idx] = tmp;
		}
		idx = parent;
	}
}

static inline void
SlabMaxHeapSiftDown(HnswSlabHeuristicCandidate *heap, int size, int idx)
{
	for (;;)
	{
		int left = idx * 2 + 1;
		int right = left + 1;
		int largest = idx;

		if (left < size && heap[left].distance > heap[largest].distance)
			largest = left;
		if (right < size && heap[right].distance > heap[largest].distance)
			largest = right;

		if (largest == idx)
			break;

		{
			HnswSlabHeuristicCandidate tmp = heap[largest];
			heap[largest] = heap[idx];
			heap[idx] = tmp;
		}
		idx = largest;
	}
}

static inline void
SlabMaxHeapPush(HnswSlabHeuristicCandidate *heap, int *size, HnswSlabHeuristicCandidate cand)
{
	int idx = (*size)++;

	heap[idx] = cand;
	SlabMaxHeapSiftUp(heap, idx);
}

static inline void
SlabMaxHeapReplaceTop(HnswSlabHeuristicCandidate *heap, int size, HnswSlabHeuristicCandidate cand)
{
	heap[0] = cand;
	SlabMaxHeapSiftDown(heap, size, 0);
}

static int
SlabCompareSearchCandidateDistanceAsc(const void *a, const void *b)
{
	const HnswSearchCandidate *const *pa = (const HnswSearchCandidate *const *) a;
	const HnswSearchCandidate *const *pb = (const HnswSearchCandidate *const *) b;
	const HnswSearchCandidate *ca = *pa;
	const HnswSearchCandidate *cb = *pb;

	if (ca->distance < cb->distance)
		return -1;
	if (ca->distance > cb->distance)
		return 1;
	return 0;
}

static int
HnswSlabGetConnectionsWithLock(HnswSlabHandle *handle, uint32 internalId, int level,
							   uint32 *out, int outMax)
{
	LWLock *lock;
	uint32 *adj;
	uint32 size;
	int i;
	int outCount = 0;

	if (internalId >= HnswSlabGetElementCount(handle))
		return 0;

	lock = HnswSlabGetNodeLock(handle, internalId);
	if (lock != NULL)
		LWLockAcquire(lock, LW_EXCLUSIVE);

	adj = HnswSlabGetAdjAtLevel(handle, internalId, level);
	if (adj != NULL)
	{
		size = HnswSlabGetAdjCount(adj);
		if ((int) size > outMax)
			size = (uint32) outMax;
		for (i = 0; i < (int) size; i++)
			out[outCount++] = HnswSlabGetAdjNeighbor(adj, i);
	}

	if (lock != NULL)
		LWLockRelease(lock);

	return outCount;
}

static void
HnswSlabRepairConnectionsForUpdate(HnswSlabHandle *handle,
								   Datum queryValue,
								   HnswSupport *support,
								   uint32 entryPointInternalId,
								   uint32 dataPointInternalId,
								   int dataPointLevel,
								   int maxLevel,
								   int efConstruction)
{
	uint32 currObj = entryPointInternalId;
	double curdist;
	int level;
	HnswSlabVisited *visited = handle->visited;

	if (dataPointInternalId >= HnswSlabGetElementCount(handle))
		return;

	if (currObj >= HnswSlabGetElementCount(handle))
		return;

	if (dataPointLevel < maxLevel)
	{
		curdist = HnswSlabComputeDistance(handle, queryValue, currObj, support);
		for (level = maxLevel; level > dataPointLevel; level--)
		{
			bool changed = true;

			while (changed)
			{
				LWLock *lock;
				uint32 *adj;
				uint32 size;
				uint32 *datal;
				uint32 i;

				changed = false;

				lock = HnswSlabGetNodeLock(handle, currObj);
				if (lock != NULL)
					LWLockAcquire(lock, LW_EXCLUSIVE);

				adj = HnswSlabGetAdjAtLevel(handle, currObj, level);
				if (adj == NULL)
				{
					if (lock != NULL)
						LWLockRelease(lock);
					break;
				}

				size = HnswSlabClampNeighborCount(handle, level, HnswSlabGetAdjCount(adj));
				datal = adj + 1;

#ifdef USE_SSE
				if (size > 0 && HnswSlabIsValidInternalId(handle, datal[0]))
					_mm_prefetch(HnswSlabGetVector(handle, datal[0]), _MM_HINT_T0);
#endif

				for (i = 0; i < size; i++)
				{
					uint32 cand = datal[i];
					double d;

#ifdef USE_SSE
					if (i + 1 < size && HnswSlabIsValidInternalId(handle, datal[i + 1]))
						_mm_prefetch(HnswSlabGetVector(handle, datal[i + 1]), _MM_HINT_T0);
#endif
					if (!HnswSlabIsValidInternalId(handle, cand))
						continue;

					d = HnswSlabComputeDistance(handle, queryValue, cand, support);
					if (d < curdist)
					{
						curdist = d;
						currObj = cand;
						changed = true;
					}
				}

				if (lock != NULL)
					LWLockRelease(lock);
			}
		}
	}

	if (dataPointLevel > maxLevel)
		elog(ERROR, "Level of item to be updated cannot be bigger than max level");

	for (level = dataPointLevel; level >= 0; level--)
	{
		List *topCandidates;
		ListCell *lc;
		HnswSearchCandidate **candArray;
		int candCount;
		int i;
		bool epDeleted;

		topCandidates = HnswSlabSearchBaseLayerForInsert(handle, visited, queryValue, support,
														 currObj, level, efConstruction);
		candArray = palloc(sizeof(HnswSearchCandidate *) * (list_length(topCandidates) + 1));
		candCount = 0;

		foreach(lc, topCandidates)
		{
			HnswSearchCandidate *sc = (HnswSearchCandidate *) lfirst(lc);
			uint32 id = (uint32) (uintptr_t) sc->element.ptr;

			/* Prevent self loops */
			if (id == dataPointInternalId)
			{
				pfree(sc);
				continue;
			}
			candArray[candCount++] = sc;
		}
		list_free(topCandidates);

		/*
		 * If entry point is deleted, push it to candidates (bounded by efConstruction),
		 */
		epDeleted = HnswSlabIsDeleted(handle, entryPointInternalId);
		if (epDeleted)
		{
			HnswSearchCandidate *epCand = palloc(sizeof(HnswSearchCandidate));

			epCand->distance = HnswSlabComputeDistance(handle, queryValue, entryPointInternalId, support);
			epCand->element.ptr = (HnswElementData *) (uintptr_t) entryPointInternalId;
			candArray[candCount++] = epCand;
		}

		if (candCount > 0)
		{
			/* Keep efConstruction closest candidates (smallest distances) */
			qsort(candArray, candCount, sizeof(HnswSearchCandidate *), SlabCompareSearchCandidateDistanceAsc);

			for (i = efConstruction; i < candCount; i++)
				pfree(candArray[i]);
			if (candCount > efConstruction)
				candCount = efConstruction;

			if (candCount > 0)
			{
				List *filtered = NIL;

				for (i = 0; i < candCount; i++)
					filtered = lappend(filtered, candArray[i]);

				currObj = HnswSlabMutuallyConnectNewElement(handle, dataPointInternalId, filtered, level, support, true);
				list_free_deep(filtered);
			}
		}

		pfree(candArray);
	}
}

/*
 * Update an existing point -
 */
void
HnswSlabUpdatePoint(HnswSlabHandle *handle,
					uint32 internalId,
					Pointer vectorData, Size vectorSize,
					Datum queryValue, HnswSupport *support,
					int efConstruction)
{
	char *dest;
	HnswSlabNodeMeta *meta;
	int maxLevelCopy;
	uint32 entryPointCopy;
	HnswSlabHeader *header = handle->header;
	int elemLevel;
	int layer;
	const float updateNeighborProbability = 1.0f;

	if (internalId >= HnswSlabGetElementCount(handle))
	{
		elog(WARNING, "HnswSlabUpdatePoint: invalid internalId %u", internalId);
		return;
	}

	/* update the feature vector associated with existing point with new vector */
	dest = HnswSlabGetVector(handle, internalId);
	if (vectorSize > handle->vectorStride)
		elog(ERROR, "HNSW slab update: vector size %zu exceeds stride %zu", vectorSize, handle->vectorStride);
	memcpy(dest, vectorData, vectorSize);
	if (vectorSize < handle->vectorStride)
		MemSet(dest + vectorSize, 0, handle->vectorStride - vectorSize);

	/* Snapshot entry point and max level (entry point level) */
	HnswSlabRefreshEntryPoint(handle);
	maxLevelCopy = handle->entryPointLevel;
	entryPointCopy = handle->entryPointId;

	/* If point to be updated is entry point and graph just contains single element then just return. */
	if (entryPointCopy == internalId && pg_atomic_read_u32(&header->curElementCount) == 1)
		return;

	meta = HnswSlabGetNodeMeta(handle, internalId);
	elemLevel = (int) meta->level;

	for (layer = 0; layer <= elemLevel; layer++)
	{
		uint32 listOneHop[HNSW_MAX_M * 2];
		int listOneHopCount;
		uint32 *sCand;
		uint32 *sNeigh;
		int sCandCount = 0;
		int sNeighCount = 0;
		int maxNeighbors = (layer == 0) ? handle->maxM0 : handle->m;
		int maxCand = 1 + maxNeighbors + maxNeighbors * maxNeighbors;
		int idx;

		listOneHopCount = HnswSlabGetConnectionsWithLock(handle, internalId, layer,
														 listOneHop, lengthof(listOneHop));
		if (listOneHopCount == 0)
			continue;

		sCand = palloc(sizeof(uint32) * maxCand);
		sNeigh = palloc(sizeof(uint32) * maxNeighbors);

		SlabUniqueAddU32(sCand, &sCandCount, maxCand, internalId);

		for (idx = 0; idx < listOneHopCount; idx++)
		{
			uint32 elOneHop = listOneHop[idx];
			uint32 listTwoHop[HNSW_MAX_M * 2];
			int listTwoHopCount;
			int t;

			SlabUniqueAddU32(sCand, &sCandCount, maxCand, elOneHop);

			if ((float) RandomDouble() > updateNeighborProbability)
				continue;

			SlabUniqueAddU32(sNeigh, &sNeighCount, maxNeighbors, elOneHop);

			listTwoHopCount = HnswSlabGetConnectionsWithLock(handle, elOneHop, layer,
															 listTwoHop, lengthof(listTwoHop));
			for (t = 0; t < listTwoHopCount; t++)
				SlabUniqueAddU32(sCand, &sCandCount, maxCand, listTwoHop[t]);
		}

		for (idx = 0; idx < sNeighCount; idx++)
		{
			uint32 neigh = sNeigh[idx];
			int keepSize;
			int elementsToKeep;
			HnswSlabHeuristicCandidate *heap;
			int heapSize = 0;
			int c;
			uint32 *neighAdj;
			LWLock *neighLock;
			int pruneM = (layer == 0) ? handle->maxM0 : handle->m;

			/* size = |sCand \ {neigh}| */
			keepSize = sCandCount;
			for (c = 0; c < sCandCount; c++)
			{
				if (sCand[c] == neigh)
				{
					keepSize--;
					break;
				}
			}
			if (keepSize <= 0)
				continue;

			elementsToKeep = Min(efConstruction, keepSize);
			heap = palloc(sizeof(HnswSlabHeuristicCandidate) * elementsToKeep);

			for (c = 0; c < sCandCount; c++)
			{
				uint32 candId = sCand[c];
				double distance;
				HnswSlabHeuristicCandidate cand;

				if (candId == neigh)
					continue;

				distance = HnswComputeDistance(
					PointerGetDatum(HnswSlabGetVector(handle, neigh)),
					PointerGetDatum(HnswSlabGetVector(handle, candId)),
					support);

				cand.internalId = candId;
				cand.distance = distance;

				if (heapSize < elementsToKeep)
				{
					SlabMaxHeapPush(heap, &heapSize, cand);
				}
				else if (distance < heap[0].distance)
				{
					SlabMaxHeapReplaceTop(heap, heapSize, cand);
				}
			}

			/* Retrieve neighbours using heuristic and set connections. */
			heapSize = HnswSlabGetNeighborsByHeuristic2(handle, heap, heapSize, pruneM, support);

			neighLock = HnswSlabGetNodeLock(handle, neigh);
			if (neighLock != NULL)
				LWLockAcquire(neighLock, LW_EXCLUSIVE);

			neighAdj = HnswSlabGetAdjAtLevel(handle, neigh, layer);
			if (neighAdj != NULL)
			{
				for (c = 0; c < heapSize; c++)
					neighAdj[c + 1] = heap[c].internalId;
				neighAdj[heapSize + 1] = 0;
				pg_write_barrier();
				neighAdj[0] = (uint32) heapSize;
			}

			if (neighLock != NULL)
				LWLockRelease(neighLock);

			pfree(heap);
		}

		pfree(sCand);
		pfree(sNeigh);
	}

	HnswSlabRepairConnectionsForUpdate(handle, queryValue, support,
									   entryPointCopy, internalId,
									   elemLevel, maxLevelCopy,
									   efConstruction);
}

/*
 * Mark a node as deleted
 */
void
HnswSlabMarkDeleted(HnswSlabHandle *handle, uint32 internalId)
{
	HnswSlabNodeMeta *meta;
	LWLock *lock;

	if (internalId >= HnswSlabGetElementCount(handle))
		return;

	lock = HnswSlabGetNodeLock(handle, internalId);
	if (lock != NULL)
		LWLockAcquire(lock, LW_EXCLUSIVE);

	meta = HnswSlabGetNodeMeta(handle, internalId);

	/* Check if already deleted */
	if (meta->deleted == 0)
	{
		meta->deleted = 1;

		if (lock != NULL)
			LWLockRelease(lock);

		/* Atomically increment deleted count */
		pg_atomic_fetch_add_u32(&handle->header->numDeleted, 1);

		/* push to deleted pool */
		HnswSlabPushDeletedId(handle, internalId);
	}
	else
	{
		if (lock != NULL)
			LWLockRelease(lock);
	}
}

/*
 * Unmark a node as deleted -
 */
void
HnswSlabUnmarkDeleted(HnswSlabHandle *handle, uint32 internalId)
{
	HnswSlabNodeMeta *meta;
	LWLock *lock;

	if (internalId >= HnswSlabGetElementCount(handle))
		return;

	lock = HnswSlabGetNodeLock(handle, internalId);
	if (lock != NULL)
		LWLockAcquire(lock, LW_EXCLUSIVE);

	meta = HnswSlabGetNodeMeta(handle, internalId);

	/* Check if actually deleted */
	if (meta->deleted != 0)
	{
		meta->deleted = 0;

		if (lock != NULL)
			LWLockRelease(lock);

		/* Atomically decrement deleted count */
		pg_atomic_fetch_sub_u32(&handle->header->numDeleted, 1);

		/* unmarkDeletedInternal also removes from deleted_elements set.
		 * Our ring buffer doesn't support efficient removal, but this is fine since
		 * HnswSlabPopDeletedId will skip already-undeleted nodes. */
	}
	else
	{
		if (lock != NULL)
			LWLockRelease(lock);
	}
}

/*
 * Check if a node is deleted.
 */
bool
HnswSlabIsDeleted(HnswSlabHandle *handle, uint32 internalId)
{
	HnswSlabNodeMeta *meta;

	if (internalId >= HnswSlabGetElementCount(handle))
		return true;

	meta = HnswSlabGetNodeMeta(handle, internalId);
	return (meta->deleted != 0);
}

/*
 * Label lookup -
 * Uses open addressing with linear probing.
 * Returns true if found, false otherwise.
 */
bool
HnswSlabLabelLookup(HnswSlabHandle *handle, uint64 label, uint32 *internalId)
{
	HnswSlabLabelEntry *table;
	uint32 capacity;
	uint32 idx;
	uint32 startIdx;

	if (handle->labelLookup == NULL || label == HNSW_SLAB_LABEL_EMPTY || label == HNSW_SLAB_LABEL_TOMBSTONE)
		return false;

	table = handle->labelLookup;
	capacity = handle->labelLookupCapacity;
	idx = (uint32)(label % capacity);
	startIdx = idx;

	/* Linear probing */
	SpinLockAcquire(&handle->header->labelLookupLock);
	do
	{
		if (table[idx].label == HNSW_SLAB_LABEL_EMPTY)
		{
			/* Empty slot - not found */
			SpinLockRelease(&handle->header->labelLookupLock);
			return false;
		}
		if (table[idx].label == label)
		{
			*internalId = table[idx].internalId;
			SpinLockRelease(&handle->header->labelLookupLock);
			return true;
		}
		idx = (idx + 1) % capacity;
	} while (idx != startIdx);

	SpinLockRelease(&handle->header->labelLookupLock);
	return false;
}

/*
 * Label insert -
 */
void
HnswSlabLabelInsert(HnswSlabHandle *handle, uint64 label, uint32 internalId)
{
	HnswSlabLabelEntry *table;
	uint32 capacity;
	uint32 idx;
	uint32 startIdx;

	if (handle->labelLookup == NULL || label == HNSW_SLAB_LABEL_EMPTY || label == HNSW_SLAB_LABEL_TOMBSTONE)
		return;

	table = handle->labelLookup;
	capacity = handle->labelLookupCapacity;
	idx = (uint32)(label % capacity);
	startIdx = idx;

	SpinLockAcquire(&handle->header->labelLookupLock);
	do
	{
		if (table[idx].label == HNSW_SLAB_LABEL_EMPTY ||
			table[idx].label == HNSW_SLAB_LABEL_TOMBSTONE ||
			table[idx].label == label)
		{
			/* Empty slot or update existing */
			table[idx].label = label;
			table[idx].internalId = internalId;
			SpinLockRelease(&handle->header->labelLookupLock);
			return;
		}
		idx = (idx + 1) % capacity;
	} while (idx != startIdx);

	SpinLockRelease(&handle->header->labelLookupLock);
	/* Hash table full - should not happen with 2x capacity */
	elog(WARNING, "HNSW slab label lookup table full");
}

/*
 * Label remove -
 * Uses tombstone deletion (set label to special value).
 */
void
HnswSlabLabelRemove(HnswSlabHandle *handle, uint64 label)
{
	HnswSlabLabelEntry *table;
	uint32 capacity;
	uint32 idx;
	uint32 startIdx;

	if (handle->labelLookup == NULL || label == HNSW_SLAB_LABEL_EMPTY || label == HNSW_SLAB_LABEL_TOMBSTONE)
		return;

	table = handle->labelLookup;
	capacity = handle->labelLookupCapacity;
	idx = (uint32)(label % capacity);
	startIdx = idx;

	SpinLockAcquire(&handle->header->labelLookupLock);
	do
	{
		if (table[idx].label == HNSW_SLAB_LABEL_EMPTY)
		{
			/* Empty slot - not found */
			SpinLockRelease(&handle->header->labelLookupLock);
			return;
		}
		if (table[idx].label == label)
		{
			/* Found - mark as deleted (tombstone) */
			table[idx].label = HNSW_SLAB_LABEL_TOMBSTONE;
			table[idx].internalId = UINT32_MAX;
			SpinLockRelease(&handle->header->labelLookupLock);
			return;
		}
		idx = (idx + 1) % capacity;
	} while (idx != startIdx);

	SpinLockRelease(&handle->header->labelLookupLock);
}

/*
 * Lookup by heaptid - convenience wrapper.
 */
uint32
HnswSlabLabelLookupByHeaptid(HnswSlabHandle *handle, ItemPointer heaptid, bool *found)
{
	uint64 label;
	uint32 internalId = 0;

	label = HnswSlabEncodeHeaptid(heaptid);
	*found = HnswSlabLabelLookup(handle, label, &internalId);
	return internalId;
}

/*
 * Pop a deleted ID from the pool -
 * Returns true if an ID was popped, false if pool is empty.
 */
bool
HnswSlabPopDeletedId(HnswSlabHandle *handle, uint32 *internalId)
{
	HnswSlabHeader *header = handle->header;
	uint32 head;
	uint32 tail;
	uint32 *pool;
	uint32 id;

	if (handle->deletedPool == NULL)
		return false;

	pool = handle->deletedPool;

	SpinLockAcquire(&header->deletedPoolLock);

	head = pg_atomic_read_u32(&header->deletedPoolHead);
	tail = pg_atomic_read_u32(&header->deletedPoolTail);

	/* Pool empty? */
	if (head == tail)
	{
		SpinLockRelease(&header->deletedPoolLock);
		return false;
	}

	/* Pop from head */
	id = pool[head];
	head = (head + 1) % header->deletedPoolCapacity;
	pg_atomic_write_u32(&header->deletedPoolHead, head);

	SpinLockRelease(&header->deletedPoolLock);

	/* Verify the ID is still deleted (could have been undeleted) */
	if (!HnswSlabIsDeleted(handle, id))
	{
		/* Try to pop another one (recursive, but bounded by pool size) */
		return HnswSlabPopDeletedId(handle, internalId);
	}

	*internalId = id;
	return true;
}

/*
 * Push a deleted ID to the pool -
 */
void
HnswSlabPushDeletedId(HnswSlabHandle *handle, uint32 internalId)
{
	HnswSlabHeader *header = handle->header;
	uint32 head;
	uint32 tail;
	uint32 newTail;
	uint32 *pool;

	if (handle->deletedPool == NULL)
		return;

	pool = handle->deletedPool;

	SpinLockAcquire(&header->deletedPoolLock);

	head = pg_atomic_read_u32(&header->deletedPoolHead);
	tail = pg_atomic_read_u32(&header->deletedPoolTail);
	newTail = (tail + 1) % header->deletedPoolCapacity;

	/* Pool full? */
	if (newTail == head)
	{
		SpinLockRelease(&header->deletedPoolLock);
		/* Pool is full - silently drop. This is acceptable as the ID
		 * will still be marked deleted and won't be returned in searches. */
		return;
	}

	/* Push to tail */
	pool[tail] = internalId;
	pg_atomic_write_u32(&header->deletedPoolTail, newTail);

	SpinLockRelease(&header->deletedPoolLock);
}

/*
 * Sync delete from vacuum - mark node as deleted in slab.
 * Called when vacuum removes a heaptid from an element.
 */
void
HnswSlabSyncDeleteFromVacuum(Oid dbid, Oid relid, ItemPointer heaptid)
{
	HnswSlabHandle *handle;
	uint32 internalId;
	bool found;

	/* Try to attach to slab - if not available, nothing to sync */
	PG_TRY();
	{
		handle = HnswSlabAttach(dbid, relid);
	}
	PG_CATCH();
	{
		/* Slab not available - ignore */
		FlushErrorState();
		return;
	}
	PG_END_TRY();

	/* Lookup internalId by heaptid */
	internalId = HnswSlabLabelLookupByHeaptid(handle, heaptid, &found);

	if (found)
	{
		/* Mark as deleted */
		HnswSlabMarkDeleted(handle, internalId);

		/* Remove from label lookup */
		HnswSlabLabelRemove(handle, HnswSlabEncodeHeaptid(heaptid));
	}

	HnswSlabDetach(handle);
}

/*
 * Sync heaptid removal from vacuum - remove heaptid from node's heaptids array.
 */
void
HnswSlabSyncRemoveHeaptidFromVacuum(Oid dbid, Oid relid, ItemPointer heaptid)
{
	HnswSlabHandle *handle;
	uint32 internalId;
	bool found;
	HnswSlabNodeMeta *meta;
	LWLock *lock;
	int i, j;
	bool lastHeaptid = true;

	/* Try to attach to slab */
	PG_TRY();
	{
		handle = HnswSlabAttach(dbid, relid);
	}
	PG_CATCH();
	{
		FlushErrorState();
		return;
	}
	PG_END_TRY();

	/* Lookup internalId by heaptid */
	internalId = HnswSlabLabelLookupByHeaptid(handle, heaptid, &found);

	if (!found)
	{
		HnswSlabDetach(handle);
		return;
	}

	lock = HnswSlabGetNodeLock(handle, internalId);
	if (lock != NULL)
		LWLockAcquire(lock, LW_EXCLUSIVE);

	meta = HnswSlabGetNodeMeta(handle, internalId);

	/* Remove the heaptid from the array and compact */
	j = 0;
	for (i = 0; i < meta->heaptidsLength; i++)
	{
		if (ItemPointerEquals(&meta->heaptids[i], heaptid))
			continue;  /* Skip this one */
		if (j != i)
			meta->heaptids[j] = meta->heaptids[i];
		j++;
	}

	/* Invalidate remaining slots */
	for (i = j; i < HNSW_HEAPTIDS; i++)
		ItemPointerSetInvalid(&meta->heaptids[i]);

	meta->heaptidsLength = j;
	lastHeaptid = (j == 0);

	if (lock != NULL)
		LWLockRelease(lock);

	/* Remove from label lookup */
	HnswSlabLabelRemove(handle, HnswSlabEncodeHeaptid(heaptid));

	/* If no heaptids left, mark as deleted */
	if (lastHeaptid)
		HnswSlabMarkDeleted(handle, internalId);

	HnswSlabDetach(handle);
}

/*
 * Sync heaptids array from vacuum in one shot.
 */
void
HnswSlabSyncUpdateHeaptidsFromVacuum(Oid dbid, Oid relid,
									 ItemPointer oldPrimaryHeaptid,
									 ItemPointerData *heaptids,
									 uint8 heaptidsLength)
{
	HnswSlabHandle *handle;
	uint32 internalId;
	bool found;
	HnswSlabNodeMeta *meta;
	LWLock *lock;
	uint64 oldLabel;
	uint64 newLabel;
	int i;

	if (oldPrimaryHeaptid == NULL || !ItemPointerIsValid(oldPrimaryHeaptid))
		return;

	oldLabel = HnswSlabEncodeHeaptid(oldPrimaryHeaptid);
	if (oldLabel == HNSW_SLAB_LABEL_EMPTY)
		return;

	/* Try to attach to slab */
	PG_TRY();
	{
		handle = HnswSlabAttach(dbid, relid);
	}
	PG_CATCH();
	{
		FlushErrorState();
		return;
	}
	PG_END_TRY();

	found = HnswSlabLabelLookup(handle, oldLabel, &internalId);
	if (!found)
	{
		HnswSlabDetach(handle);
		return;
	}

	lock = HnswSlabGetNodeLock(handle, internalId);
	if (lock != NULL)
		LWLockAcquire(lock, LW_EXCLUSIVE);

	meta = HnswSlabGetNodeMeta(handle, internalId);

	/* Update heaptids array */
	for (i = 0; i < HNSW_HEAPTIDS && i < heaptidsLength; i++)
		meta->heaptids[i] = heaptids[i];
	for (i = heaptidsLength; i < HNSW_HEAPTIDS; i++)
		ItemPointerSetInvalid(&meta->heaptids[i]);
	pg_write_barrier();
	meta->heaptidsLength = heaptidsLength;

	/* Update label mapping and delete state */
	if (heaptidsLength == 0)
	{
		/* No heaptids left -> mark deleted and remove label entry */
		if (meta->deleted == 0)
		{
			meta->deleted = 1;
			pg_atomic_fetch_add_u32(&handle->header->numDeleted, 1);
			HnswSlabPushDeletedId(handle, internalId);
		}
		HnswSlabLabelRemove(handle, oldLabel);
	}
	else
	{
		newLabel = HnswSlabEncodeHeaptid(&heaptids[0]);

		if (newLabel != HNSW_SLAB_LABEL_EMPTY && newLabel != oldLabel)
		{
			HnswSlabLabelRemove(handle, oldLabel);
			HnswSlabLabelInsert(handle, newLabel, internalId);
		}
	}

	if (lock != NULL)
		LWLockRelease(lock);

	HnswSlabDetach(handle);
}

/*
 * Scan all elements in the slab (for NULL query or full scan).
 * Returns a list of HnswSearchCandidate with distance 0.
 */
List *
HnswSlabScanAll(HnswSlabHandle *handle)
{
	List *result = NIL;
	uint32 numElements;
	uint32 i;

	if (handle == NULL || handle->header == NULL)
		return NIL;

	/* Use atomic read for current element count */
	numElements = pg_atomic_read_u32(&handle->header->curElementCount);

	for (i = 0; i < numElements; i++)
	{
		HnswSlabNodeMeta *meta;

		if ((i & 0x3FFF) == 0)
			CHECK_FOR_INTERRUPTS();
		meta = HnswSlabGetNodeMeta(handle, i);
		
		if (meta->heaptidsLength != 0)
		{
			HnswSearchCandidate *candidate = palloc(sizeof(HnswSearchCandidate));
			candidate->distance = 0.0;
			candidate->element.ptr = (HnswElementData *)(uintptr_t)i;
			result = lappend(result, candidate);
		}
	}

	return result;
}

/*
 * ========================================================================
 * Slab auto-rebuild from disk index
 * ========================================================================
 */

/* Hash table for index TID -> internal ID mapping during rebuild */
typedef struct HnswRebuildTidMapEntry
{
	uint64 key;  /* (blkno << 16) | offno */
	uint32 internalId;
} HnswRebuildTidMapEntry;

/*
 * Wait for slab to become READY, with timeout and backoff.
 * Returns true if READY, false if timeout/invalid.
 */
static bool
HnswSlabWaitForReady(Oid dbid, Oid relid, int maxWaitMs)
{
	int waitMs = 0;
	int sleepMs = 10;
	HnswSlabDirKey key;
	HnswSlabDirEntry *entry;
	bool found;
	
	key.dbid = dbid;
	key.relid = relid;
	
	while (waitMs < maxWaitMs)
	{
		CHECK_FOR_INTERRUPTS();
		
		LWLockAcquire(HnswSlabDirLock, LW_SHARED);
		entry = (HnswSlabDirEntry *) hash_search(HnswSlabDirHash, &key, HASH_FIND, &found);
		
		if (!found)
		{
			LWLockRelease(HnswSlabDirLock);
			return false;
		}
		
		if (entry->state == HNSW_SLAB_READY)
		{
			LWLockRelease(HnswSlabDirLock);
			return true;
		}
		
		if (entry->state == HNSW_SLAB_INVALID)
		{
			LWLockRelease(HnswSlabDirLock);
			return false;
		}
		
		LWLockRelease(HnswSlabDirLock);
		
		pg_usleep(sleepMs * 1000L);
		waitMs += sleepMs;
		sleepMs = Min(sleepMs * 2, 1000); /* Exponential backoff, max 1s */
	}
	
	return false;
}

/*
 * Rebuild slab from disk index.
 * Uses 3-pass algorithm to avoid loading all vectors into memory.
 */
static void
HnswSlabRebuildFromDisk(Relation index)
{
	Oid dbid = MyDatabaseId;
	Oid relid = RelationGetRelid(index);
	BlockNumber nblocks;
	BlockNumber blkno;
	BufferAccessStrategy bas;
	int m;
	int dimensions;
	bool isSq8Index;
	float sq8LowerBound;
	float sq8Diff;
	HnswElement diskEntryPoint;
	BlockNumber entryBlkno = InvalidBlockNumber;
	OffsetNumber entryOffno = InvalidOffsetNumber;
	int entryPointLevel = -1;
	
	/* Pass 0 statistics */
	uint32 numElements = 0;
	Size maxValueSize = 0;
	int maxLevelActual = -1;
	uint64 totalUpperChunks = 0;
	
	/* Build context */
	HnswSlabBuildContext *ctx = NULL;
	HTAB *tidMap = NULL;
	HASHCTL tidMapCtl;
	uint32 entryPointInternalId = 0;
	
	elog(NOTICE, "HNSW slab rebuild started for index %u.%u", dbid, relid);
	
	/* Read metadata from metapage */
	HnswGetMetaPageInfoEx(index, &m, &diskEntryPoint, &sq8LowerBound, &sq8Diff, &isSq8Index);
	
	if (diskEntryPoint != NULL)
	{
		entryBlkno = diskEntryPoint->blkno;
		entryOffno = diskEntryPoint->offno;
		entryPointLevel = diskEntryPoint->level;
	}
	
	dimensions = TupleDescAttr(index->rd_att, 0)->atttypmod;
	
	/* Get total blocks */
	nblocks = RelationGetNumberOfBlocksInFork(index, MAIN_FORKNUM);
	bas = GetAccessStrategy(BAS_BULKREAD);
	
	/* Pass 0: Statistics gathering */
	elog(NOTICE, "HNSW slab rebuild: Pass 0 - gathering statistics");
	
	for (blkno = 0; blkno < nblocks; blkno++)
	{
		Buffer buf;
		Page page;
		OffsetNumber offno;
		OffsetNumber maxoff;
		
			if ((blkno & 0xFF) == 0)
				CHECK_FOR_INTERRUPTS();
			
			buf = ReadBufferExtended(index, MAIN_FORKNUM, blkno, RBM_NORMAL, bas);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
		
		if (PageIsNew(page))
		{
			UnlockReleaseBuffer(buf);
			continue;
		}
		
		maxoff = PageGetMaxOffsetNumber(page);
		
		for (offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId itemid = PageGetItemId(page, offno);
			HnswElementTuple etup;
			Size vectorSize;
			int level;
			
			if (!ItemIdIsNormal(itemid))
				continue;
			
			etup = (HnswElementTuple) PageGetItem(page, itemid);
			
			if (etup->type != HNSW_ELEMENT_TUPLE_TYPE)
				continue;
			
			/* Count element */
			numElements++;
			
			/* Track max vector size */
			vectorSize = VARSIZE_ANY(&etup->data);
			if (vectorSize > maxValueSize)
				maxValueSize = vectorSize;
			
			/* Track max level */
			level = (int) etup->level;
			if (level > maxLevelActual)
				maxLevelActual = level;
			
			/* Count upper chunks */
			if (level > 0)
				totalUpperChunks += level;
		}
		
		UnlockReleaseBuffer(buf);
	}
	
	if (numElements == 0)
	{
		elog(NOTICE, "HNSW slab rebuild: empty index, skipping rebuild");
		return;
	}
	
	elog(NOTICE, "HNSW slab rebuild: Pass 0 complete - %u elements, maxLevel=%d, totalUpperChunks=%lu",
		 numElements, maxLevelActual, (unsigned long)totalUpperChunks);
	
	/* Begin build */
	{
		Size valueStride = MAXALIGN(maxValueSize);
		uint32 estimatedElements = numElements + 10000;
		uint64 reservedUpperChunks = totalUpperChunks + 10000;
		
		ctx = HnswSlabBuildBegin(dbid, relid, m, dimensions, valueStride,
								 estimatedElements, maxLevelActual, reservedUpperChunks,
								 isSq8Index, sq8LowerBound, sq8Diff);
	}
	
	/* Create TID mapping hash table */
	MemSet(&tidMapCtl, 0, sizeof(tidMapCtl));
	tidMapCtl.keysize = sizeof(uint64);
	tidMapCtl.entrysize = sizeof(HnswRebuildTidMapEntry);
	tidMapCtl.hcxt = CurrentMemoryContext;
	
	tidMap = hash_create("HNSW Rebuild TID Map",
						 numElements * 2,
						 &tidMapCtl,
						 HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	
	/* Pass 1: Vectors + metadata + TID mapping */
	elog(NOTICE, "HNSW slab rebuild: Pass 1 - loading vectors and metadata");
	
	for (blkno = 0; blkno < nblocks; blkno++)
	{
		Buffer buf;
		Page page;
		OffsetNumber offno;
		OffsetNumber maxoff;
		
		if ((blkno & 0xFF) == 0)
			CHECK_FOR_INTERRUPTS();
		
		buf = ReadBufferExtended(index, MAIN_FORKNUM, blkno, RBM_NORMAL, bas);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		
		if (PageIsNew(page))
		{
			UnlockReleaseBuffer(buf);
			continue;
		}
		
		maxoff = PageGetMaxOffsetNumber(page);
		
		for (offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId itemid = PageGetItemId(page, offno);
			HnswElementTuple etup;
			Pointer vectorPtr;
			Size vectorSize;
			uint32 internalId;
			ItemPointerData heaptids[HNSW_HEAPTIDS];
			uint8 heaptidsLength;
			uint64 tidKey;
			HnswRebuildTidMapEntry *mapEntry;
			bool found;
			int i;
			
			if (!ItemIdIsNormal(itemid))
				continue;
			
			etup = (HnswElementTuple) PageGetItem(page, itemid);
			
			if (etup->type != HNSW_ELEMENT_TUPLE_TYPE)
				continue;
			
			/* Add vector */
			vectorPtr = (Pointer) &etup->data;
			vectorSize = VARSIZE_ANY(&etup->data);
			internalId = HnswSlabBuildAddVector(ctx, vectorPtr, vectorSize);
			
			/* Copy heaptids */
			heaptidsLength = 0;
			for (i = 0; i < HNSW_HEAPTIDS; i++)
			{
				if (!ItemPointerIsValid(&etup->heaptids[i]))
					break;
				heaptids[i] = etup->heaptids[i];
				heaptidsLength++;
			}
			
			/* Set metadata */
			HnswSlabBuildSetNodeMeta(ctx, internalId, heaptids, heaptidsLength,
									 etup->level, etup->deleted, etup->version);
			
			/* Store TID mapping */
			tidKey = ((uint64) blkno << 16) | (uint64) offno;
			mapEntry = (HnswRebuildTidMapEntry *) hash_search(tidMap, &tidKey, HASH_ENTER, &found);
			mapEntry->internalId = internalId;
			
			/* Check if this is the entry point */
			if (BlockNumberIsValid(entryBlkno) && blkno == entryBlkno && offno == entryOffno)
			{
				entryPointInternalId = internalId;
			}
		}
		
		UnlockReleaseBuffer(buf);
	}
	
	elog(NOTICE, "HNSW slab rebuild: Pass 1 complete");
	
	/* Pass 2: Adjacency */
	elog(NOTICE, "HNSW slab rebuild: Pass 2 - loading adjacency lists");
	
	for (blkno = 0; blkno < nblocks; blkno++)
	{
		Buffer buf;
		Page page;
		OffsetNumber offno;
		OffsetNumber maxoff;
		
		if ((blkno & 0xFF) == 0)
			CHECK_FOR_INTERRUPTS();
		
		buf = ReadBufferExtended(index, MAIN_FORKNUM, blkno, RBM_NORMAL, bas);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		
		if (PageIsNew(page))
		{
			UnlockReleaseBuffer(buf);
			continue;
		}
		
		maxoff = PageGetMaxOffsetNumber(page);
		
		for (offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId itemid = PageGetItemId(page, offno);
			HnswElementTuple etup;
			uint64 selfTidKey;
			HnswRebuildTidMapEntry *selfEntry;
			bool found;
			uint32 selfInternalId;
			int level;
			BlockNumber neighborPage;
			OffsetNumber neighborOffno;
			Buffer nbuf;
			Page npage;
			HnswNeighborTuple ntup;
			ItemId nitemid;
			int lc;
			
			if (!ItemIdIsNormal(itemid))
				continue;
			
			etup = (HnswElementTuple) PageGetItem(page, itemid);
			
			if (etup->type != HNSW_ELEMENT_TUPLE_TYPE)
				continue;
			
			/* Lookup self internal ID */
			selfTidKey = ((uint64) blkno << 16) | (uint64) offno;
			selfEntry = (HnswRebuildTidMapEntry *) hash_search(tidMap, &selfTidKey, HASH_FIND, &found);
			if (!found)
				continue;
			
			selfInternalId = selfEntry->internalId;
			level = (int) etup->level;
			
			/* Read neighbor tuple */
			neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
			neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
			
			if (!BlockNumberIsValid(neighborPage))
				continue;
			
			nbuf = ReadBufferExtended(index, MAIN_FORKNUM, neighborPage, RBM_NORMAL, bas);
			LockBuffer(nbuf, BUFFER_LOCK_SHARE);
			npage = BufferGetPage(nbuf);
			
			if (PageIsNew(npage))
			{
				UnlockReleaseBuffer(nbuf);
				continue;
			}
			
			nitemid = PageGetItemId(npage, neighborOffno);
			if (!ItemIdIsNormal(nitemid))
			{
				UnlockReleaseBuffer(nbuf);
				continue;
			}
			
			ntup = (HnswNeighborTuple) PageGetItem(npage, nitemid);
			
			/* Verify neighbor tuple */
			if (ntup->type != HNSW_NEIGHBOR_TUPLE_TYPE)
			{
				UnlockReleaseBuffer(nbuf);
				continue;
			}
			
			if (ntup->version != etup->version || ntup->count != (level + 2) * m)
			{
				UnlockReleaseBuffer(nbuf);
				continue;
			}
			
			/* Process each layer */
			for (lc = 0; lc <= level; lc++)
			{
				int lm = HnswGetLayerM(m, lc);
				int start = (level - lc) * m;
				uint32 neighborIds[HNSW_MAX_M * 2];
				int neighborCount = 0;
				int i;
				
				for (i = 0; i < lm; i++)
				{
					ItemPointer indextid = &ntup->indextids[start + i];
					uint64 neighborTidKey;
					HnswRebuildTidMapEntry *neighborEntry;
					BlockNumber nblk;
					OffsetNumber noff;
					
					if (!ItemPointerIsValid(indextid))
						break;
					
					nblk = ItemPointerGetBlockNumber(indextid);
					noff = ItemPointerGetOffsetNumber(indextid);
					neighborTidKey = ((uint64) nblk << 16) | (uint64) noff;
					
					neighborEntry = (HnswRebuildTidMapEntry *) hash_search(tidMap, &neighborTidKey,
																		  HASH_FIND, &found);
					if (found)
					{
						neighborIds[neighborCount++] = neighborEntry->internalId;
					}
				}
				
				/* Set adjacency */
				HnswSlabBuildSetAdjacency(ctx, selfInternalId, lc, neighborIds, neighborCount);
			}
			
			UnlockReleaseBuffer(nbuf);
		}
		
		UnlockReleaseBuffer(buf);
	}
	
	elog(NOTICE, "HNSW slab rebuild: Pass 2 complete");
	
	/* Set entry point */
	if (BlockNumberIsValid(entryBlkno))
	{
		HnswSlabBuildSetEntryPoint(ctx, entryPointInternalId, entryPointLevel);
	}
	
	/* Finish build */
	HnswSlabBuildFinish(ctx);
	
	/* Cleanup */
	hash_destroy(tidMap);
	FreeAccessStrategy(bas);
	
	elog(NOTICE, "HNSW slab rebuild complete for index %u.%u", dbid, relid);
}

/*
 * Attach to slab or rebuild from disk if not available.
 * 
 * This is the main entry point for query/insert operations.
 * Handles concurrent rebuild attempts.
 */
HnswSlabHandle *
HnswSlabAttachOrBuildFromDisk(Relation index)
{
	Oid dbid = MyDatabaseId;
	Oid relid = RelationGetRelid(index);
	HnswSlabDirKey key;
	HnswSlabDirEntry *entry;
	bool found;
	HnswSlabHandle *handle = NULL;
	
	key.dbid = dbid;
	key.relid = relid;
	
	/* Fast path: try normal attach first */
	PG_TRY();
	{
		handle = HnswSlabAttach(dbid, relid);
		return handle;
	}
	PG_CATCH();
	{
		/* Slab not available - we'll try to rebuild */
		FlushErrorState();
	}
	PG_END_TRY();
	
	/* Slow path: slab not available, check state */
	LWLockAcquire(HnswSlabDirLock, LW_SHARED);
	entry = (HnswSlabDirEntry *) hash_search(HnswSlabDirHash, &key, HASH_FIND, &found);
	
	if (found && entry->state == HNSW_SLAB_BUILDING)
	{
		/* Another backend is building - wait for it */
		LWLockRelease(HnswSlabDirLock);
		
		elog(NOTICE, "HNSW slab is being rebuilt by another backend, waiting...");
		
		if (HnswSlabWaitForReady(dbid, relid, 30000))  /* Wait up to 30s */
		{
			/* Now try to attach */
			PG_TRY();
			{
				handle = HnswSlabAttach(dbid, relid);
				return handle;
			}
			PG_CATCH();
			{
				FlushErrorState();
				elog(ERROR, "HNSW slab rebuild completed but attach failed for index %u.%u",
					 dbid, relid);
			}
			PG_END_TRY();
		}
		else
		{
			elog(ERROR, "HNSW slab rebuild timeout or failed for index %u.%u", dbid, relid);
		}
	}
	
	LWLockRelease(HnswSlabDirLock);
	
	/* We need to rebuild - try to claim BUILDING state */
	LWLockAcquire(HnswSlabDirLock, LW_EXCLUSIVE);
	entry = (HnswSlabDirEntry *) hash_search(HnswSlabDirHash, &key, HASH_ENTER, &found);
	
	if (!found || entry->state == HNSW_SLAB_INVALID)
	{
		/* We can rebuild */
		entry->state = HNSW_SLAB_BUILDING;
		entry->handle = 0;
		LWLockRelease(HnswSlabDirLock);
		
		/* Perform rebuild */
		PG_TRY();
		{
			HnswSlabRebuildFromDisk(index);
		}
		PG_CATCH();
		{
			/* Mark as invalid on error */
			LWLockAcquire(HnswSlabDirLock, LW_EXCLUSIVE);
			entry = (HnswSlabDirEntry *) hash_search(HnswSlabDirHash, &key, HASH_FIND, &found);
			if (found)
				entry->state = HNSW_SLAB_INVALID;
			LWLockRelease(HnswSlabDirLock);
			
			PG_RE_THROW();
		}
		PG_END_TRY();
		
		/* Attach to rebuilt slab */
		handle = HnswSlabAttach(dbid, relid);
		return handle;
	}
	else if (entry->state == HNSW_SLAB_BUILDING)
	{
		/* Race: another backend just started building */
		LWLockRelease(HnswSlabDirLock);
		
		if (HnswSlabWaitForReady(dbid, relid, 30000))
		{
			handle = HnswSlabAttach(dbid, relid);
			return handle;
		}
		else
		{
			elog(ERROR, "HNSW slab rebuild timeout for index %u.%u", dbid, relid);
		}
	}
	else if (entry->state == HNSW_SLAB_READY)
	{
		/* Race: became ready while we were checking */
		LWLockRelease(HnswSlabDirLock);
		handle = HnswSlabAttach(dbid, relid);
		return handle;
	}
	
	LWLockRelease(HnswSlabDirLock);
	elog(ERROR, "HNSW slab rebuild failed for index %u.%u", dbid, relid);
	return NULL;  /* Not reached */
}
