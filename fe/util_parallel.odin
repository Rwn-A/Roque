package fe

/*
 Parallel model based on https://www.dgtlgrove.com/p/multi-core-by-default.
 Does not follow the article implementation exactly. Uses the term `rank` instead of
 the article's `lane`, as lane is reserved terminology for SIMD.

 This is a single program multiple data approach to parallelization.

 A "rank group" is the set of threads currently executing the same code together
 (created by `widen`). Every thread always has a valid rank context: the zero value
 of Rank_Ctx means "solo" (idx 0, count 1), so threads that were not created by
 `widen` (main thread, foreign threads) need no setup.

 Code that requires synchronization must be run by all ranks in the group to avoid deadlocks.
*/

import "base:intrinsics"
import "core:sync"
import "core:thread"

MAX_RANKS :: 128

@(thread_local, private)
rank_ctx: Rank_Ctx

Rank_Ctx :: struct {
	idx:         int,
	count:       int, // 0 is treated as 1, see `rank_count`.
	barrier:     ^sync.Barrier,
	shared_ptrs: ^[MAX_RANKS]rawptr,
}

//== rank accessing

rank_idx :: proc() -> int {
	return rank_ctx.idx
}

// Number of ranks, always >= 1.
rank_count :: proc() -> int {
	return max(rank_ctx.count, 1)
}

//== narrowing (temporarily single threaded execution)

// Note: this is more ceremony than the article had, but it lets code that expects a valid rank
// context run inside a narrow. In the original article, narrow was simpler but would fail for
// code that expected a valid rank context. Manual `if rank_idx() == 0 {...} rank_sync()`
// retains the article's semantics if needed.

// Nested narrows are flat: once narrowed, further narrows only bump the depth.
// This state describes the *current group level*, so `widen` saves and resets it.
@(thread_local, private)
narrow_saved_ctx: Rank_Ctx

@(thread_local, private)
narrow_depth: int

// narrow execution to a single rank for the following block.
// usage: `if narrow() {my serial code}`, automatically widens at end of scope.
@(deferred_out = narrow_end)
narrow :: proc() -> bool {
	if rank_idx() != 0 { return false }
	if narrow_depth == 0 {
		narrow_saved_ctx = rank_ctx
		rank_ctx = Rank_Ctx {
			idx   = 0,
			count = 1,
		}
	}
	narrow_depth += 1
	return true
}

@(private)
narrow_end :: proc(was_narrowed: bool) {
	if was_narrowed {
		narrow_depth -= 1
		if narrow_depth == 0 { rank_ctx = narrow_saved_ctx }
	}
	rank_sync()
}

//== solo (all ranks think they are single-threaded)


// Each rank believes its the only rank in a group, unlike narrow, where only one rank continues.
// usage: `solo() ... solo_end()`.
solo :: proc() -> Rank_Ctx {
	saved := rank_ctx
	rank_ctx = Rank_Ctx {
		idx   = 0,
		count = 1,
	}
	return saved
}

solo_end :: proc(saved: Rank_Ctx) {
	rank_ctx = saved
}

//== synchronization

// Block until all ranks have arrived
rank_sync :: proc() {
	if rank_count() > 1 { sync.barrier_wait(rank_ctx.barrier) }
}

// Copy `value` from `source_idx` into every other rank's `value`.
// Warning: by default writes the value from rank 0, the default narrowing rank.
rank_sync_value :: proc(value: ^$T, source_idx: int = 0) {
	if rank_count() <= 1 { return }

	assert(source_idx >= 0 && source_idx < rank_count())

	// everybody lets everyone know the ptr of their local copy
	rank_ctx.shared_ptrs[rank_idx()] = value
	rank_sync()

	// source writes into everyone
	if rank_idx() == source_idx {
		for i in 0 ..< rank_count() {
			if i != source_idx { (cast(^T)rank_ctx.shared_ptrs[i])^ = value^ }
		}
	}
	rank_sync()
}

// Combine all ranks' `local` with the given procedure. Every rank returns the same result.
// For simple numerical types see `rank_sum`.
rank_reduce :: proc(local: $T, $combine: proc(a, b: T) -> T) -> T {
	if rank_count() <= 1 { return local }

	mine := local
	rank_ctx.shared_ptrs[rank_idx()] = &mine
	rank_sync()

	result := (cast(^T)rank_ctx.shared_ptrs[0])^
	for i in 1 ..< rank_count() {
		result = combine(result, (cast(^T)rank_ctx.shared_ptrs[i])^)
	}

	// `mine` must outlive every rank's reads.
	rank_sync()
	return result
}

// Sum each rank's `local`. Given local must support `+` operator.
rank_sum :: proc(local: $T) -> T where intrinsics.type_is_numeric(T) {
	return rank_reduce(local, proc(a, b: T) -> T { return a + b })
}

//== static work distribution

Range :: struct {
	min, max: int,
}

// Split count roughly equally among ranks, each rank will receive its range.
rank_range :: proc(count: int) -> Range {
	// Ripped directly from our boy Ryan Fleury
	per := count / rank_count()
	leftover := count % rank_count()
	has_left := rank_idx() < leftover
	before := has_left ? rank_idx() : leftover
	first := per * rank_idx() + before
	opl := first + per + (has_left ? 1 : 0)
	return Range{first, opl}
}

// This rank's portion of s, split the same way as rank_range(len(s)).
rank_slice :: proc(s: []$T) -> []T {
	r := rank_range(len(s))
	return s[r.min:r.max]
}

//== Dynamic work distribution

// The pool must be the same object on every rank (e.g. rank 0 owns it and broadcasts the
// pointer with `rank_sync_value`, or it's created before `widen` and passed in through data).
Task_Pool :: struct {
	counter: int, // atomic
	count:   int,
}

// create a task pool, all ranks must be supplying the same pool pointer.
rank_task_pool_init :: proc(pool: ^Task_Pool, count: int) {
	rank_sync()
	if narrow() { pool.counter = 0; pool.count = count }
}

// Retrieve next task, returns false if theres no task left.
rank_task_pool_next :: proc(pool: ^Task_Pool) -> (idx: int, ok: bool) {
	idx = sync.atomic_add(&pool.counter, 1)
	ok = idx < pool.count
	return
}

/*
Example: hand out variable-cost tasks dynamically (use when tasks outnumber ranks
and cost varies; use `rank_range` when the work is uniform).
  Job :: struct {
    tasks:   []Task,
    results: []Result,
    pool:    Task_Pool,
  }
  process :: proc(job: ^Job) {
    rank_task_pool_init(&job.pool, len(job.tasks))
    for {
        i := rank_task_pool_next(&job.pool) or_break
        job.results[i] = run_task(job.tasks[i])
    }
    rank_sync()
  }
  widen(process, &job, 8)
*/

//== Widening to multi-rank context.

// Run the provided function in the widened-rank context, calling thread participates as rank 0.
widen :: proc(entry: proc(data: $T), data: T, n_ranks: int) {
	assert(n_ranks > 0)
	assert(n_ranks <= MAX_RANKS, "rank_count exceeds MAX_RANKS")
	assert(rank_count() <= 1, "Widening from multiple existing threads is probably a bad idea.")

	scratch_guard()

	// Save this thread's rank context and narrow state.
	old_ctx := rank_ctx
	old_narrow_depth := narrow_depth
	old_narrow_saved := narrow_saved_ctx
	narrow_depth = 0
	defer {
		rank_ctx = old_ctx
		narrow_depth = old_narrow_depth
		narrow_saved_ctx = old_narrow_saved
	}

	// Serial fast path.
	if n_ranks == 1 {
		rank_ctx = Rank_Ctx {
			idx   = 0,
			count = 1,
		}
		entry(data)
		return
	}

	barrier: sync.Barrier
	sync.barrier_init(&barrier, n_ranks)

	shared_ptrs: [MAX_RANKS]rawptr

	Thread_Params :: struct {
		rank:  Rank_Ctx,
		entry: proc(data: T),
		data:  T,
	}

	params := make([]Thread_Params, n_ranks, scratch())
	threads := make([]^thread.Thread, n_ranks - 1, scratch())

	// Launch worker ranks
	for i in 1 ..< n_ranks {
		params[i] = Thread_Params {
			rank = Rank_Ctx{idx = i, count = n_ranks, barrier = &barrier, shared_ptrs = &shared_ptrs},
			entry = entry,
			data = data,
		}

		threads[i - 1] = thread.create_and_start_with_poly_data(&params[i], proc(p: ^Thread_Params) {
			rank_ctx = p.rank
			p.entry(p.data)
		})
	}

	// Rank 0 runs on the calling thread.
	rank_ctx = Rank_Ctx {
		idx         = 0,
		count       = n_ranks,
		barrier     = &barrier,
		shared_ptrs = &shared_ptrs,
	}

	entry(data)

	thread.join_multiple(..threads)
	for t in threads { thread.destroy(t) }
}
