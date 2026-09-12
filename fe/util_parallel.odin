package fe

/*
 Parallel model based on https://www.dgtlgrove.com/p/multi-core-by-default.
 Does not follow article implementation exactly. Uses terminolog `rank` instead of `lane`
 used by the article as lane is reserved terminology for SIMD.

 This is a single program multiple data approach to parallelization.
 All ranks within a `widen` region must execute collective operations
 (`rank_sync`, `rank_reduce`, `rank_sync_value`, `narrow`, etc.) in the
 same order. Divergence between ranks may result in deadlock.
*/

import "base:intrinsics"
import "core:sync"
import "core:thread"

MAX_RANKS :: 128

@(thread_local, private)
rank_ctx: Rank_Ctx

Rank_Ctx :: struct {
	idx:         int,
	count:       int,
	barrier:     ^sync.Barrier,
	shared_ptrs: ^[MAX_RANKS]rawptr,
}

// main thread has valid context even without a `widen` call.
@(init)
init_default_rank_ctx :: proc "contextless" () {
	rank_ctx = Rank_Ctx {
		idx   = 0,
		count = 1,
	}
}

//== rank accessing

rank_idx :: proc() -> int {
	return rank_ctx.idx
}

rank_count :: proc() -> int {
	return rank_ctx.count
}

//== narrowing (temporarily single threaded execution)

// Note: this is more ceremony then the article had, but its for running code that expects to be run in a rank
// context run within a narrow. In the original article, narrow was simpler but would fail for code that expected a valid
// rank context. Manual `if rank_idx() == 0 {...} rank_sync()` retains the articles semantics if needed.


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

//== synchronization

// Block until all ranks have arrived
rank_sync :: proc() {
	if rank_count() > 1 { sync.barrier_wait(rank_ctx.barrier) }
}

// This diverges from the article, there broadcast approach was faster but less flexible.
// Warning: by defualt writes the value from rank 0, the default narrowing rank.
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

// Combine all ranks `local` with the given procedure. For simple numerical types see `rank_sum`.
rank_reduce :: proc(local: $T, $combine: proc(a, b: T) -> T) -> T {
	if rank_count() <= 1 { return local }
	result := local
	rank_ctx.shared_ptrs[rank_idx()] = &result
	rank_sync()
	if narrow() {
		for i in 1 ..< rank_count() {
			other := (cast(^T)rank_ctx.shared_ptrs[i])^
			result = combine(result, other)
		}
	}
	rank_sync_value(&result)
	return result
}

// Sum each ranks `local`. Given local must support `+` operator.
rank_sum :: proc(local: $T) -> T where intrinsics.type_is_numeric(T) {
	return rank_reduce(local, proc(a, b: T) -> T { return a + b })
}

//== static work distribution

Range :: struct {
	min, max: int,
}

// Split count roughly equally among ranks, each rank will recieve its range.
rank_range :: proc(count: int) -> Range {
	// Ripped directly from our boy Ryan Fluery
	per := count / rank_count()
	leftover := count % rank_count()
	has_left := rank_idx() < leftover
	before := has_left ? rank_idx() : leftover
	first := per * rank_idx() + before
	opl := first + per + (has_left ? 1 : 0)
	return Range{first, opl}
}

//== Dynamic work distribution

Task_Pool :: struct {
	counter: int, // atomic
	count:   int,
}

rank_task_pool_init :: proc(pool: ^Task_Pool, count: int) {
	if narrow() { pool.counter = 0; pool.count = count }
}

rank_task_pool_next :: proc(pool: ^Task_Pool) -> (idx: int, ok: bool) {
	idx = sync.atomic_add(&pool.counter, 1)
	ok = idx < pool.count
	return
}

//== Widening to mulit-rank context.

// Run the provided function in the widened-rank context, calling thread participates.
// It is safe to widen, within a widen, but the inner work must end before the outer widen can continue.
widen :: proc(entry: proc(data: $T), data: T, rank_count: int) {
	assert(rank_count > 0)
	assert(rank_count <= MAX_RANKS, "rank_count exceeds MAX_RANKS")

	scratch_guard()

	old_ctx := rank_ctx
	defer rank_ctx = old_ctx

	// Serial fast path.
	if rank_count == 1 {
		rank_ctx = Rank_Ctx {
			idx   = 0,
			count = 1,
		}
		entry(data)
		return
	}

	barrier: sync.Barrier
	sync.barrier_init(&barrier, rank_count)

	shared_ptrs: [MAX_RANKS]rawptr

	Thread_Params :: struct {
		rank:  Rank_Ctx,
		entry: proc(data: $T),
		data:  T,
	}

	params := make([]Thread_Params, rank_count, scratch())
	threads := make([]^thread.Thread, rank_count - 1, scratch())

	// Launch worker ranks
	for i in 1 ..< rank_count {
		params[i] = Thread_Params {
			rank = Rank_Ctx{idx = i, count = rank_count, barrier = &barrier, shared_ptrs = &shared_ptrs},
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
		count       = rank_count,
		barrier     = &barrier,
		shared_ptrs = &shared_ptrs,
	}

	entry(data)

	thread.join_multiple(..threads)
}
