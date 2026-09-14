package fe

/*
 Basic thread-local scratch allocator.

 Lazily initialized for convienence when starting up new threads.
 Backed by an arena for quick allocations.
 Arena is automatically destroyed when thread is released.
*/

import "base:runtime"
import "core:mem"
import "core:mem/virtual"

// TODO: replace with bespoke implementation that doenst have a mutex we dont need thread safety here.

Scratch_Temp  :: virtual.Arena_Temp
Scratch_Arena :: virtual.Arena

@(thread_local, private)
_scratch_arena: Scratch_Arena

scratch :: proc() -> mem.Allocator {
	return virtual.arena_allocator(scratch_arena())
}

scratch_used :: proc() -> uint {
	return _scratch_arena.total_used
}

scratch_begin_temp :: proc() -> Scratch_Temp {
	return virtual.arena_temp_begin(scratch_arena())
}

scratch_end_temp :: proc(temp: Scratch_Temp) {
	virtual.arena_temp_end(temp)
}

// Automatically frees any memory allocated at the end of calling scope.
@(deferred_out = scratch_end_temp)
scratch_guard :: proc() -> Scratch_Temp {
	return scratch_begin_temp()
}

@(private)
scratch_arena :: proc() -> ^Scratch_Arena {
	if _scratch_arena == {} { 	// lazy setup
		if err := virtual.arena_init_growing(&_scratch_arena); err != nil { panic("failed to init scratch arena") }
		runtime.add_thread_local_cleaner(proc "contextless" () {
			context = runtime.default_context()
			virtual.arena_destroy(&_scratch_arena)
		})
	}
	return &_scratch_arena
}
