package fe

/*
 Time integration.

 The clock manages the time between integrators, the integrator manages the time stepping coefficients. Start the
 clock and the integrator at the same t0.

 Each step solves, for any first order scheme, with F the spatial operator and M the mass,
     shift M (u - hist) + F(u, t) + old_weight F(old, t_old) = 0

 old_weight is 0 for BDF. Second order schemes solve the same form with shift M (u - hist) the inertia, the velocity
 v = shift_v (u - hist_v) inside F(u, v, t), and F(old, old_v, t_old) in the old term (0 for Newmark).
*/

import "core:mem/virtual"

Time_Scheme :: enum {
	BDF1,
	BDF2,
	Theta, // theta = 0.5 is Crank-Nicolson
	Newmark, // average acceleration, beta = 1/4, gamma = 1/2
	Generalized_Alpha,
}

// History of one array and the buffers of its current step.
Integrator :: struct {
	scheme: Time_Scheme,
	theta:  f64,
	t:      f64, // time of past[0]
	past:   [MAX_HISTORY][]f64, // accepted solutions, newest first
	times:  [MAX_HISTORY]f64,
	n_past: int,
	hist:   []f64,
	pred:   []f64,
	arena:  virtual.Arena,

	// second order schemes
	v, a:             []f64, // velocity and acceleration at t
	hist_v:           []f64,
	beta, gamma:      f64,
	alpha_m, alpha_f: f64,
}

// One step from t_old to t: du/dt ~ shift (u - hist).
Step :: struct {
	t, t_old, dt: f64,
	order:        int, // ramps up while the history fills
	shift:        f64,
	hist:         []f64,
	old_weight:   f64, // weight on F(u_old, t_old), 0 for BDF and Newmark
	old:          []f64, // u at t_old
	pred:         []f64, // extrapolated to t: Newton initial guess, linearised convection

	// second order schemes: shift (u - hist) is d2u/dt2 and du/dt ~ shift_v (u - hist_v)
	shift_v:      f64,
	hist_v:       []f64,
	old_v:        []f64, // du/dt at t_old, for damping in the old_weight term
}

// A fixed step loop with an output schedule. Plain data, each rank keeps its own.
Clock :: struct {
	t, t_end, dt: f64,
	n:            int, // steps taken
	out_dt:       f64, // 0 = no output schedule
	next_out:     f64,
	n_out:        int, // output times passed
}

MAX_HISTORY :: 2

// Times within this fraction of a step are the same time.
CLOSE_ENOUGH :: 1e-12

// First order in time. `initial` (an array, or a Sys State) is copied as the solution at t0.
integ_create :: proc(scheme: Time_Scheme, initial: []f64, t0: f64, theta := 0.5) -> (integ: Integrator) {
	assert(scheme == .BDF1 || scheme == .BDF2 || scheme == .Theta, "second order schemes use integ_create_second_order")
	assert(scheme != .Theta || (theta > 0 && theta <= 1), "theta must be in (0, 1]")
	virtual.arena_init_growing(&integ.arena) or_else panic("Failed to create arena")
	alloc := virtual.arena_allocator(&integ.arena)

	integ.scheme = scheme
	integ.theta = theta
	integ.t = t0

	for &p in integ.past { p = make([]f64, len(initial), alloc) }

	integ.hist = make([]f64, len(initial), alloc)
	integ.pred = make([]f64, len(initial), alloc)
	copy(integ.past[0], initial)
	integ.times[0] = t0
	integ.n_past = 1
	return
}

// Second order in time. `a0` nil is a zero initial acceleration: solve M a0 = f - F(u0, v0) for a consistent one,
// or expect a start-up transient. `rho_inf` is generalized alpha's high frequency damping, Newmark ignores it.
integ_create_second_order :: proc(
	scheme: Time_Scheme,
	u0, v0: []f64,
	t0: f64,
	a0: []f64 = nil,
	rho_inf := 1.0,
) -> (
	integ: Integrator,
) {
	assert(scheme == .Newmark || scheme == .Generalized_Alpha, "first order schemes use integ_create")
	assert(len(v0) == len(u0) && (a0 == nil || len(a0) == len(u0)))
	assert(rho_inf >= 0 && rho_inf <= 1, "rho_inf must be in [0, 1]")
	virtual.arena_init_growing(&integ.arena) or_else panic("Failed to create arena")
	alloc := virtual.arena_allocator(&integ.arena)

	integ.scheme = scheme
	integ.t = t0
	for &p in integ.past { p = make([]f64, len(u0), alloc) }
	integ.hist = make([]f64, len(u0), alloc)
	integ.pred = make([]f64, len(u0), alloc)
	integ.hist_v = make([]f64, len(u0), alloc)
	integ.v = make([]f64, len(u0), alloc)
	integ.a = make([]f64, len(u0), alloc)
	copy(integ.past[0], u0)
	copy(integ.v, v0)
	if a0 != nil { copy(integ.a, a0) }
	integ.times[0] = t0
	integ.n_past = 1

	integ.beta, integ.gamma = 0.25, 0.5
	if scheme == .Generalized_Alpha {
		integ.alpha_m = (2 * rho_inf - 1) / (rho_inf + 1)
		integ.alpha_f = rho_inf / (rho_inf + 1)
		integ.gamma = 0.5 - integ.alpha_m + integ.alpha_f
		integ.beta = 0.25 * (1 - integ.alpha_m + integ.alpha_f) * (1 - integ.alpha_m + integ.alpha_f)
	}
	return
}

integ_destroy :: proc(integs: ..^Integrator) {
	for integ in integs { virtual.arena_destroy(&integ.arena) }
}

// Coefficients and hist / old / pred for the step to integ.t + dt. The history is untouched until integ_accept, so a
// rejected step can be retried with another dt. Collective.
integ_step :: proc(integ: ^Integrator, dt: f64) -> (step: Step) {
	assert(dt > 0)
	step.t_old = integ.t
	step.t = integ.t + dt
	step.dt = dt
	step.old = integ.past[0]
	step.hist = integ.hist
	step.pred = integ.pred

	// hist = sum_j h[j] past[j], pred = sum_j e[j] past[j]
	h, e: [MAX_HISTORY]f64
	n_pred := 0
	switch integ.scheme {
	case .BDF1, .BDF2:
		k := min(int(integ.scheme) + 1, integ.n_past)
		step.order = k

		// BDF-k is the derivative at t of the interpolant through t and the k past times:
		// du/dt = l0' u + sum_j lj' past[j], so shift = l0' and hist = -sum_j lj' / l0' past[j]
		nodes: [MAX_HISTORY + 1]f64
		nodes[0] = step.t
		for j in 0 ..< k { nodes[j + 1] = integ.times[j] }
		for m in 1 ..= k { step.shift += 1 / (step.t - nodes[m]) }
		for j in 1 ..= k {
			d := 1 / (nodes[j] - step.t)
			for m in 1 ..= k {
				if m != j { d *= (step.t - nodes[m]) / (nodes[j] - nodes[m]) }
			}
			h[j - 1] = -d / step.shift
		}
		n_pred = k
	case .Theta:
		step.order = 2 if integ.theta == 0.5 else 1
		step.shift = 1 / (integ.theta * dt)
		step.old_weight = (1 - integ.theta) / integ.theta
		h[0] = 1
		n_pred = min(2, integ.n_past)
	case .Newmark, .Generalized_Alpha:
		// Newmark: a = (u - un_hat) / (beta dt^2), v = vn + dt (1 - gamma) an + gamma dt a, with
		// un_hat = un + dt vn + dt^2 (1/2 - beta) an. Generalized alpha weighs the equation
		// (1 - am) M a + am M an + (1 - af) F(u) + af F(un) = 0, divided through by (1 - af).
		beta, gamma, am, af := integ.beta, integ.gamma, integ.alpha_m, integ.alpha_f
		step.order = 2
		step.shift = (1 - am) / ((1 - af) * beta * dt * dt)
		step.old_weight = af / (1 - af)
		step.shift_v = gamma / (beta * dt)
		step.hist_v = integ.hist_v
		step.old_v = integ.v

		rank_sync()
		sub := rank_range(len(integ.hist))
		for i in sub.min ..< sub.max {
			un, vn, an := integ.past[0][i], integ.v[i], integ.a[i]
			un_hat := un + dt * vn + dt * dt * (0.5 - beta) * an
			integ.hist[i] = un_hat - am * beta * dt * dt / (1 - am) * an
			integ.hist_v[i] = un_hat - beta * dt / gamma * (vn + dt * (1 - gamma) * an)
			integ.pred[i] = un + dt * vn + 0.5 * dt * dt * an
		}
		rank_sync()
		return
	}

	// extrapolation to t through the last n_pred solutions
	for j in 0 ..< n_pred {
		e[j] = 1
		for m in 0 ..< n_pred {
			if m != j { e[j] *= (step.t - integ.times[m]) / (integ.times[j] - integ.times[m]) }
		}
	}

	rank_sync()
	sub := rank_range(len(integ.hist))
	for i in sub.min ..< sub.max {
		hi, ei: f64
		for j in 0 ..< integ.n_past {
			hi += h[j] * integ.past[j][i]
			ei += e[j] * integ.past[j][i]
		}
		integ.hist[i], integ.pred[i] = hi, ei
	}
	rank_sync()
	return
}

// `solution` (an array, or a Sys State) is the accepted solution at step.t, it goes into the history. Collective.
integ_accept :: proc(integ: ^Integrator, step: Step, solution: []f64) {
	assert(step.t_old == integ.t, "step is not from the integrator's current time")
	assert(len(solution) == len(integ.hist))

	// the oldest buffer is reused for the new solution
	newest := integ.past[MAX_HISTORY - 1]
	rank_sync()
	sub := rank_range(len(solution))
	if integ.scheme == .Newmark || integ.scheme == .Generalized_Alpha {
		dt, beta, gamma := step.dt, integ.beta, integ.gamma
		for i in sub.min ..< sub.max {
			un, vn, an := integ.past[0][i], integ.v[i], integ.a[i]
			a := (solution[i] - un - dt * vn - dt * dt * (0.5 - beta) * an) / (beta * dt * dt)
			integ.v[i] = vn + dt * ((1 - gamma) * an + gamma * a)
			integ.a[i] = a
		}
	}
	copy(newest[sub.min:sub.max], solution[sub.min:sub.max])
	rank_sync()

	if rank_idx() == 0 {
		for j := MAX_HISTORY - 1; j > 0; j -= 1 {
			integ.past[j] = integ.past[j - 1]
			integ.times[j] = integ.times[j - 1]
		}
		integ.past[0] = newest
		integ.times[0] = step.t
		integ.n_past = min(integ.n_past + 1, MAX_HISTORY)
		integ.t = step.t
	}
	rank_sync()
}

// Steps of `dt` from t0 to t_end, the last one shortened to land on t_end. Outputs every `out_dt` if non-zero.
clock_create :: proc(t0, t_end, dt: f64, out_dt := 0.0) -> Clock {
	assert(dt > 0 && t_end > t0)
	return {t = t0, t_end = t_end, dt = dt, out_dt = out_dt, next_out = t0 + out_dt}
}

// Iterator: the next step's dt, advancing clock.t to the step's end time.
clock_next :: proc(c: ^Clock) -> (dt: f64, ok: bool) {
	remaining := c.t_end - c.t
	if remaining <= CLOSE_ENOUGH * c.dt { return }
	dt = c.dt if remaining > c.dt * (1 + CLOSE_ENOUGH) else remaining
	c.t += dt
	c.n += 1
	return dt, true
}

// True once per output time passed by the step just taken, with the output's index: output k is at t0 + k out_dt, so
// the first due is 1 and t0 itself is 0.
clock_output_due :: proc(c: ^Clock) -> (index: int, ok: bool) {
	if c.out_dt <= 0 || c.t < c.next_out - CLOSE_ENOUGH * c.dt { return }
	for c.next_out <= c.t + CLOSE_ENOUGH * c.dt {
		c.next_out += c.out_dt
		c.n_out += 1
	}
	return c.n_out, true
}
