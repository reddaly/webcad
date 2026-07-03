package lm

import (
	"math"
	"testing"

	"github.com/gonzojive/webcad/go/solvers/core"
	"github.com/gonzojive/webcad/schema"
	"gonum.org/v1/gonum/lapack/lapack64"
	"gonum.org/v1/gonum/mat"
)

func TestLMSolverConvergence(t *testing.T) {
	// 1. Construct a simple scenario:
	// - Point p1 (fixed at 1.0, 2.0)
	// - Point p2 (concentric/coincident with p1, starting at 3.0, 4.0)
	p1 := &schema.Entity{
		Id: "p1",
		EntityType: &schema.Entity_Point{
			Point: &schema.PointEntity{X: 1.0, Y: 2.0},
		},
	}
	p2 := &schema.Entity{
		Id: "p2",
		EntityType: &schema.Entity_Point{
			Point: &schema.PointEntity{X: 3.0, Y: 4.0}, // Initial guess far away
		},
	}
	cFixed := &schema.Constraint{
		Id: "c_fixed",
		ConstraintType: &schema.Constraint_Fixed{
			Fixed: &schema.FixedConstraint{
				EntityId: "p1",
			},
		},
	}
	cConcentric := &schema.Constraint{
		Id: "c_concentric",
		ConstraintType: &schema.Constraint_Concentric{
			Concentric: &schema.ConcentricConstraint{
				EntityA: "p1",
				EntityB: "p2",
			},
		},
	}
	scenario := &schema.Sketch{
		Entities:    []*schema.Entity{p1, p2},
		Constraints: []*schema.Constraint{cFixed, cConcentric},
	}

	// 2. Initialize solver
	solver := New()

	// 3. Run SolveCold (which runs multiple trials, but we just want to verify one solve first)
	scenario.ColdRepeatedTrials = 1
	result, err := solver.SolveCold(scenario)
	if err != nil {
		t.Fatalf("SolveCold failed: %v", err)
	}

	if !result.Success {
		t.Fatalf("solver reported failure: %s", result.ErrorMessage)
	}

	// 4. Verify B converged to A (1.0, 2.0)
	solvedP2 := result.SolvedState.Entities["p2"].GetPoint()
	if math.Abs(solvedP2.X-1.0) > 1e-8 {
		t.Errorf("expected p2.X to be 1.0, got %f", solvedP2.X)
	}
	if math.Abs(solvedP2.Y-2.0) > 1e-8 {
		t.Errorf("expected p2.Y to be 2.0, got %f", solvedP2.Y)
	}

	t.Logf("Converged in %d iterations", result.Telemetry.Iterations)
	t.Logf("Final residual (sum of squares): %e", result.Telemetry.FinalResidual)
}

func TestLMSolverZeroAllocations(t *testing.T) {
	// Setup the same scenario
	p1 := &schema.Entity{
		Id: "p1",
		EntityType: &schema.Entity_Point{
			Point: &schema.PointEntity{X: 1.0, Y: 2.0},
		},
	}
	p2 := &schema.Entity{
		Id: "p2",
		EntityType: &schema.Entity_Point{
			Point: &schema.PointEntity{X: 3.0, Y: 4.0},
		},
	}
	cFixed := &schema.Constraint{
		Id: "c_fixed",
		ConstraintType: &schema.Constraint_Fixed{
			Fixed: &schema.FixedConstraint{
				EntityId: "p1",
			},
		},
	}
	cConcentric := &schema.Constraint{
		Id: "c_concentric",
		ConstraintType: &schema.Constraint_Concentric{
			Concentric: &schema.ConcentricConstraint{
				EntityA: "p1",
				EntityB: "p2",
			},
		},
	}
	scenario := &schema.Sketch{
		Entities:    []*schema.Entity{p1, p2},
		Constraints: []*schema.Constraint{cFixed, cConcentric},
	}

	solver := New()
	sys, err := core.NewConstraintSystem(scenario)
	if err != nil {
		t.Fatalf("NewConstraintSystem failed: %v", err)
	}
	x := sys.ExtractVariables()
	initialX := make([]float64, len(x))
	copy(initialX, x)

	// Warm up the solver (pre-allocates workspace in the pool)
	_, _ = solver.solve(sys, x)

	// Measure allocations of the core solve loop
	allocs := testing.AllocsPerRun(100, func() {
		copy(x, initialX)
		_, _ = solver.solve(sys, x)
	})

	if allocs > 0 {
		t.Errorf("expected 0 allocations in solver loop, got %f", allocs)
	} else {
		t.Logf("Verified 0 heap allocations in the solver hot loop!")
	}
}

func TestQRSolver(t *testing.T) {
	// Construct a simple J and f to solve (JᵀJ + μI) dx = -Jᵀf
	J := mat.NewDense(2, 2, []float64{
		2, 1,
		1, 3,
	})
	f := mat.NewVecDense(2, []float64{5, 1})
	mu := 0.1
	n := 2
	m := 2

	// 1. Compute expected solution using Gonum's standard Cholesky on normal equations
	var JT mat.Dense
	JT.CloneFrom(J.T())
	var JTJ mat.Dense
	JTJ.Mul(&JT, J)

	H := mat.NewSymDense(n, nil)
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			val := JTJ.At(r, c)
			if r == c {
				val += mu
			}
			H.SetSym(r, c, val)
		}
	}

	var g mat.VecDense
	g.MulVec(&JT, f)

	var chol mat.Cholesky
	if ok := chol.Factorize(H); !ok {
		t.Fatalf("Cholesky factorization failed")
	}
	var dxExpected mat.VecDense
	var negG mat.VecDense
	negG.ScaleVec(-1.0, &g)
	if err := chol.SolveVecTo(&dxExpected, &negG); err != nil {
		t.Fatalf("Cholesky solve failed: %v", err)
	}

	// 2. Solve using our 100% allocation-free solveQRAugmented
	aAug := mat.NewDense(m+n, n, nil)
	bAug := mat.NewVecDense(m+n, nil)
	dxActual := mat.NewVecDense(n, nil)
	tau := make([]float64, n)

	// Query work size
	workQuery := []float64{0}
	lapack64.Geqrf(aAug.RawMatrix(), tau, workQuery, -1)
	lwork := int(workQuery[0])
	work := make([]float64, lwork)

	// Call our QR solver
	solveQRAugmented(J, f, mu, aAug, bAug, dxActual, tau, work)

	// 3. Verify results match
	for i := 0; i < n; i++ {
		diff := math.Abs(dxActual.AtVec(i) - dxExpected.AtVec(i))
		if diff > 1e-12 {
			t.Errorf("expected dx[%d] = %f, got %f (diff: %e)", i, dxExpected.AtVec(i), dxActual.AtVec(i), diff)
		}
	}
}
