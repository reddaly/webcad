package constraints_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonzojive/webcad/go/solvers/core"
	"github.com/gonzojive/webcad/go/solvers/core/constraints"
	"github.com/gonzojive/webcad/schema"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/mat"
)

func TestConcentricEvaluator(t *testing.T) {
	// 1. Construct a scenario with two concentric circles
	c1 := &schema.Entity{
		Id: "c1",
		EntityType: &schema.Entity_Circle{
			Circle: &schema.CircleEntity{Cx: 1.0, Cy: 2.0, R: 3.0},
		},
	}
	c2 := &schema.Entity{
		Id: "c2",
		EntityType: &schema.Entity_Circle{
			Circle: &schema.CircleEntity{Cx: 1.5, Cy: 2.5, R: 4.0},
		},
	}
	concentric := &schema.Constraint{
		Id: "con1",
		ConstraintType: &schema.Constraint_Concentric{
			Concentric: &schema.ConcentricConstraint{
				EntityA: "c1",
				EntityB: "c2",
			},
		},
	}
	scenario := &schema.Sketch{
		Entities:    []*schema.Entity{c1, c2},
		Constraints: []*schema.Constraint{concentric},
	}

	// 2. Construct monolithic ConstraintSystem
	sys, err := core.NewConstraintSystem(scenario)
	if err != nil {
		t.Fatalf("NewConstraintSystem failed: %v", err)
	}

	// 3. Construct decomposed evaluator
	entities := map[string]*schema.Entity{"c1": c1, "c2": c2}
	eval, err := constraints.NewEvaluator(concentric, entities)
	if err != nil {
		t.Fatalf("failed to create evaluator: %v", err)
	}

	// Verify it implements JacobianEvaluator
	je, ok := eval.(constraints.JacobianEvaluator)
	if !ok {
		t.Fatalf("evaluator does not implement JacobianEvaluator")
	}

	// 4. Test over multiple random states
	rng := rand.New(rand.NewSource(42))
	paramIndices := map[string]int{"c1": 0, "c2": 3} // c1 has 3 params, c2 starts at 3
	n := sys.NumVars()
	m := je.NumEquations()

	if m != 2 {
		t.Errorf("expected 2 equations, got %d", m)
	}

	for k := 0; k < 10; k++ {
		// Perturb state (centers and radii)
		x := []float64{
			1.0 + (rng.Float64()-0.5)*2.0,
			2.0 + (rng.Float64()-0.5)*2.0,
			3.0 + rng.Float64(),
			1.5 + (rng.Float64()-0.5)*2.0,
			2.5 + (rng.Float64()-0.5)*2.0,
			4.0 + rng.Float64(),
		}

		// 1. Test backward-compatible Evaluate
		valExpected := sys.Objective(x)
		gradExpected := make([]float64, n)
		sys.ObjectiveGradient(gradExpected, x)

		gradActual := make([]float64, n)
		valActual := eval.Evaluate(x, gradActual, paramIndices)

		// Compare value
		if math.Abs(valActual-valExpected) > 1e-9 {
			t.Errorf("[%d] Evaluate value mismatch: got %f, want %f", k, valActual, valExpected)
		}

		// Compare gradient
		for i := range gradExpected {
			if math.Abs(gradActual[i]-gradExpected[i]) > 1e-9 {
				t.Errorf("[%d] Evaluate gradient mismatch at %d: got %f, want %f", k, i, gradActual[i], gradExpected[i])
			}
		}

		// 2. Test EvaluateJacobian
		residuals := make([]float64, m)
		J := mat.NewDense(m, n, nil)
		je.EvaluateJacobian(x, residuals, J, 0, paramIndices)

		// Verify residuals sum of squares matches Evaluate value
		resSqSum := 0.0
		for _, r := range residuals {
			resSqSum += r * r
		}
		if math.Abs(resSqSum-valActual) > 1e-9 {
			t.Errorf("[%d] Jacobian residuals sum of squares %f does not match Evaluate value %f", k, resSqSum, valActual)
		}

		// Verify analytical Jacobian against numerical finite differences
		f := func(dst, xs []float64) {
			je.EvaluateJacobian(xs, dst, nil, 0, paramIndices)
		}
		fdJacobian := mat.NewDense(m, n, nil)
		fd.Jacobian(fdJacobian, f, x, &fd.JacobianSettings{Formula: fd.Central})

		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				diff := math.Abs(J.At(i, j) - fdJacobian.At(i, j))
				if diff > 1e-5 {
					t.Errorf("[%d] Jacobian mismatch at (%d, %d): got %f, want %f", k, i, j, J.At(i, j), fdJacobian.At(i, j))
				}
			}
		}
	}
}
