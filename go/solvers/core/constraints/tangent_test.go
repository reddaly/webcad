package constraints_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gonzojive/webcad/go/solvers/core/constraints"
	"github.com/gonzojive/webcad/schema"
	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/mat"
)

func TestTangentEvaluator_CirCir_Jacobian(t *testing.T) {
	c1 := &schema.Entity{Id: "c1", EntityType: &schema.Entity_Circle{Circle: &schema.CircleEntity{Cx: 0.0, Cy: 0.0, R: 1.0}}}
	c2 := &schema.Entity{Id: "c2", EntityType: &schema.Entity_Circle{Circle: &schema.CircleEntity{Cx: 3.0, Cy: 0.0, R: 2.0}}}
	c := &schema.Constraint{
		Id: "c",
		ConstraintType: &schema.Constraint_Tangent{
			Tangent: &schema.TangentConstraint{EntityA: "c1", EntityB: "c2"},
		},
	}
	eval, err := constraints.NewEvaluator(c, map[string]*schema.Entity{"c1": c1, "c2": c2})
	if err != nil {
		t.Fatalf("failed to create evaluator: %v", err)
	}

	je, ok := eval.(constraints.JacobianEvaluator)
	if !ok {
		t.Fatalf("evaluator does not implement JacobianEvaluator")
	}

	rng := rand.New(rand.NewSource(42))
	paramIndices := map[string]int{"c1": 0, "c2": 3}
	n := 6
	m := je.NumEquations()

	if m != 1 {
		t.Errorf("expected 1 equation for Cir-Cir, got %d", m)
	}

	for k := 0; k < 10; k++ {
		x := []float64{
			0.0 + (rng.Float64()-0.5)*0.2, 0.0 + (rng.Float64()-0.5)*0.2,
			1.0 + (rng.Float64()-0.5)*0.1,
			3.0 + (rng.Float64()-0.5)*0.2, 0.0 + (rng.Float64()-0.5)*0.2,
			2.0 + (rng.Float64()-0.5)*0.1,
		}

		gradActual := make([]float64, n)
		valActual := eval.Evaluate(x, gradActual, paramIndices)

		// Test EvaluateJacobian
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

func TestTangentEvaluator_CirLn_Jacobian(t *testing.T) {
	c1 := &schema.Entity{Id: "c1", EntityType: &schema.Entity_Circle{Circle: &schema.CircleEntity{Cx: 0.0, Cy: 2.0, R: 1.0}}}
	l1 := &schema.Entity{Id: "l1", EntityType: &schema.Entity_Line{Line: &schema.LineEntity{X1: -2.0, Y1: 0.0, X2: 2.0, Y2: 0.0}}}
	c := &schema.Constraint{
		Id: "c",
		ConstraintType: &schema.Constraint_Tangent{
			Tangent: &schema.TangentConstraint{EntityA: "c1", EntityB: "l1"},
		},
	}
	eval, err := constraints.NewEvaluator(c, map[string]*schema.Entity{"c1": c1, "l1": l1})
	if err != nil {
		t.Fatalf("failed to create evaluator: %v", err)
	}

	je, ok := eval.(constraints.JacobianEvaluator)
	if !ok {
		t.Fatalf("evaluator does not implement JacobianEvaluator")
	}

	rng := rand.New(rand.NewSource(42))
	paramIndices := map[string]int{"c1": 0, "l1": 3}
	n := 7
	m := je.NumEquations()

	if m != 1 {
		t.Errorf("expected 1 equation for Cir-Ln, got %d", m)
	}

	for k := 0; k < 10; k++ {
		x := []float64{
			0.0 + (rng.Float64()-0.5)*0.2, 2.0 + (rng.Float64()-0.5)*0.2,
			1.0 + (rng.Float64()-0.5)*0.1,
			-2.0 + (rng.Float64()-0.5)*0.2, 0.0 + (rng.Float64()-0.5)*0.2,
			2.0 + (rng.Float64()-0.5)*0.2, 0.0 + (rng.Float64()-0.5)*0.2,
		}

		gradActual := make([]float64, n)
		valActual := eval.Evaluate(x, gradActual, paramIndices)

		// Test EvaluateJacobian
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
