package core

import (
	"math"
	"testing"

	"github.com/gonzojive/webcad/schema"
	"gonum.org/v1/gonum/diff/fd"
)

// Helpers to create entities
func Point(id string, x, y float64) *schema.Entity {
	return &schema.Entity{
		Id: id,
		EntityType: &schema.Entity_Point{
			Point: &schema.PointEntity{X: x, Y: y},
		},
	}
}

func Line(id string, x1, y1, x2, y2 float64) *schema.Entity {
	return &schema.Entity{
		Id: id,
		EntityType: &schema.Entity_Line{
			Line: &schema.LineEntity{X1: x1, Y1: y1, X2: x2, Y2: y2},
		},
	}
}

func Circle(id string, cx, cy, r float64) *schema.Entity {
	return &schema.Entity{
		Id: id,
		EntityType: &schema.Entity_Circle{
			Circle: &schema.CircleEntity{Cx: cx, Cy: cy, R: r},
		},
	}
}

// Helpers to create constraints
func Coincidence(id, entA, entB string) *schema.Constraint {
	return &schema.Constraint{
		Id: id,
		ConstraintType: &schema.Constraint_Coincidence{
			Coincidence: &schema.CoincidenceConstraint{
				EntityA: entA,
				EntityB: entB,
			},
		},
	}
}

func Distance(id, entA, entB string, value float64) *schema.Constraint {
	return &schema.Constraint{
		Id: id,
		ConstraintType: &schema.Constraint_Distance{
			Distance: &schema.DistanceConstraint{
				EntityA: entA,
				EntityB: entB,
				Value:   value,
			},
		},
	}
}

func Angle(id, entA, entB string, valueRad float64) *schema.Constraint {
	return &schema.Constraint{
		Id: id,
		ConstraintType: &schema.Constraint_Angle{
			Angle: &schema.AngleConstraint{
				EntityA:      entA,
				EntityB:      entB,
				ValueRadians: valueRad,
			},
		},
	}
}

func Parallel(id, lnA, lnB string) *schema.Constraint {
	return &schema.Constraint{
		Id: id,
		ConstraintType: &schema.Constraint_Parallel{
			Parallel: &schema.ParallelConstraint{
				LineA: lnA,
				LineB: lnB,
			},
		},
	}
}

func Perpendicular(id, lnA, lnB string) *schema.Constraint {
	return &schema.Constraint{
		Id: id,
		ConstraintType: &schema.Constraint_Perpendicular{
			Perpendicular: &schema.PerpendicularConstraint{
				LineA: lnA,
				LineB: lnB,
			},
		},
	}
}

func Tangent(id, entA, entB string) *schema.Constraint {
	return &schema.Constraint{
		Id: id,
		ConstraintType: &schema.Constraint_Tangent{
			Tangent: &schema.TangentConstraint{
				EntityA: entA,
				EntityB: entB,
			},
		},
	}
}

func Concentric(id, entA, entB string) *schema.Constraint {
	return &schema.Constraint{
		Id: id,
		ConstraintType: &schema.Constraint_Concentric{
			Concentric: &schema.ConcentricConstraint{
				EntityA: entA,
				EntityB: entB,
			},
		},
	}
}

func Symmetric(id, entA, entB, symLn string) *schema.Constraint {
	return &schema.Constraint{
		Id: id,
		ConstraintType: &schema.Constraint_Symmetric{
			Symmetric: &schema.SymmetricConstraint{
				EntityA:      entA,
				EntityB:      entB,
				SymmetryLine: symLn,
			},
		},
	}
}

func Midpoint(id, pt, ln string) *schema.Constraint {
	return &schema.Constraint{
		Id: id,
		ConstraintType: &schema.Constraint_Midpoint{
			Midpoint: &schema.MidpointConstraint{
				Point: pt,
				Line:  ln,
			},
		},
	}
}

func Fixed(id, entId string) *schema.Constraint {
	return &schema.Constraint{
		Id: id,
		ConstraintType: &schema.Constraint_Fixed{
			Fixed: &schema.FixedConstraint{
				EntityId: entId,
			},
		},
	}
}

func TestCalculateConstraintResidual(t *testing.T) {
	tests := []struct {
		name       string
		constraint *schema.Constraint
		entities   []*schema.Entity
		initial    []*schema.Entity // Only needed for Fixed constraint
		want       float64
		tolerance  float64
	}{
		// --- Coincidence ---
		{
			name:       "Coincidence Pt-Pt Pass",
			constraint: Coincidence("c1", "p1", "p2"),
			entities:   []*schema.Entity{Point("p1", 1, 2), Point("p2", 1, 2)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Coincidence Pt-Pt Fail",
			constraint: Coincidence("c1", "p1", "p2"),
			entities:   []*schema.Entity{Point("p1", 1, 2), Point("p2", 3, 4)},
			want:       math.Sqrt(8), // dist(1,2, 3,4) = sqrt(4+4)
			tolerance:  1e-9,
		},
		{
			name:       "Coincidence Pt-Ln Pass",
			constraint: Coincidence("c1", "p1", "l1"),
			entities:   []*schema.Entity{Point("p1", 1, 1), Line("l1", 0, 0, 2, 2)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Coincidence Pt-Ln Fail",
			constraint: Coincidence("c1", "p1", "l1"),
			entities:   []*schema.Entity{Point("p1", 1, 2), Line("l1", 0, 0, 2, 0)}, // pt at (1,2), line y=0
			want:       2,
			tolerance:  1e-9,
		},
		{
			name:       "Coincidence Pt-Circ Pass",
			constraint: Coincidence("c1", "p1", "circ1"),
			entities:   []*schema.Entity{Point("p1", 1, 0), Circle("circ1", 0, 0, 1)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Coincidence Pt-Circ Fail",
			constraint: Coincidence("c1", "p1", "circ1"),
			entities:   []*schema.Entity{Point("p1", 2, 0), Circle("circ1", 0, 0, 1)},
			want:       3, // Polynomial: dSq - R^2 = 4 - 1 = 3
			tolerance:  1e-9,
		},

		// --- Distance ---
		{
			name:       "Distance Pt-Pt Pass",
			constraint: Distance("c1", "p1", "p2", 5),
			entities:   []*schema.Entity{Point("p1", 0, 0), Point("p2", 3, 4)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Distance Pt-Pt Fail",
			constraint: Distance("c1", "p1", "p2", 4),
			entities:   []*schema.Entity{Point("p1", 0, 0), Point("p2", 3, 4)},
			want:       9, // Polynomial: dSq - D^2 = 25 - 16 = 9
			tolerance:  1e-9,
		},
		{
			name:       "Distance Pt-Ln Pass",
			constraint: Distance("c1", "p1", "l1", 2),
			entities:   []*schema.Entity{Point("p1", 0, 2), Line("l1", 0, 0, 2, 0)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Distance Pt-Ln Fail",
			constraint: Distance("c1", "p1", "l1", 2),
			entities:   []*schema.Entity{Point("p1", 0, 3), Line("l1", 0, 0, 2, 0)},
			want:       5, // Polynomial: dSq - D^2 = 9 - 4 = 5
			tolerance:  1e-9,
		},

		// --- Angle ---
		{
			name:       "Angle Pass",
			constraint: Angle("c1", "l1", "l2", math.Pi/2),
			entities:   []*schema.Entity{Line("l1", 0, 0, 1, 0), Line("l2", 0, 0, 0, 1)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Angle Fail",
			constraint: Angle("c1", "l1", "l2", math.Pi/4),
			entities:   []*schema.Entity{Line("l1", 0, 0, 1, 0), Line("l2", 0, 0, 0, 1)},
			want:       1.0 / math.Sqrt(2), // Polynomial: |(v1.v2)sin(T) - (v1xv2)cos(T)| = |0 - 1/sqrt(2)| = 1/sqrt(2)
			tolerance:  1e-9,
		},

		// --- Parallel ---
		{
			name:       "Parallel Pass",
			constraint: Parallel("c1", "l1", "l2"),
			entities:   []*schema.Entity{Line("l1", 0, 0, 1, 0), Line("l2", 0, 1, 2, 1)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Parallel Fail",
			constraint: Parallel("c1", "l1", "l2"),
			entities:   []*schema.Entity{Line("l1", 0, 0, 1, 0), Line("l2", 0, 0, 1, 1)},
			want:       math.Sin(math.Pi / 4),
			tolerance:  1e-9,
		},

		// --- Perpendicular ---
		{
			name:       "Perpendicular Pass",
			constraint: Perpendicular("c1", "l1", "l2"),
			entities:   []*schema.Entity{Line("l1", 0, 0, 1, 0), Line("l2", 0, 0, 0, 1)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Perpendicular Fail",
			constraint: Perpendicular("c1", "l1", "l2"),
			entities:   []*schema.Entity{Line("l1", 0, 0, 1, 0), Line("l2", 0, 0, 1, 1)},
			want:       math.Cos(math.Pi / 4),
			tolerance:  1e-9,
		},

		// --- Tangent ---
		{
			name:       "Tangent Circ-Circ Ext Pass",
			constraint: Tangent("c1", "circ1", "circ2"),
			entities:   []*schema.Entity{Circle("circ1", 0, 0, 1), Circle("circ2", 3, 0, 2)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Tangent Circ-Circ Int Pass",
			constraint: Tangent("c1", "circ1", "circ2"),
			entities:   []*schema.Entity{Circle("circ1", 0, 0, 3), Circle("circ2", 1, 0, 2)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Tangent Circ-Ln Pass",
			constraint: Tangent("c1", "circ1", "l1"),
			entities:   []*schema.Entity{Circle("circ1", 0, 1, 1), Line("l1", -2, 0, 2, 0)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Tangent Circ-Circ Fail",
			constraint: Tangent("c1", "circ1", "circ2"),
			entities:   []*schema.Entity{Circle("circ1", 0, 0, 1), Circle("circ2", 4, 0, 2)},
			want:       7, // Polynomial: dSq - (r1+r2)^2 = 16 - 9 = 7
			tolerance:  1e-9,
		},
		{
			name:       "Tangent Circ-Ln Fail",
			constraint: Tangent("c1", "circ1", "l1"),
			entities:   []*schema.Entity{Circle("circ1", 0, 2, 1), Line("l1", -2, 0, 2, 0)},
			want:       3, // Polynomial: dSq - R^2 = 4 - 1 = 3
			tolerance:  1e-9,
		},

		// --- Concentric ---
		{
			name:       "Concentric Circ-Circ Pass",
			constraint: Concentric("c1", "circ1", "circ2"),
			entities:   []*schema.Entity{Circle("circ1", 1, 2, 3), Circle("circ2", 1, 2, 5)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Concentric Pt-Circ Pass",
			constraint: Concentric("c1", "p1", "circ1"),
			entities:   []*schema.Entity{Point("p1", 1, 2), Circle("circ1", 1, 2, 3)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Concentric Circ-Circ Fail",
			constraint: Concentric("c1", "circ1", "circ2"),
			entities:   []*schema.Entity{Circle("circ1", 1, 2, 3), Circle("circ2", 2, 2, 5)},
			want:       1,
			tolerance:  1e-9,
		},
		{
			name:       "Concentric Pt-Circ Fail",
			constraint: Concentric("c1", "p1", "circ1"),
			entities:   []*schema.Entity{Point("p1", 1, 3), Circle("circ1", 1, 2, 3)},
			want:       1,
			tolerance:  1e-9,
		},

		// --- Symmetric ---
		{
			name:       "Symmetric Pass",
			constraint: Symmetric("c1", "p1", "p2", "l1"),
			entities:   []*schema.Entity{Point("p1", 1, 1), Point("p2", -1, 1), Line("l1", 0, -2, 0, 2)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Symmetric Fail",
			constraint: Symmetric("c1", "p1", "p2", "l1"),
			entities:   []*schema.Entity{Point("p1", 1, 1), Point("p2", -2, 1), Line("l1", 0, -2, 0, 2)},
			want:       0.5, // Unified math: sqrt(rMid^2 + rPerp^2) = sqrt(0.5^2 + 0) = 0.5
			tolerance:  1e-9,
		},

		// --- Midpoint ---
		{
			name:       "Midpoint Pass",
			constraint: Midpoint("c1", "p1", "l1"),
			entities:   []*schema.Entity{Point("p1", 1, 1), Line("l1", 0, 0, 2, 2)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Midpoint Fail",
			constraint: Midpoint("c1", "p1", "l1"),
			entities:   []*schema.Entity{Point("p1", 1, 2), Line("l1", 0, 0, 2, 2)},
			want:       1,
			tolerance:  1e-9,
		},

		// --- Fixed ---
		{
			name:       "Fixed Pass",
			constraint: Fixed("c1", "p1"),
			entities:   []*schema.Entity{Point("p1", 1, 2)},
			initial:    []*schema.Entity{Point("p1", 1, 2)},
			want:       0,
			tolerance:  1e-9,
		},
		{
			name:       "Fixed Fail",
			constraint: Fixed("c1", "p1"),
			entities:   []*schema.Entity{Point("p1", 2, 4)},
			initial:    []*schema.Entity{Point("p1", 1, 2)},
			want:       math.Sqrt(5000.0), // Unified math: sqrt(1000 * ((2-1)^2 + (4-2)^2)) = sqrt(5000)
			tolerance:  1e-9,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entityMap := make(map[string]*schema.Entity)
			for _, ent := range tt.entities {
				entityMap[ent.Id] = ent
			}

			// Construct scenario
			scenario := &schema.Sketch{
				Entities:    tt.entities,
				Constraints: []*schema.Constraint{tt.constraint},
			}

			if len(tt.initial) > 0 {
				initialMap := make(map[string]*schema.Entity)
				for _, ent := range tt.initial {
					initialMap[ent.Id] = ent
				}
				scenario.InitialState = &schema.StateSnapshot{
					Entities: initialMap,
				}
			}

			got, err := CalculateConstraintResidual(tt.constraint, scenario, entityMap)
			if err != nil {
				t.Fatalf("CalculateConstraintResidual failed: %v", err)
			}
			if math.Abs(got-tt.want) > tt.tolerance {
				t.Errorf("CalculateConstraintResidual() = %v, want %v (tolerance %v)", got, tt.want, tt.tolerance)
			}
		})
	}
}

func TestAnalyticalGradients(t *testing.T) {
	tests := []struct {
		name        string
		entities    []*schema.Entity
		constraints []*schema.Constraint
		initial     []*schema.Entity // Only needed for Fixed constraint
	}{
		{
			name:        "Coincidence Pt-Pt",
			entities:    []*schema.Entity{Point("p1", 1, 2), Point("p2", 3, 4)},
			constraints: []*schema.Constraint{Coincidence("c1", "p1", "p2")},
		},
		{
			name:        "Coincidence Pt-Ln",
			entities:    []*schema.Entity{Point("p1", 1, 2), Line("l1", 0, 0, 2, 0)},
			constraints: []*schema.Constraint{Coincidence("c1", "p1", "l1")},
		},
		{
			name:        "Coincidence Pt-Circ",
			entities:    []*schema.Entity{Point("p1", 2, 0), Circle("circ1", 0, 0, 1)},
			constraints: []*schema.Constraint{Coincidence("c1", "p1", "circ1")},
		},
		{
			name:        "Distance Pt-Pt",
			entities:    []*schema.Entity{Point("p1", 0, 0), Point("p2", 3, 4)},
			constraints: []*schema.Constraint{Distance("c1", "p1", "p2", 4)},
		},
		{
			name:        "Distance Pt-Ln",
			entities:    []*schema.Entity{Point("p1", 0, 3), Line("l1", 0, 0, 2, 0)},
			constraints: []*schema.Constraint{Distance("c1", "p1", "l1", 2)},
		},
		{
			name:        "Angle",
			entities:    []*schema.Entity{Line("l1", 0, 0, 1, 0), Line("l2", 0, 0, 1, 1)},
			constraints: []*schema.Constraint{Angle("c1", "l1", "l2", math.Pi/3)}, // Changed to non-matching angle to ensure gradient is tested
		},
		{
			name:        "Parallel",
			entities:    []*schema.Entity{Line("l1", 0, 0, 1, 0), Line("l2", 0, 0, 1, 1)},
			constraints: []*schema.Constraint{Parallel("c1", "l1", "l2")},
		},
		{
			name:        "Perpendicular",
			entities:    []*schema.Entity{Line("l1", 0, 0, 1, 0), Line("l2", 0, 0, 1, 1)},
			constraints: []*schema.Constraint{Perpendicular("c1", "l1", "l2")},
		},
		{
			name:        "Tangent Circ-Circ",
			entities:    []*schema.Entity{Circle("circ1", 0, 0, 1), Circle("circ2", 4, 0, 2)},
			constraints: []*schema.Constraint{Tangent("c1", "circ1", "circ2")},
		},
		{
			name:        "Tangent Circ-Ln",
			entities:    []*schema.Entity{Circle("circ1", 0, 2, 1), Line("l1", -2, 0, 2, 0)},
			constraints: []*schema.Constraint{Tangent("c1", "circ1", "l1")},
		},
		{
			name:        "Concentric Circ-Circ",
			entities:    []*schema.Entity{Circle("circ1", 1, 2, 3), Circle("circ2", 2, 2, 5)},
			constraints: []*schema.Constraint{Concentric("c1", "circ1", "circ2")},
		},
		{
			name:        "Concentric Pt-Circ",
			entities:    []*schema.Entity{Point("p1", 1, 3), Circle("circ1", 1, 2, 3)},
			constraints: []*schema.Constraint{Concentric("c1", "p1", "circ1")},
		},
		{
			name:        "Symmetric",
			entities:    []*schema.Entity{Point("p1", 1, 1), Point("p2", -2, 1), Line("l1", 0, -2, 0, 2)},
			constraints: []*schema.Constraint{Symmetric("c1", "p1", "p2", "l1")},
		},
		{
			name:        "Midpoint",
			entities:    []*schema.Entity{Point("p1", 1, 2), Line("l1", 0, 0, 2, 2)},
			constraints: []*schema.Constraint{Midpoint("c1", "p1", "l1")},
		},
		{
			name:        "Fixed Pt",
			entities:    []*schema.Entity{Point("p1", 2, 4)},
			initial:     []*schema.Entity{Point("p1", 1, 2)},
			constraints: []*schema.Constraint{Fixed("c1", "p1")},
		},
		{
			name:        "Fixed Ln",
			entities:    []*schema.Entity{Line("l1", 2, 4, 6, 8)},
			initial:     []*schema.Entity{Line("l1", 1, 2, 3, 4)},
			constraints: []*schema.Constraint{Fixed("c1", "l1")},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scenario := &schema.Sketch{
				Entities:    tt.entities,
				Constraints: tt.constraints,
			}
			if len(tt.initial) > 0 {
				initialMap := make(map[string]*schema.Entity)
				for _, ent := range tt.initial {
					initialMap[ent.Id] = ent
				}
				scenario.InitialState = &schema.StateSnapshot{
					Entities: initialMap,
				}
			}

			sys, err := NewConstraintSystem(scenario)
			if err != nil {
				t.Fatalf("NewConstraintSystem failed: %v", err)
			}
			x := sys.ExtractVariables()

			// Compute analytical gradient
			gradAnalytic := make([]float64, sys.NumVars())
			sys.ObjectiveGradient(gradAnalytic, x)

			// Compute numerical gradient
			gradNumeric := make([]float64, sys.NumVars())
			fd.Gradient(gradNumeric, sys.Objective, x, &fd.Settings{Formula: fd.Central})

			// Compare
			for i := range gradAnalytic {
				// Use a slightly relaxed tolerance of 1e-5 for complex constraints
				// like Symmetric or Angle which can have minor numerical precision differences
				if math.Abs(gradAnalytic[i]-gradNumeric[i]) > 1e-5 {
					t.Errorf("Gradient mismatch at index %d: analytic=%f, numeric=%f (diff=%f)",
						i, gradAnalytic[i], gradNumeric[i], math.Abs(gradAnalytic[i]-gradNumeric[i]))
				}
			}
		})
	}
}
