package bfgs
 
import (
	"math"
	"testing"
 
	"github.com/gonzojive/webcad/schema"
)
 
func TestBFGSSolver(t *testing.T) {

	tests := []struct {
		name     string
		scenario *schema.Sketch
		assert   func(*testing.T, *schema.SolveResult)
	}{
		{
			name: "BasicSanity",
			scenario: &schema.Sketch{
				Id: "basic_sanity",
				Entities: []*schema.Entity{
					{
						Id: "p1",
						EntityType: &schema.Entity_Point{
							Point: &schema.PointEntity{X: 0, Y: 0},
						},
					},
					{
						Id: "p2",
						EntityType: &schema.Entity_Point{
							Point: &schema.PointEntity{X: 10, Y: 0},
						},
					},
				},
				Constraints: []*schema.Constraint{
					{
						Id: "c1",
						ConstraintType: &schema.Constraint_Distance{
							Distance: &schema.DistanceConstraint{
								EntityA: "p1",
								EntityB: "p2",
								Value:   20,
							},
						},
					},
				},
			},
			assert: func(t *testing.T, res *schema.SolveResult) {
				p1Solved := res.SolvedState.Entities["p1"].GetPoint()
				p2Solved := res.SolvedState.Entities["p2"].GetPoint()
				dist := math.Sqrt(math.Pow(p2Solved.X-p1Solved.X, 2) + math.Pow(p2Solved.Y-p1Solved.Y, 2))
				if math.Abs(dist-20) > 1e-4 {
					t.Errorf("Distance is incorrect! Got %f, expected 20", dist)
				}
			},
		},
		{
			name: "FixedPoint_Distance",
			scenario: &schema.Sketch{
				Id: "fixed_point_dist",
				Entities: []*schema.Entity{
					{
						Id: "p1",
						EntityType: &schema.Entity_Point{
							Point: &schema.PointEntity{X: 0, Y: 0},
						},
					},
					{
						Id: "p2",
						EntityType: &schema.Entity_Point{
							Point: &schema.PointEntity{X: 10, Y: 0},
						},
					},
				},
				Constraints: []*schema.Constraint{
					{
						Id: "c_fixed",
						ConstraintType: &schema.Constraint_Fixed{
							Fixed: &schema.FixedConstraint{
								EntityId: "p1",
							},
						},
					},
					{
						Id: "c_dist",
						ConstraintType: &schema.Constraint_Distance{
							Distance: &schema.DistanceConstraint{
								EntityA: "p1",
								EntityB: "p2",
								Value:   20,
							},
						},
					},
				},
			},
			assert: func(t *testing.T, res *schema.SolveResult) {
				p1Solved := res.SolvedState.Entities["p1"].GetPoint()
				p2Solved := res.SolvedState.Entities["p2"].GetPoint()
				if math.Abs(p1Solved.X) > 1e-4 || math.Abs(p1Solved.Y) > 1e-4 {
					t.Errorf("Fixed point P1 moved! Got (%f, %f), expected (0,0)", p1Solved.X, p1Solved.Y)
				}
				dist := math.Sqrt(math.Pow(p2Solved.X-p1Solved.X, 2) + math.Pow(p2Solved.Y-p1Solved.Y, 2))
				if math.Abs(dist-20) > 1e-4 {
					t.Errorf("Distance is incorrect! Got %f, expected 20", dist)
				}
			},
		},
		{
			name: "FixedLine_Distance",
			scenario: &schema.Sketch{
				Id: "fixed_line_dist",
				Entities: []*schema.Entity{
					{
						Id: "l1",
						EntityType: &schema.Entity_Line{
							Line: &schema.LineEntity{X1: 0, Y1: 0, X2: 10, Y2: 0},
						},
					},
					{
						Id: "p2",
						EntityType: &schema.Entity_Point{
							Point: &schema.PointEntity{X: 5, Y: 5},
						},
					},
				},
				Constraints: []*schema.Constraint{
					{
						Id: "c_fixed",
						ConstraintType: &schema.Constraint_Fixed{
							Fixed: &schema.FixedConstraint{
								EntityId: "l1",
							},
						},
					},
					{
						Id: "c_dist",
						ConstraintType: &schema.Constraint_Distance{
							Distance: &schema.DistanceConstraint{
								EntityA: "l1",
								EntityB: "p2",
								Value:   10,
							},
						},
					},
				},
			},
			assert: func(t *testing.T, res *schema.SolveResult) {
				l1Solved := res.SolvedState.Entities["l1"].GetLine()
				p2Solved := res.SolvedState.Entities["p2"].GetPoint()
				if math.Abs(l1Solved.X1) > 1e-4 || math.Abs(l1Solved.Y1) > 1e-4 ||
					math.Abs(l1Solved.X2-10) > 1e-4 || math.Abs(l1Solved.Y2) > 1e-4 {
					t.Errorf("Fixed line L1 moved! Got (%f,%f) -> (%f,%f), expected (0,0) -> (10,0)",
						l1Solved.X1, l1Solved.Y1, l1Solved.X2, l1Solved.Y2)
				}
				if math.Abs(math.Abs(p2Solved.Y)-10) > 1e-4 {
					t.Errorf("Distance to line is incorrect! P2.Y = %f, expected 10 or -10", p2Solved.Y)
				}
			},
		},
		{
			name: "ConcentricCircles",
			scenario: &schema.Sketch{
				Id: "concentric_circles",
				Entities: []*schema.Entity{
					{
						Id: "p1",
						EntityType: &schema.Entity_Point{
							Point: &schema.PointEntity{X: 0, Y: 0},
						},
					},
					{
						Id: "c1",
						EntityType: &schema.Entity_Circle{
							Circle: &schema.CircleEntity{Cx: 5, Cy: 5, R: 3},
						},
					},
					{
						Id: "c2",
						EntityType: &schema.Entity_Circle{
							Circle: &schema.CircleEntity{Cx: 10, Cy: 10, R: 5},
						},
					},
				},
				Constraints: []*schema.Constraint{
					{
						Id: "c_fixed",
						ConstraintType: &schema.Constraint_Fixed{
							Fixed: &schema.FixedConstraint{
								EntityId: "p1",
							},
						},
					},
					{
						Id: "c_concentric_p1_c1",
						ConstraintType: &schema.Constraint_Concentric{
							Concentric: &schema.ConcentricConstraint{
								EntityA: "p1",
								EntityB: "c1",
							},
						},
					},
					{
						Id: "c_concentric_c1_c2",
						ConstraintType: &schema.Constraint_Concentric{
							Concentric: &schema.ConcentricConstraint{
								EntityA: "c1",
								EntityB: "c2",
							},
						},
					},
				},
			},
			assert: func(t *testing.T, res *schema.SolveResult) {
				c1Solved := res.SolvedState.Entities["c1"].GetCircle()
				c2Solved := res.SolvedState.Entities["c2"].GetCircle()
				if math.Abs(c1Solved.Cx) > 1e-4 || math.Abs(c1Solved.Cy) > 1e-4 {
					t.Errorf("Circle c1 center is incorrect! Got (%f, %f), expected (0,0)", c1Solved.Cx, c1Solved.Cy)
				}
				if math.Abs(c2Solved.Cx) > 1e-4 || math.Abs(c2Solved.Cy) > 1e-4 {
					t.Errorf("Circle c2 center is incorrect! Got (%f, %f), expected (0,0)", c2Solved.Cx, c2Solved.Cy)
				}
			},
		},
		{
			name: "TangentCircles",
			scenario: &schema.Sketch{
				Id: "tangent_circles",
				Entities: []*schema.Entity{
					{
						Id: "c1",
						EntityType: &schema.Entity_Circle{
							Circle: &schema.CircleEntity{Cx: 0, Cy: 0, R: 5},
						},
					},
					{
						Id: "c2",
						EntityType: &schema.Entity_Circle{
							Circle: &schema.CircleEntity{Cx: 10, Cy: 0, R: 5},
						},
					},
				},
				Constraints: []*schema.Constraint{
					{
						Id: "c_t",
						ConstraintType: &schema.Constraint_Tangent{
							Tangent: &schema.TangentConstraint{
								EntityA: "c1",
								EntityB: "c2",
							},
						},
					},
				},
			},
			assert: func(t *testing.T, res *schema.SolveResult) {
				// The initial state is already tangent, so it should remain tangent.
				// We can assert that the distance between centers is equal to sum of radiuses.
				c1Solved := res.SolvedState.Entities["c1"].GetCircle()
				c2Solved := res.SolvedState.Entities["c2"].GetCircle()
				dist := math.Sqrt(math.Pow(c2Solved.Cx-c1Solved.Cx, 2) + math.Pow(c2Solved.Cy-c1Solved.Cy, 2))
				sumR := c1Solved.R + c2Solved.R
				if math.Abs(dist-sumR) > 1e-4 {
					t.Errorf("Circles are not tangent! Distance=%f, sum of radiuses=%f", dist, sumR)
				}
			},
		},
	}

	solvers := []*Solver{
		New(),
		NewNumerical(),
	}

	for _, solver := range solvers {
		t.Run(string(solver.ID()), func(t *testing.T) {
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					res, err := solver.SolveCold(tt.scenario)
					if err != nil {
						t.Fatalf("SolveCold failed: %v", err)
					}
					if !res.Success {
						t.Fatalf("SolveCold did not succeed: %s", res.ErrorMessage)
					}
					tt.assert(t, res)
				})
			}
		})
	}
}
