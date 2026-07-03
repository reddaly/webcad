package constraints

import (
	"fmt"
	"github.com/gonzojive/webcad/schema"
)

// NewEvaluator is a factory function that creates a new Evaluator for a given constraint.
// It analyzes the initial state of the involved entities to lock chirality or precompute constants.
func NewEvaluator(c *schema.Constraint, entities map[string]*schema.Entity) (Evaluator, error) {
	switch tc := c.GetConstraintType().(type) {
	case *schema.Constraint_Fixed:
		return NewFixedEvaluator(c, entities)
	case *schema.Constraint_Concentric:
		return NewConcentricEvaluator(c, entities)
	case *schema.Constraint_Midpoint:
		return NewMidpointEvaluator(c, entities)
	case *schema.Constraint_Parallel:
		return NewParallelEvaluator(c, entities)
	case *schema.Constraint_Perpendicular:
		return NewPerpendicularEvaluator(c, entities)
	case *schema.Constraint_Coincidence:
		return NewCoincidenceEvaluator(c, entities)
	case *schema.Constraint_Distance:
		return NewDistanceEvaluator(c, entities)
	case *schema.Constraint_Angle:
		return NewAngleEvaluator(c, entities)
	case *schema.Constraint_Tangent:
		return NewTangentEvaluator(c, entities)
	case *schema.Constraint_Symmetric:
		return NewSymmetricEvaluator(c, entities)
	default:
		// Return a placeholder evaluator for unregistered types to allow incremental migration.
		return &PlaceholderEvaluator{ConstraintType: fmt.Sprintf("%T", tc)}, nil
	}
}

// PlaceholderEvaluator silently ignores evaluation for unimplemented constraints.
type PlaceholderEvaluator struct {
	ConstraintType string
}

func (p *PlaceholderEvaluator) Evaluate(x []float64, grad []float64, paramIndices map[string]int) float64 {
	return 0.0
}
