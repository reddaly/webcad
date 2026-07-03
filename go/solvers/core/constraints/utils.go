package constraints

import (
	"github.com/gonzojive/webcad/schema"
)

// getParams extracts the parameters of an entity as a flat float64 slice.
// This is a copy of core.GetParams to avoid circular dependencies.
func getParams(ent *schema.Entity) []float64 {
	if ent == nil {
		return nil
	}
	switch e := ent.GetEntityType().(type) {
	case *schema.Entity_Point:
		if e.Point == nil {
			return nil
		}
		return []float64{e.Point.X, e.Point.Y}
	case *schema.Entity_Line:
		if e.Line == nil {
			return nil
		}
		return []float64{e.Line.X1, e.Line.Y1, e.Line.X2, e.Line.Y2}
	case *schema.Entity_Circle:
		if e.Circle == nil {
			return nil
		}
		return []float64{e.Circle.Cx, e.Circle.Cy, e.Circle.R}
	case *schema.Entity_Arc:
		if e.Arc == nil {
			return nil
		}
		return []float64{e.Arc.Cx, e.Arc.Cy, e.Arc.R, e.Arc.StartAngle, e.Arc.EndAngle}
	case *schema.Entity_Ellipse:
		if e.Ellipse == nil {
			return nil
		}
		return []float64{e.Ellipse.Cx, e.Ellipse.Cy, e.Ellipse.Rx, e.Ellipse.Ry, e.Ellipse.Theta}
	case *schema.Entity_Spline:
		if e.Spline == nil {
			return nil
		}
		return e.Spline.ControlPoints
	}
	return nil
}

// Geometry type helpers (copied from core to avoid cycles)

func isCircleOrArc(ent *schema.Entity) bool {
	switch ent.GetEntityType().(type) {
	case *schema.Entity_Circle, *schema.Entity_Arc:
		return true
	}
	return false
}

func isLine(ent *schema.Entity) bool {
	_, ok := ent.GetEntityType().(*schema.Entity_Line)
	return ok
}

func isPoint(ent *schema.Entity) bool {
	_, ok := ent.GetEntityType().(*schema.Entity_Point)
	return ok
}

func isPointOrCenter(ent *schema.Entity) bool {
	switch ent.GetEntityType().(type) {
	case *schema.Entity_Point, *schema.Entity_Circle, *schema.Entity_Arc, *schema.Entity_Ellipse:
		return true
	}
	return false
}
