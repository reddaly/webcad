// Package core defines the primary interfaces and types used across the webcad
// solver library.
package core

import (
	"os"

	"github.com/gonzojive/webcad/schema"
)

// FindBinary attempts to locate a binary at various candidate paths.
// It returns the first path that exists, or the default path (first candidate) if none are found.
func FindBinary(candidates ...string) string {
	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	if len(candidates) > 0 {
		return candidates[0]
	}
	return ""
}

// GetParams extracts the parameters of an entity as a flat float64 slice.
// Returns nil if the entity or its specific type struct is nil.
func GetParams(ent *schema.Entity) []float64 {
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

// SetParams updates the parameters of an entity from a flat float64 slice.
// It assumes the entity type is already set and matches the slice length.
func SetParams(ent *schema.Entity, params []float64) {
	if ent == nil || params == nil {
		return
	}
	switch e := ent.GetEntityType().(type) {
	case *schema.Entity_Point:
		if e.Point != nil && len(params) >= 2 {
			e.Point.X = params[0]
			e.Point.Y = params[1]
		}
	case *schema.Entity_Line:
		if e.Line != nil && len(params) >= 4 {
			e.Line.X1 = params[0]
			e.Line.Y1 = params[1]
			e.Line.X2 = params[2]
			e.Line.Y2 = params[3]
		}
	case *schema.Entity_Circle:
		if e.Circle != nil && len(params) >= 3 {
			e.Circle.Cx = params[0]
			e.Circle.Cy = params[1]
			e.Circle.R = params[2]
		}
	case *schema.Entity_Arc:
		if e.Arc != nil && len(params) >= 5 {
			e.Arc.Cx = params[0]
			e.Arc.Cy = params[1]
			e.Arc.R = params[2]
			e.Arc.StartAngle = params[3]
			e.Arc.EndAngle = params[4]
		}
	case *schema.Entity_Ellipse:
		if e.Ellipse != nil && len(params) >= 5 {
			e.Ellipse.Cx = params[0]
			e.Ellipse.Cy = params[1]
			e.Ellipse.Rx = params[2]
			e.Ellipse.Ry = params[3]
			e.Ellipse.Theta = params[4]
		}
	case *schema.Entity_Spline:
		if e.Spline != nil {
			e.Spline.ControlPoints = params
		}
	}
}
