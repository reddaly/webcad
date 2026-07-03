package lm

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack/lapack64"
	"gonum.org/v1/gonum/mat"
)

type SolverStatus int

const (
	Success SolverStatus = iota
	Inconsistent
	Stalled
	MaxIterationsExceeded
)

func (s SolverStatus) String() string {
	switch s {
	case Success:
		return "Success"
	case Inconsistent:
		return "Inconsistent (Over-constrained)"
	case Stalled:
		return "Stalled (No progress possible)"
	case MaxIterationsExceeded:
		return "Max Iterations Exceeded"
	default:
		return "Unknown"
	}
}

// SolverWorkspace holds all pre-allocated matrices, vectors, and solvers to avoid heap allocations.
type SolverWorkspace struct {
	J          *mat.Dense
	F          *mat.VecDense
	FNew       *mat.VecDense
	G          *mat.VecDense
	Dx         *mat.VecDense
	Hres       *mat.SymDense
	HresDamped *mat.SymDense
	Jjt        *mat.SymDense
	JjtDamped  *mat.SymDense
	Z          *mat.VecDense
	AAug       *mat.Dense
	BAug       *mat.VecDense
	Tau        []float64
	Work       []float64
	X          []float64 // Holds the active accepted state (prevents memory aliasing)
	XNew       []float64 // Holds candidate state
}

func (w *SolverWorkspace) Reset(m, n int) {
	if w.J == nil || w.J.RawMatrix().Rows != m || w.J.RawMatrix().Cols != n {
		w.J = mat.NewDense(m, n, nil)
		w.F = mat.NewVecDense(m, nil)
		w.FNew = mat.NewVecDense(m, nil)
		w.G = mat.NewVecDense(n, nil)
		w.Dx = mat.NewVecDense(n, nil)
		w.Hres = mat.NewSymDense(n, nil)
		w.HresDamped = mat.NewSymDense(n, nil)
		w.Jjt = mat.NewSymDense(m, nil)
		w.JjtDamped = mat.NewSymDense(m, nil)
		w.Z = mat.NewVecDense(m, nil)
		w.AAug = mat.NewDense(m+n, n, nil)
		w.BAug = mat.NewVecDense(m+n, nil)
		w.Tau = make([]float64, n)
		w.X = make([]float64, n)
		w.XNew = make([]float64, n)

		// Query optimal work size for Geqrf
		workQuery := []float64{0}
		lapack64.Geqrf(w.AAug.RawMatrix(), w.Tau, workQuery, -1)
		lwork := int(workQuery[0])

		// Query optimal work size for Ormqr
		bGeneral := blas64.General{
			Rows:   m + n,
			Cols:   1,
			Stride: 1,
			Data:   w.BAug.RawVector().Data,
		}
		lapack64.Ormqr(
			blas.Left,
			blas.Trans,
			w.AAug.RawMatrix(),
			w.Tau,
			bGeneral,
			workQuery,
			-1,
		)
		if int(workQuery[0]) > lwork {
			lwork = int(workQuery[0])
		}
		w.Work = make([]float64, lwork)
	} else {
		w.J.Zero()
		w.F.Zero()
		w.FNew.Zero()
		w.G.Zero()
		w.Dx.Zero()
		w.Hres.Zero()
		w.HresDamped.Zero()
		w.Jjt.Zero()
		w.JjtDamped.Zero()
		w.Z.Zero()
		w.AAug.Zero()
		w.BAug.Zero()
	}
}
