package lm

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack/lapack64"
	"gonum.org/v1/gonum/mat"
)

// solveQRAugmented solves the Levenberg-Marquardt step using an augmented matrix
// via direct LAPACK/BLAS calls with zero allocations.
func solveQRAugmented(
	J *mat.Dense,
	f *mat.VecDense,
	mu float64,
	aAug *mat.Dense,
	bAug *mat.VecDense,
	dx *mat.VecDense,
	tau []float64,
	work []float64,
) {
	m, n := J.Dims()
	sqrtMu := math.Sqrt(mu)

	aRaw := aAug.RawMatrix()
	jRaw := J.RawMatrix()

	// 1. Construct A_aug = [ J ; sqrt(μ)*I ] in-place without Slice() allocations
	for r := 0; r < m; r++ {
		copy(aRaw.Data[r*aRaw.Stride:r*aRaw.Stride+n], jRaw.Data[r*jRaw.Stride:r*jRaw.Stride+n])
	}
	for r := 0; r < n; r++ {
		rowStart := (m + r) * aRaw.Stride
		for c := 0; c < n; c++ {
			if r == c {
				aRaw.Data[rowStart+c] = sqrtMu
			} else {
				aRaw.Data[rowStart+c] = 0.0
			}
		}
	}

	// 2. Construct b_aug = [ -f ; 0 ]
	bRaw := bAug.RawVector()
	fRaw := f.RawVector()
	for i := 0; i < m; i++ {
		bRaw.Data[i*bRaw.Inc] = -fRaw.Data[i*fRaw.Inc]
	}
	for i := m; i < m+n; i++ {
		bRaw.Data[i*bRaw.Inc] = 0.0
	}

	// 3. Compute QR factorization of A_aug in-place
	lapack64.Geqrf(aRaw, tau, work, len(work))

	// 4. Apply Qᵀ to b_aug: b_aug = Qᵀ * b_aug
	bAugMat := blas64.General{
		Rows:   m + n,
		Cols:   1,
		Stride: bRaw.Inc,
		Data:   bRaw.Data,
	}
	lapack64.Ormqr(
		blas.Left,
		blas.Trans,
		aRaw,
		tau,
		bAugMat,
		work,
		len(work),
	)

	// 5. Solve the upper triangular system R * dx = Qᵀ * b_aug (first n elements)
	dxRaw := dx.RawVector()
	copy(dxRaw.Data, bRaw.Data[:n]) // Fast memmove copy

	// Correctly construct a blas64.Triangular view of the R block and call Trsv
	blas64.Trsv(
		blas.NoTrans,
		blas64.Triangular{
			Uplo:   blas.Upper,
			Diag:   blas.NonUnit,
			N:      n,
			Stride: aRaw.Stride,
			Data:   aRaw.Data,
		},
		dxRaw,
	)
}
