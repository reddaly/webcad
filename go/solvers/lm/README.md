# Levenberg-Marquardt (LM) Solver

A high-performance, native Go implementation of the Levenberg-Marquardt (LM) optimization algorithm, optimized specifically for solving 2D geometric constraint systems in CAD applications.

This package achieves **zero heap allocations** in its hot optimization loop by reusing memory workspaces via a `sync.Pool` and calling raw BLAS/LAPACK bindings directly on slice-backed matrices.

---

## 📐 Mathematical Formulation

Geometric Constraint Solving (GCS) involves finding a state $x \in \mathbb{R}^n$ (coordinates of points, lines, circles) that satisfies a system of $m$ non-linear constraint residuals:

$$f(x) = 0, \quad f : \mathbb{R}^n \to \mathbb{R}^m$$

The Levenberg-Marquardt algorithm solves this by computing a step $\Delta x$ at each iteration. Depending on the system dimensions, this package automatically selects between two formulations:

### 1. Primal Space Formulation (Standard LM)
When $m \ge n$ (more or equal constraints than variables), we solve the damped normal equations:

$$(J^T J + \mu I) \Delta x = -J^T f(x)$$

Where $J \in \mathbb{R}^{m \times n}$ is the Jacobian of $f(x)$, and $\mu > 0$ is the damping factor.

### 2. Dual Space Formulation
When $m < n$ (under-constrained systems), solving in primal space can be slow and computationally redundant. We instead solve in the dual (residual) space:

$$(J J^T + \mu I) z = -f(x)$$

$$\Delta x = J^T z$$

This reduces the linear system size from $n \times n$ to $m \times m$, offering significant speedups for under-constrained sketches.

---

## ⚡ High-Performance Zero-Allocation Design

In CAD systems, constraint solving runs inside interactive loops (e.g., dragging entities). Garbage collection (GC) pauses can ruin the frame rate. This package guarantees **0 heap allocations** per solve iteration using:

### 1. Pre-allocated Solver Workspaces
A `SolverWorkspace` struct holds all intermediate vectors, matrices, and LAPACK working slices. 
*   Workspaces are managed by a `sync.Pool`.
*   During a solve, the workspace is retrieved and reset.
*   If the system dimensions ($m, n$) match the cached workspace, it uses the existing slices. Slices are only re-allocated if the problem size changes.

### 2. Direct BLAS & LAPACK Callbacks
Instead of using Gonum's standard matrix wrapper APIs (which allocate intermediate matrices on the heap for operations like multiplication and solving), this solver operates directly on raw flat slices via Gonum's `blas64` and `lapack64` bindings.

#### LAPACK Functions Used (`lapack64`)
*   **`Potrf`**: Computes the Cholesky factorization of a real symmetric positive definite matrix ($A = U^T U$ or $A = L L^T$).
*   **`Potrs`**: Solves a system of linear equations $A X = B$ using the Cholesky factorization computed by `Potrf`.
*   **`Geqrf`**: Computes a QR factorization of a real $M \times N$ matrix $A$ in-place.
*   **`Ormqr`**: Overwrites a general matrix $C$ with $Q^T C$, where $Q$ is the orthogonal matrix represented as a product of elementary reflectors returned by `Geqrf`.

#### BLAS Functions Used (`blas64`)
*   **`Dot`**: Computes the dot product of two vectors ($x^T y$).
*   **`Gemv`**: Computes matrix-vector multiplication ($y = \alpha A x + \beta y$ or $y = \alpha A^T x + \beta y$).
*   **`Syrk`**: Computes a symmetric rank-k update ($C = \alpha A A^T + \beta C$ or $C = \alpha A^T A + \beta C$).
*   **`Trsv`**: Solves a triangular system of equations ($R x = b$ or $R^T x = b$).

#### BLAS Types & Constants Used
*   **`blas64.General`**: Struct defining a general matrix in row-major layout with dimensions (`Rows`, `Cols`), stride, and backing data slice.
*   **`blas64.Triangular`**: Struct defining a triangular matrix view.
*   **`blas.Transpose` (`Trans`, `NoTrans`)**: Constants specifying transpose configuration for matrix operations.
*   **`blas.ULocation` (`Upper`)**: Constant specifying upper triangular storage.
*   **`blas.Diag` (`NonUnit`)**: Constant specifying non-unit diagonal status.
*   **`blas.Side` (`Left`)**: Constant specifying left-side matrix operations.

### 3. Allocation-Free Linear Solvers
The step equations are solved in-place:
1.  **Cholesky Decomposition (`Potrf` / `Potrs`)**: The solver first attempts to solve the damped system using symmetric positive-definite Cholesky factorization.
2.  **QR Decomposition Fallback (`Geqrf` / `Ormqr` / `Trsv`)**: If the Cholesky solver fails (e.g., due to numerical instability or near-singular matrices), the solver falls back to a direct QR solver on the augmented matrix:
    $$A_{\text{aug}} = \begin{bmatrix} J \\ \sqrt{\mu} I \end{bmatrix}, \quad b_{\text{aug}} = \begin{bmatrix} -f(x) \\ 0 \end{bmatrix}$$
    Both solver paths execute without a single heap allocation.

---

## 📋 Solver Status Codes

*   **`Success`**: The system successfully converged to a solution where the infinity norm of residuals is below `EpGeom` ($\|f(x)\|_\infty < \epsilon_{\text{geom}}$).
*   **`Inconsistent`**: The solver converged to a local minimum where the gradient of the objective is zero ($\|J^T f(x)\|_\infty < \epsilon_{\text{grad}}$) but the residuals remain high. This indicates an inconsistent over-constrained sketch.
*   **`Stalled`**: The step size $\Delta x$ fell below `EpStep`, meaning no further progress can be made.
*   **`MaxIterationsExceeded`**: The solver reached the user-configured iteration limit without converging.

---

## 🚀 Usage Example

```go
import (
	"github.com/gonzojive/webcad/go/solvers/core"
	"github.com/gonzojive/webcad/go/solvers/lm"
)

// Initialize the solver
solver := lm.New()
solver.EpGeom = 1e-8
solver.MaxIter = 100

// Run a cold solve trial against a protobuf sketch
result, err := solver.SolveCold(sketch)
if err != nil {
	log.Fatalf("Solving failed: %v", err)
}

if result.Success {
	fmt.Printf("Solved in %d iterations (residual: %e)\n", 
		result.Telemetry.Iterations, 
		result.Telemetry.FinalResidual)
} else {
	fmt.Printf("Solver failed: %s\n", result.ErrorMessage)
}
```
