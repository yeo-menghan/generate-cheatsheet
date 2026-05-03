# Lecture 1: Intro


## 1. Optimization Problem (General Form)

$$\min_x f(x) \quad \text{s.t.} \quad x \in S \subseteq \mathbb{R}^n$$

- **Variable**: $x = (x_1, \ldots, x_n)^T$
- **Objective function**: $f : \mathbb{R}^n \to \mathbb{R}$
- **Feasible set**: $S$ — points satisfying all constraints
- **Optimal solution**: $x^* = \arg\min_{x \in S} f(x)$, where $f(x^*) \leq f(x)\ \forall x \in S$
- **Optimal value**: $f(x^*)$
- **max ↔ min equivalence**: $\max f(x) \equiv \min{-f(x)}$

### Constraint types (Constrained NLP)
$$S = \{x \in \mathbb{R}^n \mid g_i(x) = 0,\ i=1,\ldots,m;\ h_j(x) \leq 0,\ j=1,\ldots,p\}$$


## 2. Local vs Global Minimizers

| Term | Definition |
|---|---|
| **Local minimizer** | $\exists \epsilon > 0$ s.t. $f(x) \geq f(x^*)\ \forall x \in S \cap B_\epsilon(x^*)$ |
| **Strict local min** | $f(x) > f(x^*)\ \forall x \in S \cap B_\epsilon(x^*) \setminus \{x^*\}$ |
| **Global minimizer** | $f(x) \geq f(x^*)\ \forall x \in S$ |
| **Strict global min** | $f(x) > f(x^*)\ \forall x \in S \setminus \{x^*\}$ |

> Every global minimizer is a local minimizer. The converse is **not** generally true.


## 3. Gradient Vector

$$\nabla f(x) = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)^T$$

- $-\nabla f(x^*)$: direction of **steepest descent**
- $\nabla f(x^*)$: direction of **steepest ascent**
- **Key result**: $\nabla(b^T x) = b$

## 4. Hessian Matrix

$$H_f(x) = \begin{pmatrix} \frac{\partial^2 f}{\partial x_i \partial x_j} \end{pmatrix}_{n \times n}$$

- Symmetric when $f$ has continuous second-order derivatives.

**Example**: $f(x) = x_1^3 + 2x_1x_2 + x_2^2$

$$\nabla f = \begin{pmatrix} 3x_1^2 + 2x_2 \\ 2x_1 + 2x_2 \end{pmatrix}, \quad H_f = \begin{pmatrix} 6x_1 & 2 \\ 2 & 2 \end{pmatrix}$$

## 5. Matrix Definiteness

| Type | Condition |
|---|---|
| Positive semidefinite (PSD) | $x^T A x \geq 0\ \forall x$ — all eigenvalues $\lambda \geq 0$ |
| Positive definite (PD) | $x^T A x > 0\ \forall x \neq 0$ — all eigenvalues $\lambda > 0$ |
| Negative semidefinite | all eigenvalues $\lambda \leq 0$ |
| Negative definite | all eigenvalues $\lambda < 0$ |
| **Indefinite** | $\exists$ both positive and negative eigenvalues |

> PD $\Rightarrow$ PSD, but PSD $\not\Rightarrow$ PD.

## 6. Optimality Conditions (Unconstrained NLP)

A point $x^*$ is a **stationary point** if $\nabla f(x^*) = 0$.

### Necessary Conditions (if $x^*$ is a local min):
1. $\nabla f(x^*) = 0$
2. $H_f(x^*)$ is **positive semidefinite**

### Sufficient Conditions (to confirm $x^*$ is a local min):
1. $\nabla f(x^*) = 0$
2. $H_f(x^*)$ is **positive definite**

> If $H_f(x^*)$ is **not** PSD ⟹ $x^*$ is **not** a local minimizer.

## 7. Convex Sets & Functions

**Convex set**: $x, y \in D \Rightarrow \lambda x + (1-\lambda)y \in D\ \forall \lambda \in [0,1]$

**Convex function**: $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)\ \forall x,y \in D,\ \lambda \in [0,1]$

**Strictly convex**: strict inequality for distinct $x, y$, $\lambda \in (0,1)$

### Hessian Test for Convexity (Theorem 1.25)
| Hessian condition (at all $x \in D$) | Conclusion |
|---|---|
| $H_f(x)$ is PSD | $f$ is convex |
| $H_f(x)$ is PD | $f$ is **strictly** convex |
| $H_f(\hat{x})$ is indefinite at some $\hat{x}$ | $f$ is neither convex nor concave |

### Why convexity matters:
- Any **local minimizer** of a convex function is a **global minimizer**.
- If $f$ is **strictly convex**, the global minimizer is **unique**.

## 8. Quick Reference: Calculation Steps

**To find & verify a local minimizer (unconstrained)**:
1. Solve $\nabla f(x^*) = 0$ → find stationary points
2. Compute $H_f(x^*)$
3. Find eigenvalues of $H_f(x^*)$:
   - All $> 0$: local min ✓ (sufficient)
   - Any $< 0$: not a local min ✗ (necessary fails)

**To check convexity**:
1. Compute $H_f(x)$ generally
2. Check eigenvalues for all $x \in D$

**To find eigenvalues of $A$**:
1. Solve $\det(A - \lambda I) = 0$ (characteristic equation)
2. For $2\times2$: $\lambda^2 - \text{tr}(A)\lambda + \det(A) = 0$, so $\lambda = \frac{\text{tr}(A) \pm \sqrt{\text{tr}(A)^2 - 4\det(A)}}{2}$
3. **Shortcuts**: $\lambda_1 + \lambda_2 = \text{tr}(A)$, $\lambda_1 \lambda_2 = \det(A)$
   - PD iff $\text{tr}(A) > 0$ and $\det(A) > 0$
   - Indefinite iff $\det(A) < 0$

> **tr(A)** = sum of diagonal entries = $\lambda_1 + \lambda_2$

**Determinant formulas**:

- **2×2**: $\det\begin{pmatrix}a&b\\c&d\end{pmatrix} = ad - bc$

- **3×3** (cofactor expansion along row 1):
$$\det\begin{pmatrix}a&b&c\\d&e&f\\g&h&i\end{pmatrix} = a(ei-fh) - b(di-fg) + c(dh-eg)$$

## 9. Basic Matrix Rules

**Transpose**: $(AB)^T = B^T A^T$, $(A^T)^T = A$, $(A+B)^T = A^T + B^T$

**Inverse**: $(AB)^{-1} = B^{-1}A^{-1}$, $(A^T)^{-1} = (A^{-1})^T$, $AA^{-1} = I$

**Multiplication**: $A(B+C) = AB + AC$, generally $AB \neq BA$

**Determinant**: $\det(AB) = \det(A)\det(B)$, $\det(A^{-1}) = \frac{1}{\det(A)}$, $\det(A^T) = \det(A)$

**Rank**: $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$; $A$ invertible iff $\text{rank}(A) = n$ iff $\det(A) \neq 0$
