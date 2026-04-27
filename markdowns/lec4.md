# Lecture 4: Proximal Gradient Method

---

## 1. Norms

| Norm | Definition |
|------|-----------|
| ℓ₁ | ‖x‖₁ = Σᵢ \|xᵢ\| |
| ℓ₂ | ‖x‖₂ = √(Σᵢ xᵢ²) |
| ℓ∞ | ‖x‖∞ = maxᵢ \|xᵢ\| |
| ℓp | ‖x‖p = (Σᵢ \|xᵢ\|ᵖ)^(1/p) |
| Frobenius | ‖A‖²_F = ⟨A, A⟩ = Tr(AᵀA) |

**Inner product (matrices):** ⟨A, B⟩ = Tr(AᵀB) = Σᵢ Σⱼ Aᵢⱼ Bᵢⱼ

**Inner product (vectors):** ⟨x, y⟩ = xᵀy = Σᵢ xᵢ yᵢ

---

## 2. Projection onto Closed Convex Set C

```
ΠC(z) = argmin  ½‖x − z‖²
          x ∈ C
```

**Characterisation:**  x\* = ΠC(z)  ⟺  ⟨z − x\*, x − x\*⟩ ≤ 0  for all x ∈ C

| Set C | ΠC(z) |
|-------|--------|
| ℝⁿ₊ (positive orthant) | max{z, 0}  (elementwise) |
| ℓ₂-ball { ‖x‖₂ ≤ 1 } | z / max{‖z‖₂ , 1} |
| 𝕊ⁿ₊ (PSD cone), A = QΛQᵀ | Q · diag(max{λᵢ, 0}) · Qᵀ |

---

## 3. Normal Cone

```
NC(x̄) = { z  |  ⟨z, x − x̄⟩ ≤ 0  for all x ∈ C }
```

**Key equivalence:**  u ∈ NC(y)  ⟺  y = ΠC(y + u)

**Property:** If x̄ ∈ int(C), then NC(x̄) = {0}

**Example — C = [0, 1]:**

| x̄ | NC(x̄) |
|----|--------|
| 0 | (−∞, 0] |
| 1 | [0, +∞) |
| (0, 1) | {0} |
| x̄ ∉ C | ∅ |

**Example — C = { x ∈ ℝ² : ‖x‖ ≤ 1 }:**

| x̄ | NC(x̄) |
|----|--------|
| ‖x̄‖ = 1 | { λx̄ : λ ≥ 0 } |
| ‖x̄‖ < 1 | {0} |

---

## 4. Subdifferential

```
∂f(x) = { v  |  f(z) ≥ f(x) + ⟨v, z − x⟩  for all z }
```

- If f differentiable at x: **∂f(x) = {∇f(x)}**
- **Global optimality:** x̄ is a global minimiser  ⟺  **0 ∈ ∂f(x̄)**
- **Indicator function:** ∂δC(x) = NC(x)  for x ∈ C

**Example — f(x) = |x|:**

| x | ∂f(x) |
|---|--------|
| x < 0 | {−1} |
| x = 0 | [−1, 1] |
| x > 0 | {1} |

**Lasso sparsity condition** — at optimum β:

```
βᵢ < 0  ⟹  [Xᵀ(Xβ − Y)]ᵢ =  λ
βᵢ = 0  ⟺  |[Xᵀ(Xβ − Y)]ᵢ| ≤ λ   ← sparsity!
βᵢ > 0  ⟹  [Xᵀ(Xβ − Y)]ᵢ = −λ
```

---

## 5. Fenchel Conjugate

```
f*(y) = sup { ⟨y, x⟩ − f(x) }
         x
```

**f\* is always convex and closed** (even if f is not).

If f is closed proper convex, then (f\*)\* = f.

**Key triple equivalence:**

```
f(x) + f*(y) = ⟨x, y⟩  ⟺  y ∈ ∂f(x)  ⟺  x ∈ ∂f*(y)
```

**Examples:**

| f(x) | f\*(y) |
|------|--------|
| ‖x‖₁ | δC(y),  C = { y : ‖y‖∞ ≤ 1 } |
| δC(x) (indicator) | sup{ ⟨y, x⟩ : x ∈ C }  (support function) |

---

## 6. Moreau Envelope & Proximal Operator

```
Pf(x) = argmin { f(y) + ½‖y − x‖² }    ← proximal mapping
          y

Mf(x) =  min  { f(y) + ½‖y − x‖² }    ← Moreau envelope
          y
```

**Properties:**

- ∇Mf(x) = x − Pf(x)  &nbsp; (Mf is always differentiable)
- argmin f = argmin Mf
- Pδ\_C(x) = ΠC(x)

**Moreau Decomposition:**

```
x        = Pf(x) + Pf*(x)
½‖x‖²   = Mf(x) + Mf*(x)
```

**Soft Thresholding** — prox of f(x) = λ|x|:

```
Pf(x) = Sλ(x) = sign(x) · max{ |x| − λ, 0 }
```

Applied elementwise to x = [x₁; …; xₙ]:

```
[Sλ(x)]ᵢ = sign(xᵢ) · max{ |xᵢ| − λ, 0 }
```

**Huber function** (Moreau envelope of f(x) = λ|x|):

```
Mf(x) = ½x²           if |x| ≤ λ
         λ|x| − λ²/2   if |x| > λ
```

---

## 7. Proximal Gradient (PG) Method

**Problem:** min f(β) + g(β),  where f is smooth and g is convex non-smooth.

**Key insight** — gradient step on f only, then prox on g:

```
β^(k+1) = Pαg( β^(k) − α∇f(β^(k)) )
```

**Algorithm:**

```
choose β^(0),  step size α > 0
repeat:
    β^(k+1) = Pαg( β^(k) − α∇f(β^(k)) )
until convergence
```

**Convergence:** f(β^(k)) + g(β^(k)) − optimal ≤ O(1/k)

---

## 8. Accelerated Proximal Gradient (APG) Method

**Algorithm (FISTA-style):**

```
choose β^(0),  step size α > 0,  t₀ = t₁ = 1
repeat:
    β̄^(k)   = β^(k) + (tₖ − 1)/t_{k+1} · (β^(k) − β^(k−1))   ← momentum
    β^(k+1) = Pαg( β̄^(k) − α∇f(β̄^(k)) )
    t_{k+1} = ( 1 + √(1 + 4tₖ²) ) / 2
until convergence
```

**Convergence:** f(β^(k)) + g(β^(k)) − optimal ≤ O(1/k²)

**Step size rule:** α ∈ (0, 1/L),  where L = Lipschitz constant of ∇f

---

## 9. APG Applied to Lasso

```
min  ½‖Xβ − Y‖²  +  λ‖β‖₁
 β
```

- ∇f(β) = Xᵀ(Xβ − Y)
- Lipschitz constant: L = λ\_max(XᵀX)
- Step size: α = 1/L

**Iteration:**

```
β̄^(k)   = β^(k) + (tₖ − 1)/t_{k+1} · (β^(k) − β^(k−1))

β^(k+1) = S_{λ/L}( β̄^(k) − (1/L) · Xᵀ(Xβ̄^(k) − Y) )
```

**Optimality condition:**  β\* is optimal  ⟺  β\* = Pg(β\* − ∇f(β\*))

**Stopping criterion** (tolerance ε > 0):

```
‖ β^(k) − Sλ( β^(k) − Xᵀ(Xβ^(k) − Y) ) ‖ < ε
```

---

## 10. Complexity Summary

| Method | Convergence rate | Iterations to 10⁻⁴ error |
|--------|-----------------|--------------------------|
| PG | O(1/k) | ~O(10⁴) |
| APG | O(1/k²) | ~O(10²) |

- APG has the **same per-iteration cost** as PG (one prox + one gradient eval)
- **Restart trick:** rerun APG every 100–200 iterations from the latest iterate
