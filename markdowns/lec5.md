# Lecture 5: SVM

---

## 1. SVM Basics

**Setup:** Data $(x_i, y_i)$ where $x_i \in \mathbb{R}^p$, $y_i \in \{-1, 1\}$, classes assumed **linearly separable**.

**Classifier:** $f(x) = \text{sign}(\beta^T x + \beta_0)$

**Distance of point $x$ to hyperplane** $H = \{\beta^T x + \beta_0 = 0\}$:
$$d = \frac{|\beta^T x + \beta_0|}{\|\beta\|}$$

**Margin** (distance to closest point from hyperplane):
$$\gamma = \min_{i=1,\ldots,n} \frac{|\beta^T x_i + \beta_0|}{\|\beta\|}$$

---

## 2. SVM Primal Problem

$$\min_{\beta, \beta_0} \frac{1}{2}\|\beta\|^2 \quad \text{s.t.} \quad y_i(\beta^T x_i + \beta_0) \geq 1, \quad \forall i \in [n]$$

- Margin = $\frac{1}{\|\beta\|}$ at optimum
- **Support vectors**: points $x_i$ where $y_i(\beta^T x_i + \beta_0) = 1$ (tight constraint)

---

## 3. KKT Conditions (General)

For primal $\min f(x)$ s.t. $g_i(x)=0$, $h_j(x) \leq 0$, the **KKT conditions** at a regular point $x^*$ are:

| Condition | Equation |
|---|---|
| Stationarity | $\nabla f(x^*) + \sum v_i^* \nabla g_i(x^*) + \sum u_j^* \nabla h_j(x^*) = 0$ |
| Primal feasibility | $g_i(x^*)=0$, $h_j(x^*) \leq 0$ |
| Dual feasibility | $u_j^* \geq 0$ |
| Complementary slackness | $u_j^* h_j(x^*) = 0$ |

**Lagrangian:** $L(x,v,u) = f(x) + \sum v_i g_i(x) + \sum u_j h_j(x)$

**Dual function** (always concave): $\theta(v,u) = \min_x L(x,v,u)$

**Weak duality:** $\theta(v,u) \leq f(x)$ for any dual-feasible $(v,u)$ and primal-feasible $x$

**Strong duality** holds under: convex $f, h_j$; affine $g_i$; and **Slater's condition** ($\exists \hat{x}$: $g_i(\hat{x})=0$, $h_j(\hat{x}) < 0$).

---

## 4. SVM Dual Problem

**Lagrangian** (with $\alpha \in \mathbb{R}^n_+$):
$$L(\beta,\beta_0,\alpha) = \frac{1}{2}\|\beta\|^2 + \sum_{i=1}^n \alpha_i(1 - y_i(\beta^T x_i + \beta_0))$$

Setting $\frac{\partial L}{\partial \beta} = 0$ and $\frac{\partial L}{\partial \beta_0} = 0$ gives:
$$\beta^* = \sum_{i=1}^n \alpha_i y_i x_i \qquad \sum_{i=1}^n \alpha_i y_i = 0$$

**Dual problem:**
$$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle \quad \text{s.t.} \quad \sum_{i=1}^n \alpha_i y_i = 0,\; \alpha_i \geq 0$$

**Recovering primal from dual:**
$$\beta^* = \sum_{i=1}^n \alpha_i^* y_i x_i \qquad \beta_0^* = y_k - \sum_{i=1}^n \alpha_i^* y_i \langle x_i, x_k\rangle \text{ for any } k \text{ with } \alpha_k^* > 0$$

**Complementary slackness:** $\alpha_i^* > 0 \Rightarrow y_i((\beta^*)^T x_i + \beta_0^*) = 1$ (i.e., $x_i$ is a support vector)

**Decision boundary** (test point $x$):
$$f(x) = \text{sign}\!\left(\sum_{i:\,\alpha_i^*>0} \alpha_i^* y_i \langle x_i, x\rangle + \beta_0^*\right)$$

---

## 5. Kernels

Replace $\langle x_i, x_j \rangle \to K(x_i, x_j) = \langle \phi(x_i), \phi(x_j)\rangle$ in the dual.

| Kernel | Formula |
|---|---|
| Linear | $K(a,b) = a^T b$ |
| Homogeneous polynomial degree $d$ | $K(a,b) = (a^T b)^d$ |
| Inhomogeneous polynomial degree $d$ | $K(a,b) = (a^T b + 1)^d$ |
| Gaussian (RBF) | $K(a,b) = \exp\!\left(-\frac{\|a-b\|^2}{2\sigma^2}\right)$ |

**Key point:** Computing $K(a,b)$ is $O(p)$; computing $\phi(a)$ explicitly can be $O(p^2)$ or higher.

---

## 6. Soft-Margin SVM

When classes are **not separable**, introduce slack $\varepsilon_i \geq 0$:

$$\min_{\beta,\beta_0,\varepsilon} \frac{1}{2}\|\beta\|^2 + C\sum_{i=1}^n \varepsilon_i \quad \text{s.t.} \quad y_i(\beta^T x_i + \beta_0) \geq 1 - \varepsilon_i,\; \varepsilon_i \geq 0$$

Equivalent unconstrained form (**hinge loss**):
$$\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 + C\sum_{i=1}^n \max\{1 - y_i(\beta^T x_i + \beta_0),\, 0\}$$

**Soft-margin dual:**
$$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i\alpha_j y_i y_j\langle x_i,x_j\rangle \quad \text{s.t.} \quad \sum_{i=1}^n \alpha_i y_i = 0,\; 0 \leq \alpha_i \leq C$$

> $C$ controls the margin–violation trade-off. Large $C$ = hard margin.

---

## 7. SVM vs. Logistic Regression

| | SVM (soft) | Logistic Regression |
|---|---|---|
| Loss | $\max\{1-z, 0\}$ (hinge) | $\log(1+e^{-z})$ (logistic) |
| Regularization | $\frac{1}{2}\|\beta\|^2$ | $\lambda\|\beta\|^2$ |
| Target | $z = y_i(\beta^T x_i + \beta_0) \geq 1$ | $z \gg 0$ |

Logistic loss is a smooth approximation to hinge loss.

---

## Quick Reference: Deriving a Dual

1. Write the **Lagrangian** $L$ with multipliers ($\geq 0$ for $\leq 0$ constraints)
2. Set $\nabla_x L = 0$ to find $x^*$ in terms of multipliers
3. Substitute back to get **dual function** $\theta$
4. Maximize $\theta$ subject to multiplier constraints → **dual problem**
