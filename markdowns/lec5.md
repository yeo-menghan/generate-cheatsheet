# Lecture 5: SVM

## 1. SVM Basics

**Setup:** Data $(x_i, y_i)$ where $x_i \in \mathbb{R}^p$, $y_i \in \{-1, 1\}$, classes assumed **linearly separable**.

**Classifier:** $f(x) = \text{sign}(\beta^T x + \beta_0)$

**Distance of point $x$ to hyperplane** $H = \{\beta^T x + \beta_0 = 0\}$:
$$d = \frac{|\beta^T x + \beta_0|}{\|\beta\|}$$

**Margin** (distance to closest point from hyperplane):
$$\gamma = \min_{i=1,\ldots,n} \frac{|\beta^T x_i + \beta_0|}{\|\beta\|}$$

## 2. SVM Primal Problem

$$\min_{\beta, \beta_0} \frac{1}{2}\|\beta\|^2 \quad \text{s.t.} \quad y_i(\beta^T x_i + \beta_0) \geq 1, \quad \forall i \in [n]$$

- Margin = $\frac{1}{\|\beta\|}$ at optimum
- **Support vectors**: points $x_i$ where $y_i(\beta^T x_i + \beta_0) = 1$ (tight constraint)

## 3. KKT Conditions (General)

**LICQ (Linear Independence Constraint Qualification):** At $x^*$, the gradients of all **active** constraints are linearly independent:
$$\{ \nabla g_i(x^*) \} \cup \{ \nabla h_j(x^*) : h_j(x^*) = 0 \} \text{ are linearly independent}$$
> LICQ $\Rightarrow$ $x^*$ is a **regular point** $\Rightarrow$ KKT conditions are **necessary** for local optimality.

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

**Solving the dual under $\lambda \geq 0$** (boundary search):
1. Find unconstrained optimum via $\nabla_\lambda g = 0$
2. If any $\lambda_i^* < 0$ (infeasible): optimum lies on the boundary — set $\lambda_i = 0$ and re-optimize the reduced problem
3. If the reduced optimum is still infeasible, $g$ is decreasing on $[0,\infty)$ so $\lambda_i^* = 0$
4. Repeat for each boundary case; take the feasible solution with highest $g(\lambda)$

## 4. SVM Dual Problem

**Lagrangian** (with $\alpha \in \mathbb{R}^n_+$):
$$L(\beta,\beta_0,\alpha) = \frac{1}{2}\|\beta\|^2 + \sum_{i=1}^n \alpha_i(1 - y_i(\beta^T x_i + \beta_0))$$

Setting $\frac{\partial L}{\partial \beta} = 0$ and $\frac{\partial L}{\partial \beta_0} = 0$ gives:
$$\beta^* = \sum_{i=1}^n \alpha_i y_i x_i \qquad \sum_{i=1}^n \alpha_i y_i = 0$$

Substituting $\beta^* = \sum_i \alpha_i y_i x_i$ back into $L$:
$$L = \frac{1}{2}\left\|\sum_i \alpha_i y_i x_i\right\|^2 + \sum_i \alpha_i - \sum_i \alpha_i y_i \left(\sum_j \alpha_j y_j x_j\right)^T x_i - \beta_0 \underbrace{\sum_i \alpha_i y_i}_{=0}$$
$$= \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j \langle x_i,x_j\rangle + \sum_i \alpha_i - \sum_{i,j}\alpha_i\alpha_j y_i y_j \langle x_i,x_j\rangle$$
$$= \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j \langle x_i,x_j\rangle$$

**Dual problem:**
$$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle \quad \text{s.t.} \quad \sum_{i=1}^n \alpha_i y_i = 0,\; \alpha_i \geq 0$$

**Recovering primal from dual:**
$$\beta^* = \sum_{i=1}^n \alpha_i^* y_i x_i \qquad \beta_0^* = y_k - \sum_{i=1}^n \alpha_i^* y_i \langle x_i, x_k\rangle \text{ for any } k \text{ with } \alpha_k^* > 0$$

**Complementary slackness:** $\alpha_i^* > 0 \Rightarrow y_i((\beta^*)^T x_i + \beta_0^*) = 1$ (i.e., $x_i$ is a support vector)

**Decision boundary** (test point $x$):
$$f(x) = \text{sign}\!\left(\sum_{i:\,\alpha_i^*>0} \alpha_i^* y_i \langle x_i, x\rangle + \beta_0^*\right)$$

## 5. Kernels

Replace $\langle x_i, x_j \rangle \to K(x_i, x_j) = \langle \phi(x_i), \phi(x_j)\rangle$ in the dual.

| Kernel | Formula |
|---|---|
| Linear | $K(a,b) = a^T b$ |
| Homogeneous polynomial degree $d$ | $K(a,b) = (a^T b)^d$ |
| Inhomogeneous polynomial degree $d$ | $K(a,b) = (a^T b + 1)^d$ |
| Gaussian (RBF) | $K(a,b) = \exp\!\left(-\frac{\|a-b\|^2}{2\sigma^2}\right)$ |

**Key point:** Computing $K(a,b)$ is $O(p)$; computing $\phi(a)$ explicitly can be $O(p^2)$ or higher.

## 6. Soft-Margin SVM

When classes are **not separable**, introduce slack $\varepsilon_i \geq 0$:

$$\min_{\beta,\beta_0,\varepsilon} \frac{1}{2}\|\beta\|^2 + C\sum_{i=1}^n \varepsilon_i \quad \text{s.t.} \quad y_i(\beta^T x_i + \beta_0) \geq 1 - \varepsilon_i,\; \varepsilon_i \geq 0$$

Equivalent unconstrained form (**hinge loss**):
$$\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 + C\sum_{i=1}^n \max\{1 - y_i(\beta^T x_i + \beta_0),\, 0\}$$

**Soft-margin dual:**
$$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i\alpha_j y_i y_j\langle x_i,x_j\rangle \quad \text{s.t.} \quad \sum_{i=1}^n \alpha_i y_i = 0,\; 0 \leq \alpha_i \leq C$$

> $C$ controls the margin–violation trade-off. Large $C$ = hard margin.

## 7. SVM vs. Logistic Regression

Both use $\frac{1}{2}\|\beta\|^2$ regularization; SVM uses hinge loss $\max\{1-z,0\}$ (sparse support vectors), logistic uses $\log(1+e^{-z})$ (smooth approximation to hinge, all points contribute).
