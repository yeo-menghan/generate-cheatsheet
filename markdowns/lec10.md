# Lecture 10: Gaussian Graphical Models

## 1. Conditional Independence

**Definition:** $X \perp\!\!\!\perp Y \mid Z$ if $p(x, y \mid z) = p(x \mid z)\,p(y \mid z)$

**Factorization:** If $X \perp\!\!\!\perp Y \mid Z$, then $p(x,y,z) = g(x,z)\,h(y,z)$

## 2. Undirected Graph

- Graph $G = (V, E)$, vertex set $V = \{1,\ldots,p\}$, edge set $E \subset V \times V$
- Undirected: $(s,t) \in E \iff (t,s) \in E$; no self-loops

**Markov Property:** $x$ is Markov w.r.t. $G$ if for every cut set $S$ separating $A$ and $B$:
$$x_A \perp\!\!\!\perp x_B \mid x_S$$

## 3. Multivariate Gaussian

$$f(x) = (2\pi)^{-p/2}\det(\Theta)^{1/2} \exp\!\left(-\tfrac{1}{2}x^T\Theta x\right), \quad x \sim \mathcal{N}(0,\Sigma)$$

- **Precision matrix:** $\Theta = \Sigma^{-1} \in \mathbb{S}^p_{++}$
- $\det(\Sigma) = \prod_j \lambda_j(\Sigma)$, $\quad \det(\Theta) = 1/\det(\Sigma)$

## 4. Gaussian Graphical Model (GGM)

**Key fact:** For $x \sim \mathcal{N}(0,\Sigma)$:
$$x_s \perp\!\!\!\perp x_t \mid x_{V\setminus\{s,t\}} \iff \Theta_{st} = 0$$

**Edge set:** $E = \{(s,t) \mid \Theta_{st} \neq 0,\ s < t\}$

> Zero off-diagonal entry in $\Theta$ $\longleftrightarrow$ no edge $\longleftrightarrow$ conditional independence

**Steps to draw the graph:**
1. Compute $\Theta = \Sigma^{-1}$
2. Find all $(s,t)$ with $\Theta_{st} \neq 0$, $s < t$ → these are the edges
3. Plot

## 5. Maximum Likelihood Estimation (MLE)

Given $n$ i.i.d. samples, the **sample covariance matrix** is:
$$S = \frac{1}{n}\sum_{i=1}^n x^{(i)}(x^{(i)})^T$$

**MLE problem:**
$$\min_{\Theta \in \mathbb{S}^p_{++}} -\log\det(\Theta) + \langle S, \Theta \rangle$$

- MLE solution (when it exists): $\widehat{\Theta} = S^{-1}$
- Solution **fails** when $n < p$ (S is rank-deficient)

**Log-det function:** $h(\Theta) = -\log\det(\Theta)$
- Convex on $\mathbb{S}^p_{++}$
- Gradient: $\nabla h(\Theta) = -\Theta^{-1}$

## 6. Graphical Lasso

Adds $\ell_1$ penalty on off-diagonal entries to encourage sparsity:

$$\min_{\Theta \in \mathbb{S}^p_{++}} -\log\det(\Theta) + \langle S,\Theta\rangle + \lambda\|\Theta\|_{1,\text{off}}$$

where $\|\Theta\|_{1,\text{off}} = \sum_{s \neq t} |\Theta_{st}|$

## 7. ADMM for Graphical Lasso

Reformulate with splitting variable $Y$, Lagrange multiplier $Z$:

| Step | Update |
|---|---|
| **Θ-update** | $T = Y - \sigma^{-1}(S+Z)$; eigen-decomp $T = Q\,\text{Diag}(\rho)\,Q^T$; then $\gamma_j = \frac{1}{2}\!\left(\rho_j + \sqrt{\rho_j^2 + 4/\sigma}\right)$; $\Theta \leftarrow Q\,\text{Diag}(\gamma)\,Q^T$ |
| **Y-update** | $Y \leftarrow S^\text{off}_{\lambda/\sigma}(\Theta + \sigma^{-1}Z)$ — soft-threshold **off-diagonal** only |
| **Z-update** | $Z \leftarrow Z + \tau\sigma(\Theta - Y)$, with $0 < \tau < \frac{1+\sqrt{5}}{2}$ |

**Soft-threshold operator:** $S_\alpha(x) = \text{sign}(x)\max(|x|-\alpha, 0)$

**Proximal map of log-det:**
$$P_{\frac{1}{\sigma}h}(Y) = Q\,\text{Diag}(\gamma)\,Q^T, \quad \gamma_j = \frac{1}{2}\!\left(\rho_j + \sqrt{\rho_j^2 + 4/\sigma}\right)$$

## 8. Neighbourhood Selection

**Idea:** Regress each variable on all others using Lasso.

For vertex $j$, solve:
$$\beta^j = \arg\min_{\beta \in \mathbb{R}^{p-1}} \frac{1}{2}\|X_{\cdot,-j}\beta - X_{\cdot j}\|^2 + \lambda\|\beta\|_1$$

Neighbourhood: $\mathcal{N}(j) = \{t \mid \beta^j_t \neq 0\}$

**Graph construction:**
- **AND rule:** edge $(s,t)$ iff $s \in \mathcal{N}(t)$ **and** $t \in \mathcal{N}(s)$
- **OR rule:** edge $(s,t)$ iff $s \in \mathcal{N}(t)$ **or** $t \in \mathcal{N}(s)$
