# Lecture 3

## 1. Deep Fully-Connected Neural Networks

### Shallow (1-hidden-layer) Network
$$\mathbf{h} = g(W\mathbf{x} + \mathbf{b}), \quad \hat{y} = \mathbf{w}^T\mathbf{h} + c$$

### Deep Network ($\ell$ layers)
$$\mathbf{h}^{(i)} = g^{(i)}\!\left(W^{(i)}\mathbf{h}^{(i-1)} + \mathbf{b}^{(i)}\right), \quad \mathbf{h}^{(0)} = \mathbf{x}$$
$$\hat{y} = \mathbf{w}^T\mathbf{h}^{(\ell)} + c$$

### Why Deep?
| Pro | Con |
|-----|-----|
| Efficient approximation of complex/oscillatory functions | Harder, non-convex optimization |
| Better generalisation (with right training/regularisation) | Prone to overfitting |
| Sequential feature extraction as a useful prior | More compute required |

## 2. Stochastic Gradient Descent (SGD)

### Empirical Risk Structure
$$R(\boldsymbol{\theta}) = \frac{1}{N}\sum_{i=1}^{N} R_i(\boldsymbol{\theta}), \quad R_i(\boldsymbol{\theta}) = L\!\left(y^{(i)}, f(\mathbf{x}^{(i)};\boldsymbol{\theta})\right)$$

### Gradient Descent (GD) — $\mathcal{O}(N)$ per step
$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \epsilon\,\frac{1}{N}\sum_{i=1}^{N}\nabla R_i(\boldsymbol{\theta}_k)$$

### SGD — $\mathcal{O}(1)$ per step
$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \epsilon\,\nabla R_{\gamma_k}(\boldsymbol{\theta}_k), \quad \gamma_k \sim \text{Uniform}\{1,\ldots,N\}$$
- **Unbiased**: $\mathbb{E}[\boldsymbol{\theta}_{k+1}\mid\boldsymbol{\theta}_k] = \boldsymbol{\theta}_k - \epsilon\frac{1}{N}\sum_i \nabla R_i(\boldsymbol{\theta}_k)$

### Mini-batch SGD — $\mathcal{O}(M)$ per step
$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \epsilon\,\frac{1}{M}\sum_{j\in B_k}\nabla R_j(\boldsymbol{\theta}_k), \quad |B_k|=M$$
- One full pass over $\{1,\ldots,N\}$ = **one epoch**

### SGD Convergence (Robbins–Monro condition)
Fixed learning rate → SGD may **not** converge (residual variance $\to \frac{\epsilon}{2-\epsilon}$).  
Decaying learning rate converges if:

> $\sum_{k=0}^{\infty}\epsilon_k = \infty \quad$: ensures the steps are large enough in total to actually reach the minimum, no matter how far away you start. If steps decay too fast, you might freeze before converging.

and

> $\sum_{k=0}^{\infty}\epsilon_k^2 < \infty$: ensures steps decay fast enough that the noise from stochastic gradients eventually dies out, eliminating the residual variance problem.

### Momentum (Polyak, 1964)
Reduces zig-zag behaviour by accumulating past gradients:
$$\mathbf{v}_{k+1} = \alpha\mathbf{v}_k - \epsilon\,\nabla R(\boldsymbol{\theta}_k)$$
$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \mathbf{v}_{k+1}$$
- $\alpha \in (0,1)$: momentum parameter ($\alpha=0$ ↔ plain GD)
- Physics analogy for $\alpha$: ball rolling with friction $\leftrightarrow$ second law $\frac{d\mathbf{v}}{dt} = -\frac{\gamma}{m}\mathbf{v} + \frac{1}{m}F_\text{ext}$. 
- $\alpha \to 0$: high friction, ball stops immediately $\to$ reduces to plain GD
- $\alpha \to 1$: frictionless, velocity accumulates indefinitely $\to$ risk of overshooting
- Typical $\alpha \approx 0.9$: effective memory of $\frac{1}{1-\alpha} = 10$ steps back

**Why it helps in narrow valleys:**

$\nabla R$ oscillates left-right in the steep direction but consistently points 
downhill in the flat direction. Momentum **cancels** oscillating components 
(average to zero) while **accumulating** consistent downhill components — 
smoothing the trajectory and accelerating convergence.


### Other SGD Variants
| Method | Key idea |
|--------|----------|
| Adagrad / Adadelta | Adaptive per-parameter learning rates |
| Adam / RMSprop | Momentum + adaptive rates |
| SVRG | Variance reduction |

## 3. Backpropagation Algorithm

### Why Needed?
Chain rule alone is $\mathcal{O}(\text{nodes}^2)$; backprop is $\mathcal{O}(\text{nodes})$ by reusing stored intermediate values.

### Chain Rule
**Scalar:** $\dfrac{dz}{dx} = \dfrac{dz}{dy}\dfrac{dy}{dx}$

**Vector** ($f:\mathbb{R}^m\to\mathbb{R}^n$, $g:\mathbb{R}^n\to\mathbb{R}$):
$$\nabla_{\mathbf{x}} z = \left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right)^{\!T}\!\nabla_{\mathbf{y}} z$$

### Backprop — 3 Steps (1D linear example)

**Setup:** $h^{(j)} = w^{(j)}h^{(j-1)}$, $\hat{y} = wh^{(\ell)}$, $h^{(0)}:=x$

**Step 1 — Forward pass** (store all hidden states):
$$h^{(1)}\to h^{(2)}\to\cdots\to h^{(\ell)}\to\hat{y}$$

**Step 2 — Backward pass** (propagate $p = dR/dh$):
$$\hat{p} = \frac{\partial L}{\partial \hat{y}}, \quad p^{(\ell)} = \hat{p}\,w, \quad p^{(i)} = p^{(i+1)}w^{(i+1)}$$

**Step 3 — Parameter gradients:**
$$\frac{dR}{dw} = \hat{p}\,h^{(\ell)}, \qquad \frac{dR}{dw^{(j)}} = p^{(j)}\,h^{(j-1)}$$

### Key Properties
- **Two passes** only (forward + backward)
- **Cost:** $\mathcal{O}(\text{\#nodes})$
- **Storage:** only forward states $h^{(i)}$ need to be kept

### Why Backprop beats Naive Chain Rule
Naive chain rule recomputes intermediate derivatives **multiple times** — 
for a node deep in the network, its gradient contribution gets recomputed 
once per path leading back to it → $\mathcal{O}(\text{nodes}^2)$.

Backprop computes each node's gradient **exactly once** by storing forward 
states and reusing $p^{(t)}$ in the backward pass → $\mathcal{O}(\text{nodes})$.

### Numerical Example (3-layer, 1D)
**Setup:** $\ell = 3$, $w^{(1)}=2, w^{(2)}=3, w^{(3)}=4$, $w=1$, $x=1$

**Forward pass:**
$$h^{(1)} = 2, \quad h^{(2)} = 6, \quad h^{(3)} = 24, \quad \hat{y} = 24$$

**Backward pass** (assume $\frac{\partial L}{\partial \hat{y}} = \hat{p} = 1$):
$$p^{(3)} = \hat{p} \cdot w = 1, \quad p^{(2)} = p^{(3)} \cdot w^{(3)} = 4, \quad p^{(1)} = p^{(2)} \cdot w^{(2)} = 12$$

**Parameter gradients:**
$$\frac{dR}{dw^{(1)}} = p^{(1)} \cdot h^{(0)} = 12 \cdot 1 = 12$$
$$\frac{dR}{dw^{(2)}} = p^{(2)} \cdot h^{(1)} = 4 \cdot 2 = 8$$
$$\frac{dR}{dw^{(3)}} = p^{(3)} \cdot h^{(2)} = 1 \cdot 6 = 6$$

### Divergent Paths
When a node $h$ feeds into multiple downstream nodes (e.g. $f_1$ and $f_2$),
its gradient is the **sum of contributions from all paths**:

$$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial f_1}\frac{\partial f_1}{\partial h} + \frac{\partial L}{\partial f_2}\frac{\partial f_2}{\partial h}$$

**Example:** $h = 2$, $f_1 = 3h$, $f_2 = h^2$, $L = f_1 + f_2$

**Forward pass:**
$$h = 2, \quad f_1 = 6, \quad f_2 = 4, \quad L = 10$$

**Backward pass:**
$$\frac{\partial L}{\partial f_1} = 1, \quad \frac{\partial L}{\partial f_2} = 1$$
$$\frac{\partial f_1}{\partial h} = 3, \quad \frac{\partial f_2}{\partial h} = 2h = 4$$

**Gradient accumulation at $h$:**
$$\frac{\partial L}{\partial h} = 1 \cdot 3 + 1 \cdot 4 = 7$$

> **Intuition:** $h$ influences $L$ through *both* $f_1$ and $f_2$ — 
> ignoring either path would underestimate $h$'s total effect on $L$. 
> This generalizes to $n$ downstream nodes by summing all $n$ contributions.
> Frameworks like TensorFlow/PyTorch handle this automatically.

### TensorFlow / AD in a Nutshell
Every operation `op` implements:
- `op.forward(x)` → computes $f(\mathbf{x})$
- `op.backward(x)` → computes $\nabla f(\mathbf{x})$

Backprop chains these automatically over any acyclic computational graph.

## Summary Table

| Topic | Key formula / idea |
|-------|--------------------|
| Deep NN | $\mathbf{h}^{(i)} = g^{(i)}(W^{(i)}\mathbf{h}^{(i-1)}+\mathbf{b}^{(i)})$ |
| GD | $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \epsilon\nabla R(\boldsymbol{\theta})$ |
| SGD | Replace full gradient with single-sample estimate |
| Mini-batch | Average gradient over batch $B_k$ of size $M$ |
| Momentum | Accumulate velocity: $\mathbf{v}\leftarrow\alpha\mathbf{v}-\epsilon\nabla R$ |
| Backprop | Forward store → backward propagate → param gradients |
