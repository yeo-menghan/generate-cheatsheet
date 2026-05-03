# Lecture 8: Unsupervised & Semi-Supervised Learning

## 1. Learning Paradigms

| Paradigm | Data | Goal |
|---|---|---|
| **Supervised** | Labeled $\{(\mathbf{x}^{(i)}, y^{(i)})\}$ | Learn $\hat{f} \approx f^*$ |
| **Unsupervised** | Unlabeled $\{\mathbf{x}^{(i)}\}$ | Learn task-agnostic patterns |
| **Semi-supervised** | Labeled $\mathcal{D}_L$ + Unlabeled $\mathcal{D}_U$ | Improve prediction using both |

### Unsupervised Learning Tasks
- **Dimensionality reduction** — compress high-dim data to lower-dim
- **Generative models** — learn $\hat{p}(\mathbf{x}) \approx p(\mathbf{x})$, enable sampling
- **Clustering** — group similar data points
- **Density estimation** — estimate the data distribution

## 2. Principal Component Analysis (PCA)

**Inputs:** Data matrix $X \in \mathbb{R}^{N \times d}$ (mean-centered)

**Covariance matrix:**
$$S = \frac{1}{N} X^T X$$

$S_{ij}$ measures how much feature $i$ and feature $j$ vary together. Diagonal 
entries $S_{ii} = \lambda_i$ are the **variances** along each principal direction.

**Eigenvalue decomposition:**
$$S = U \Lambda U^T, \quad \Lambda = \text{diag}\{\lambda_1, \lambda_2, \ldots, \lambda_d\}, \quad \lambda_1 \geq \lambda_2 \geq \cdots$$

- $U = [\mathbf{u}_1, \ldots, \mathbf{u}_d]$ — eigenvectors, the **principal directions**
- $\lambda_i$ — variance along each principal direction

**PC scores (full):**
$$Z = XU$$

**Reverse transform:**
$$X = ZU^T$$

### PCA as Compression
Keep only first $m$ principal components ($m \ll d$):

$$Z_m = XU_m, \quad U_m \in \mathbb{R}^{d \times m}$$

**Reconstruction:**
$$X \approx X_m = Z_m U_m^T$$

**Reconstruction error:**
$$\frac{1}{N}\sum_{i=1}^{N} \|\mathbf{x}^{(i)} - \mathbf{x}_m^{(i)}\|^2 = \sum_{j=m+1}^{d} \lambda_j$$

> Encoder: $X \mapsto XU_m$ | Decoder: $Z \mapsto ZU_m^T$

### Why Minimizing Reconstruction Error = Maximizing Variance

Reconstruction error = sum of **discarded** eigenvalues $\sum_{j=m+1}^{d}\lambda_j$.
Minimizing this is equivalent to maximizing retained variance $\sum_{j=1}^{m}\lambda_j$ 
— two sides of the same objective.

> **Intuition:** Data shaped like a flat ellipse — long axis has high variance 
> ($\lambda_1$ large), short axis has low variance ($\lambda_2$ small). PCA keeps 
> the long axis and discards the short one, losing minimal information.

### How Many Components to Keep?

Use **explained variance ratio** — not just positive eigenvalues:
$$\text{Explained variance} = \frac{\sum_{j=1}^{m}\lambda_j}{\sum_{j=1}^{d}\lambda_j}$$

Common practice: keep enough components to explain **95% of total variance**. 
Small $\lambda_j$ means that direction has little variance — discarding it loses 
minimal information.

## 3. Autoencoders (AE)

Autoencoders generalize PCA to **nonlinear** compression.

| | PCA | Autoencoder |
|---|---|---|
| Encoder | $\mathbf{z} = U_m^T \mathbf{x}$ | $\mathbf{z} = f(\mathbf{x})$ |
| Decoder | $\mathbf{x}' = U_m \mathbf{z}$ | $\mathbf{x}' = g(\mathbf{z})$ |

- $f: \mathbb{R}^d \to \mathbb{R}^m$ (encoder), $g: \mathbb{R}^m \to \mathbb{R}^d$ (decoder)
- $\mathbb{R}^m$ = **latent space**; $\mathbf{z} = f(\mathbf{x})$ = **latent variable**

**Training objective** (self-supervised):
$$\min_{f,g} \; L\!\left(\mathbf{x},\, g(f(\mathbf{x}))\right)$$
The label *is* the input.

### The Identity Problem
An overcomplete autoencoder ($m \geq d$) can trivially learn $f(\mathbf{x}) = \mathbf{x}$,
$g(\mathbf{z}) = \mathbf{z}$ — perfect reconstruction but **nothing useful learned**.
Each regularized variant prevents this differently:

### Types of Autoencoders

| Type | Condition | Regularizer | Use | Inductive Bias |
|---|---|---|---|---|
| **Undercomplete** | $m \ll d$ | None — bottleneck suffices | Compression, feature extraction | Bottleneck forces learning compact representations |
| **Overcomplete / Regularized** | $m \geq d$ | Requires $\Omega$ | Needs regularizer to prevent identity | Without $\Omega$, trivially learns identity |
| **Denoising** | Train on $\tilde{\mathbf{x}}$ | Noisy input, clean target | Noise removal | True features survive corruption; identity shortcut fails since $\tilde{\mathbf{x}} \neq \mathbf{x}$ |
| **Sparse** | $m \geq d$ | $\Omega = \alpha\|f(\mathbf{x})\|_1$ | Sparse latent features, feature selection | Few neurons fire per input — each neuron is a specialist detector |
| **Contractive** | $m \geq d$ | $\Omega = \alpha\|\nabla f(\mathbf{x})\|_2^2$ | Stable representations | Small input changes → small latent changes; penalizes sensitivity |

**Why identity fails for each:**
- **Denoising:** $g(f(\tilde{\mathbf{x}})) = \tilde{\mathbf{x}} \neq \mathbf{x}$ — 
  passing noise through unchanged gives high loss
- **Sparse:** Identity activates all neurons — $L^1$ penalty makes this expensive
- **Contractive:** Identity has Jacobian $= I$, giving large penalty 
  $\|\nabla f\|^2 = d$ for high-dimensional data

**Regularized loss:**
$$L\!\left(\mathbf{x}, g(f(\mathbf{x}))\right) + \Omega(\mathbf{x}, f, g)$$

**Denoising loss:**
$$L\!\left(\mathbf{x}, g(f(\tilde{\mathbf{x}}))\right) \implies \mathbf{x} \approx g(f(\tilde{\mathbf{x}}))$$

**Seq-to-Seq AE:** Both encoder and decoder are RNNs → handles variable-length inputs.

## 4. Semi-Supervised Learning

**Data:**
$$\mathcal{D}_L = \{(\mathbf{x}^{(i)}, y^{(i)}): i=1,\ldots,N_L\}, \quad \mathcal{D}_U = \{\tilde{\mathbf{x}}^{(i)}: i=1,\ldots,N_U\}, \quad N_U \gg N_L$$

### Problem Types
- **Transductive:** Label the unlabeled points in $\mathcal{D}_U$
- **Inductive:** Learn $f^*$ for predictions on unseen inputs

### Key Assumptions
| Assumption | Description |
|---|---|
| **Continuity** | Nearby inputs share the same label — justifies building similarity graph |
| **Cluster** | Clusters share the same label — labels flow within clusters, stop at boundaries |
| **Manifold** | Data lies on a low-dimensional manifold — labels propagate along manifold, not through empty space |

### Self-Training Algorithm
1. Train $\hat{f}$ on $\mathcal{D}_L$
2. For $\tilde{\mathbf{x}} \in \mathcal{D}_U$, predict $y = \hat{f}(\tilde{\mathbf{x}})$
3. Add $(\tilde{\mathbf{x}}, y)$ to $\mathcal{D}_L$; repeat

**Variants:** add most confident predictions only, weight by confidence.

| Pros | Cons |
|---|---|
| Simple; model-agnostic | Early errors compound |
| General meta-algorithm | Few theoretical guarantees |

### Label Propagation

Spreads labels through the dataset like **ink diffusing through water** — 
exploits all three assumptions simultaneously.

**Step 1 — Build similarity graph** *(continuity assumption)*

Connect every pair of points with edge weight:
$$w_{ij} = \exp\!\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right)$$

Close points → high weight; distant points → near-zero weight.

**Step 2 — Initialize labels**
- Labeled points: fixed to known labels (act as **anchors**)
- Unlabeled points: initialized arbitrarily

**Step 3 — Iterative propagation** *(cluster + manifold assumption)*

Each unlabeled node takes weighted average of neighbor labels:
$$y_i \leftarrow \frac{\sum_j w_{ij} y_j}{\sum_j w_{ij}}$$

Labels flow freely **within** dense clusters but struggle to cross sparse 
regions — respecting cluster boundaries.

**Step 4 — Convergence:** Stop when labels stabilize.

**Concrete example:**
```
🐱A — 1 — 2 — 3 — 4 — 5 — 🐶B

After propagation:
1, 2 → cat (closer to A)
4, 5 → dog (closer to B)
3 → depends on edge weights (decision boundary)
```

> Labels propagate **along the manifold** (chain of connected points), 
> not through empty Euclidean space — two points far apart in raw space 
> but connected along the manifold share the same label.

**Why it can fail:** Wrong $\sigma$ → poor graph construction → early 
errors spread through entire graph.

## 5. Learning Across Tasks

### Multi-Task Learning
Train one model on multiple tasks simultaneously; shared hidden layers learn 
common representations.

**Architecture options:**
- Common inputs → shared layers → task-specific outputs $(y^{(1)}, y^{(2)})$
- Multiple inputs $(x^{(1)}, x^{(2)}, x^{(3)})$ → shared layer → common output $y$

### Transfer Learning
Data across tasks is **similar**; tasks may differ.

**Approach — Pre-training:**
$$\text{Large Dataset 1} \xrightarrow{\text{train}} \text{Model 1} \xrightarrow{\text{init}} \text{Model 2} \xleftarrow{\text{train}} \text{Small Dataset 2}$$

- Warm-start Model 2 with weights from Model 1
- Fine-tune all or some layers on Dataset 2
- Unsupervised pre-training (e.g. autoencoder features) is also valid

### Domain Adaptation
Same task, **different data distributions** (source vs target domain).

| Domain | Data | Labels |
|---|---|---|
| Source | $\mathcal{D}_S$ | Full labels |
| Target | $\mathcal{D}_T$ | None / few |

**Algorithms:**
- Label propagation from few target labels
- Learn a **common representation** via autoencoders or adversarial methods
- Learn **transport maps** to match source → target distribution

## 6. Why These Work: Representation Learning

> All methods above succeed by learning a **compact, task-relevant representation** 
> of the data — either through dimensionality reduction (PCA/AE), graph structure 
> (label propagation), or shared features (transfer learning).
