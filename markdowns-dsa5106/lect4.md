# Lecture 4 — CNN

## 1. Limitations of Fully Connected Neural Networks (FCNN)

$$\mathbf{h}^{(i+1)} = \text{ReLU}\!\left(W^{(i+1)}\mathbf{h}^{(i)} + \mathbf{b}^{(i+1)}\right), \quad \mathbf{h}^{(0)} = \mathbf{x}$$

**Permutation Invariance Problem:**  
If $f \in \mathcal{H}$, then for any permutation $p$ on indices, $f_p(\mathbf{x}) \equiv f(x_{p(1)}, \ldots, x_{p(d)}) \in \mathcal{H}$.  
- FCNN treats all permutations of the input as equally learnable (valid inputs).  
- Since every permutation of the input lives in the same hypothesis space $\mathcal{H}$, the network wastes capacity learning all possible permutations rather than exploiting the fact that nearby pixels are semantically related.
- **Loses spatial/temporal structure** (images, time series become meaningless after random permutation).

## 2. Convolution Operation

### Continuous Convolution
$$s(t) = (x * w)(t) = \int_{-\infty}^{+\infty} x(a)\, w(t - a)\, da$$

- $x(t)$ = **signal / input**
- $w(t)$ = **kernel / filter**
- $s = x * w$ = **feature map**
- Commutative: $x * w = w * x$

### Discrete Convolution
$$s(t) = (x * w)(t) = \sum_{a=-\infty}^{+\infty} x(a)\, w(t - a)$$

### 2D Discrete Convolution (Images)
$$S(i, j) = (I * K)(i, j) = \sum_m \sum_n I(m, n)\, K(i - m,\, j - n)$$

### Cross-Correlation vs. Convolution
| | Formula |
|---|---|
| **Convolution** | $S(i,j) = \sum_m \sum_n I(i-m, j-n)\, K(m,n)$ |
| **Cross-correlation** | $S(i,j) = \sum_m \sum_n I(i+m, j+n)\, K(m,n)$ |

> ⚠️ Deep learning libraries (e.g., PyTorch, TensorFlow) implement **cross-correlation** but call it "convolution."

## 3. Boundary Conditions

| Type | Description | Output Size |
|---|---|---|
| **Circular** | Wraps signal around | Same as input |
| **Valid** | No padding; kernel fully inside | $n - k + 1$ |
| **Same (zero-pad)** | Pads with zeros to match input size | Same as input |

## 4. Convolution as a Linear Operation

A 1D convolution with kernel $\mathbf{w} = (w_1, w_2, w_3)$ is equivalent to a sparse matrix multiply:

$$\mathbf{w} * \mathbf{x} = A(\mathbf{w})\,\mathbf{x}, \quad A(\mathbf{w}) = \begin{pmatrix} w_2 & w_3 & 0 & 0 & 0 \\ w_1 & w_2 & w_3 & 0 & 0 \\ 0 & w_1 & w_2 & w_3 & 0 \\ 0 & 0 & w_1 & w_2 & w_3 \\ 0 & 0 & 0 & w_1 & w_2 \end{pmatrix}$$

## 5. Why CNNs? (Three Motivations)

### Motivation 0: Effective Feature Extractors
Different kernels detect different features (blur, edges, etc.).

### Motivation 1: Sparse Interactions
Each output neuron encodes the prior that local features (**local receptive field**) matter; edges, textures, corners are determined by nearby pixels, not pixels across the image

| | Computational Cost | 
|---|---|
| **CNN** | $\mathcal{O}(k \times l \times m \times n)$ |
| **FCNN** | $\mathcal{O}(m^2 \times n^2)$ |

For image $m \times n$ with kernel $k \times l$: significant savings when $k, l \ll m, n$.

### Motivation 2: Parameter Sharing (Tied Weights)
The same kernel is reused at every position. Encodes the prior that features are position-agnostic; a horizontal edge detector should work the same everywhere in the image, not just where it was trained

| | Parameters stored |
|---|---|
| **CNN** | $\mathcal{O}(k \times l)$ |
| **Without sharing** | $\mathcal{O}(m \times n \times k \times l)$ |

### Motivation 3: Translation Equivariance → Invariance

**Equivariance:** $f(g(x)) = g(f(x))$  
**Invariance:** $f(g(x)) = f(x)$

**Convolutions are equivariant to translations.** Let $T_\tau(x)(t) = x(t - \tau)$:

$$w * T_\tau(x) = T_\tau(w * x)$$

**Proof:**
$$[w * T_\tau(x)](t) = \int w(s)\, x(t - s - \tau)\, ds = T_\tau(w * x)(t) \quad \checkmark$$

Element-wise nonlinearities $\sigma$ are also equivariant to translation:
$$\sigma(T(\mathbf{x})) = T(\sigma(\mathbf{x}))$$

**Composition of equivariant maps is equivariant** — so each CNN layer  
$$\mathbf{h}^{(i+1)} = \sigma(\mathbf{w}^{(i+1)} * \mathbf{h}^{(i)} + \mathbf{b})$$
is equivariant to translations.

> **Intuition:** Conv layers say *"there's a cat, and it's over there"* — 
> they track the feature's location as it moves across the image.

**Building invariance:** If $f_1, \ldots, f_\ell$ are equivariant and $F$ is 
invariant w.r.t. $g$, then $F \circ f_\ell \circ \cdots \circ f_1$ is invariant. 
A deep CNN:

$$\mathbf{h}^{(i+1)} = \sigma(W * \mathbf{h}^{(i)} + \mathbf{b}), \quad \hat{y} = \mathbf{1}^T \mathbf{h}^{(\ell)} + c$$

is **translation invariant** — a key inductive bias for image tasks.

**Why does the final step $\hat{y} = \mathbf{1}^T\mathbf{h}^{(\ell)} + c$ give invariance?**

$\mathbf{1}^T\mathbf{h}^{(\ell)}$ **sums over all spatial positions** — it doesn't 
care *where* a feature is, only *whether* it exists anywhere in the image. 
This is the invariant map $F$ that collapses the remaining spatial information.

> **Intuition:** The final aggregation says *"there's a cat somewhere"* — 
> discarding location entirely.

**Summary of the two-stage process:**

| Stage | Operation | Property | Intuition |
|-------|-----------|----------|-----------|
| Conv + ReLU layers | $\mathbf{h}^{(i+1)} = \sigma(W * \mathbf{h}^{(i)} + \mathbf{b})$ | Equivariant | Tracks *where* features are |
| Final aggregation | $\hat{y} = \mathbf{1}^T\mathbf{h}^{(\ell)} + c$ | Invariant | Discards location; detects *whether* features exist |

Invariance therefore emerges from **equivariant feature extraction** followed by **location-discarding aggregation** — not from any single layer alone.

## 6. Pooling Layers

Pooling builds **approximate invariance** to small translations/deformations by summarising local regions into a single value — reducing spatial dimensions while retaining dominant features.

### Max Pooling (1D, stride $p$)
$$T_{mp}(\mathbf{x})_k = \max_{i = kp,\, \ldots,\, (k+1)p} x_i$$

| Type | Formula | Description | Use |
|---|---|---|---|
| **Max pooling** | $\max_{i \in \text{window}} x_i$ | Takes the maximum in each window | Preserves dominant features |
| **Average pooling** | $\frac{1}{p}\sum_{i \in \text{window}} x_i$ | Takes the mean in each window | Smoother, preserves overall signal |

> **Intuition:** Max pooling says *"did this feature appear anywhere in this 
> region?"* — a small translation of the feature stays within the window and 
> still produces the same max output, hence approximate invariance.

## 7. Tensor Convolution (Multi-channel)

- Input: $H \times W \times C_{in}$
- Each filter: $k \times l \times C_{in}$ (one per output channel)
- Each filter produces one 2D output channel via element-wise convolution + sum over input channels
- Output: $H' \times W' \times C_{out}$, where $C_{out} = \#\text{filters}$

## 8. CNN Architecture

```
Input → [Conv → BN → ReLU → Pool] × L → Flatten → FCNN → Output
```

Convolutional layers extract hierarchical spatial features; FCNN head performs final classification/regression.

---

### Layer-by-Layer Reference

#### CONV Layer

Each filter slides across the input, computing dot products over a local region spanning **all input channels**.

**Notation:**
- Input: $W \times H \times C_{in}$
- Kernel size: $k \times k$
- Number of filters: $C_{out}$
- Stride: $s$ (how many pixels the kernel moves each step)
- Padding: $p$ (zeros added to border; $p=0$ = valid, $p=\lfloor k/2 \rfloor$ = same)

**Output spatial dimensions:**
$$W_{out} = \left\lfloor\frac{W - k + 2p}{s}\right\rfloor + 1, \qquad H_{out} = \left\lfloor\frac{H - k + 2p}{s}\right\rfloor + 1$$

> **Intuition:** Numerator $W - k + 2p$ = how much space the kernel has to slide. Dividing by stride $s$ gives number of steps. $+1$ counts the starting position.

**Output volume:** $W_{out} \times H_{out} \times C_{out}$

**Parameters:**

| | Formula | Why |
|--|---------|-----|
| Weights | $k \times k \times C_{in} \times C_{out}$ | Each of $C_{out}$ filters has $k \times k \times C_{in}$ weights — must span all input channels |
| Biases | $C_{out}$ | One bias per filter |
| **Total** | $k \times k \times C_{in} \times C_{out} + C_{out}$ | |

> **Note:** Weights are **shared across spatial positions** — the same filter is applied everywhere. This is why parameter count is $\mathcal{O}(k^2)$ regardless of input size, vs $\mathcal{O}(W^2H^2)$ for FC layers.

---

#### Activation Layer (ReLU / Leaky ReLU / Tanh etc.)

Applied **element-wise** after convolution — introduces nonlinearity. Without nonlinearity, stacking conv layers collapses into a single linear operation.

**Output volume:** Same as input — $W \times H \times C$ (unchanged)

**Parameters:** $0$

---

#### POOL Layer

Downsamples spatial dimensions by summarising local regions. Builds approximate translation invariance.

**Notation:** Pool size $p_{pool}$, typically stride = $p_{pool}$ (non-overlapping)

**Output spatial dimensions:**
$$W_{out} = \left\lfloor\frac{W}{p_{pool}}\right\rfloor, \qquad H_{out} = \left\lfloor\frac{H}{p_{pool}}\right\rfloor$$

**Output volume:** $W_{out} \times H_{out} \times C_{in}$ — **channels unchanged**

**Parameters:** $0$

> **Intuition:** Max pooling asks *"did this feature appear anywhere in this region?"* — a small translation stays within the window and still produces the same max, giving approximate translation invariance.

---

#### BATCHNORM Layer

Normalises activations across the batch, then rescales with learnable parameters.

$$\text{BN}(H; \boldsymbol{\gamma}, \boldsymbol{\beta})^{(i)} = \boldsymbol{\gamma} \circ \left(\frac{\mathbf{h}^{(i)} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}\right) + \boldsymbol{\beta}$$

**Output volume:** Same as input — $W \times H \times C$ (unchanged)

**Parameters:**

| | Formula | Why |
|--|---------|-----|
| $\boldsymbol{\gamma}$ (scale) | $C$ | One per channel — shared across all spatial positions |
| $\boldsymbol{\beta}$ (shift) | $C$ | One per channel — shared across all spatial positions |
| **Total** | $2 \times C$ | |

> **Training vs Inference:** During training, uses batch statistics $\boldsymbol{\mu}, \boldsymbol{\sigma}$. During inference, uses moving averages accumulated during training.

---

#### FLATTEN Layer

Reshapes the 3D volume into a 1D vector to feed into FC layers.

$$W \times H \times C \xrightarrow{\text{flatten}} (W \times H \times C)$$

**Output size:** $n_{in} = W \times H \times C$

**Parameters:** $0$ — pure reshaping, no learnable parameters

> **Example:** A $8 \times 8 \times 16$ volume flattens to a vector of size $8 \times 8 \times 16 = 1024$.

---

#### FC (Fully Connected) Layer

Every input neuron connects to every output neuron. Used as the final classification/regression head.

**Notation:** Flattened input size $n_{in}$, output size $n_{out}$

**Output size:** $n_{out}$

**Parameters:**

| | Formula | Why |
|--|---------|-----|
| Weights | $n_{in} \times n_{out}$ | Every input connected to every output |
| Biases | $n_{out}$ | One per output neuron |
| **Total** | $n_{in} \times n_{out} + n_{out}$ | |

> FC layers typically dominate total parameter count — e.g. $1024 \times 10 + 10 = 10250$ vs a conv layer's $3\times3\times8\times16+16 = 1168$.

---

### Quick Reference Table

| Layer | Output Size | Parameters | Notes |
|-------|------------|------------|-------|
| **CONV** | $W_{out} \times H_{out} \times C_{out}$ | $k^2 \cdot C_{in} \cdot C_{out} + C_{out}$ | Shared weights across spatial positions |
| **Activation** | $W \times H \times C$ (unchanged) | $0$ | Element-wise nonlinearity |
| **POOL** | $\lfloor W/p \rfloor \times \lfloor H/p \rfloor \times C$ | $0$ | Channels unchanged |
| **BATCHNORM** | $W \times H \times C$ (unchanged) | $2 \times C$ | $\gamma, \beta$ per channel only |
| **FLATTEN** | $W \cdot H \cdot C$ | $0$ | Pure reshape |
| **FC** | $n_{out}$ | $n_{in} \cdot n_{out} + n_{out}$ | Dense connections; dominates param count |

---

### Worked Example

For the architecture: Input $32\times32\times3$ → CONV3-8 → BN → ReLU → POOL-2 → CONV3-16 → FLATTEN → FC-10:

| Layer | Output Volume | Parameters |
|-------|--------------|------------|
| Input | $32\times32\times3$ | 0 |
| CONV3-8 | $32\times32\times8$ | $3\times3\times3\times8 + 8 = 224$ |
| BATCHNORM | $32\times32\times8$ | $2\times8 = 16$ |
| ReLU | $32\times32\times8$ | 0 |
| POOL-2 | $16\times16\times8$ | 0 |
| CONV3-16 | $16\times16\times16$ | $3\times3\times8\times16 + 16 = 1168$ |
| FLATTEN | $16\times16\times16 = 4096$ | 0 |
| FC-10 | $10$ | $4096\times10 + 10 = 40970$ |

## 3D CNN (Supplementary)

Extends 2D CNN by sliding a $k \times k \times k$ kernel across 
**width, height and depth/time** — captures spatial AND temporal features.

**Common use cases:**
- Video understanding (depth = time frames)
- Medical imaging (depth = MRI/CT slices)

### Dimension Formulas

**Notation:** Input $W \times H \times D \times C_{in}$, kernel $k \times k \times k$, 
stride $s$, padding $p$

$$W_{out} = \left\lfloor\frac{W - k + 2p}{s}\right\rfloor + 1$$
$$H_{out} = \left\lfloor\frac{H - k + 2p}{s}\right\rfloor + 1$$
$$D_{out} = \left\lfloor\frac{D - k + 2p}{s}\right\rfloor + 1$$

**Output volume:** $W_{out} \times H_{out} \times D_{out} \times C_{out}$

### Parameter Count

| | Formula |
|---|---|
| **Weights** | $k^3 \times C_{in} \times C_{out}$ |
| **Biases** | $C_{out}$ |
| **Total** | $k^3 \times C_{in} \times C_{out} + C_{out}$ |

### 2D vs 3D Comparison

| | 2D CNN | 3D CNN |
|---|---|---|
| Kernel | $k \times k$ | $k \times k \times k$ |
| Captures | Spatial features | Spatial + temporal features |
| Weights | $k^2 \times C_{in} \times C_{out}$ | $k^3 \times C_{in} \times C_{out}$ |
| Use case | Images | Video, volumetric data |

> **Note:** 3D CNNs are significantly more computationally expensive — 
> parameter count scales as $k^3$ vs $k^2$, making them costly for large inputs.

## 9. Key Comparisons: CNN vs. FCNN

| Property | FCNN | CNN |
|---|---|---|
| Connectivity | Dense (all-to-all) | Local (receptive field) |
| Parameters | $\mathcal{O}(m^2 n^2)$ | $\mathcal{O}(kl)$ |
| Compute | $\mathcal{O}(m^2 n^2)$ | $\mathcal{O}(klmn)$ |
| Permutation of input | Invariant | Sensitive (preserves structure) |
| Translation | No built-in symmetry | Equivariant / Invariant |
 