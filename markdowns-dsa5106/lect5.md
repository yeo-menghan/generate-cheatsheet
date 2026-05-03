# Lecture 5: Recurrent Neural Networks

## 1. Time Series Tasks

| Task | Input | Output | Examples |
|---|---|---|---|
| **Sequence Prediction** | $x^{(1)}, \ldots, x^{(t)}$ | $x^{(t+1)}$ or $x^{(t+1)}, \ldots, x^{(t+\tau)}$ | Stock price, weather |
| **Sequence Classification/Regression** | $x^{(1)}, \ldots, x^{(\tau)}$ | Label $y$ (discrete or continuous) | Fraud detection, arrhythmia |
| **Sequence-to-Sequence** | $x^{(1)}, \ldots, x^{(\tau)}$ | $y^{(1)}, \ldots, y^{(\tau)}$ | Machine translation |
| **Sequence Generation** | Seed $x^{(1)}, \ldots, x^{(\tau)}$ | Extended sequence $x^{(\tau+1)}, x^{(\tau+2)}, \ldots$ | Poetry, music |

**Supervised Learning Formulation:**

$$y^{(t)} = F_t^*\!\left(x^{(1)}, x^{(2)}, \ldots, x^{(\tau)}\right), \quad \text{goal: learn } \widehat{F}_t \approx F_t^*$$

## 2. Dynamical Systems & Parameter Sharing

**Autonomous dynamical system** (parameters shared in time):
$$s^{(t+1)} = f\!\left(s^{(t)};\, \theta\right)$$

**With external inputs:**
$$s^{(t)} = f\!\left(s^{(t-1)},\, x^{(t)},\, \theta\right)$$

> **Key insight:** Unlike feed-forward dynamics $h^{(t+1)} = f(h^{(t)}; \theta^{(t)})$, a single $\theta$ is shared across all time steps.

## 3. Simple RNN

$$\boxed{h^{(t)} = \sigma_r\!\left(W h^{(t-1)} + U x^{(t)} + b\right)}$$

$$\boxed{\widehat{y}^{(t)} = \sigma_o\!\left(V h^{(t)} + c\right)}$$

- **Trainable parameters:** $(W, U, b, V, c)$
- **Recurrent activation $\sigma_r$:** typically $\tanh$
- **Initialization:** $h^{(0)} = \mathbf{0}$
- **Output activation $\sigma_o$:** depends on task

### Activation Function Comparison

| Activation | Peak Gradient | Issues in RNNs |
|---|---|---|
| $\tanh$ | 1 (at $a=0$) | Saturation to $\pm 1$ |
| Sigmoid | 0.25 | Non-zero center → bias accumulation |
| ReLU | Unbounded | Dead neurons are permanently dead (shared weights) |

**Why tanh is preferred:** When $h^{(t)}$ is small and $w \approx 1$, $h^{(t)} \approx h^{(t-1)}$ and gradients $\approx 1$ (stable identity dynamics).

## 4. Loss Functions

**Single prediction** (Tasks I & II — terminal loss):
$$L\!\left(y,\, \widehat{y}^{(\tau)}\right) = \frac{1}{2}\left\|y - \widehat{y}^{(\tau)}\right\|^2$$

**Sequence prediction** (Task III — sum over time):
$$\sum_{t=1}^{\tau} L\!\left(y^{(t)},\, \widehat{y}^{(t)}\right) = \frac{1}{2}\sum_{t=1}^{\tau} \left\|y^{(t)} - \widehat{y}^{(t)}\right\|^2$$

## 5. Backpropagation Through Time (BPTT)

1. **Unroll** the computational graph
2. **Compute deltas:** $p^{(t)} := \nabla_{h^{(t)}} L$
3. **Sum contributions** across all time steps (due to parameter sharing)

**Backward recursion (linear 1D example):**

$$\frac{dL}{dw} = \frac{\partial L}{\partial \widehat{y}}\!\left(y, \widehat{y}^{(\tau)}\right) \sum_{t=1}^{\tau} w^{\tau - t}\, h^{(t-1)} = \sum_{t=1}^{\tau} (\tau - t)\, w^{\tau - t - 1}\, x^{(t)}$$

$$p^{(t-1)} = w\, p^{(t)}, \qquad p^{(\tau)} = \frac{\partial L}{\partial \widehat{y}}\!\left(y, \widehat{y}^{(\tau)}\right)$$

## 6. Gradient Explosion & Vanishing

In the linear scalar RNN $h^{(t)} = w h^{(t-1)} + x^{(t)}$:

| Condition | Effect |
|---|---|
| $w > 1$ | **Gradient explosion** — gradients diverge for large $\tau$ |
| $w < 1$ | **Gradient vanishing** — earlier inputs receive exponentially small weight |

> This is the fundamental challenge: no $w$ avoids both problems simultaneously.

**In the matrix case** ($W \in \mathbb{R}^{n \times n}$), gradients scale with 
$W^\tau$ — stability requires **all eigenvalues** of $W$ to equal exactly 1, 
which is practically impossible to maintain throughout training.

## 7. Long Short-Term Memory (LSTM)

LSTM introduces a **cell state** $C^{(t)}$ that flows through time with minimal modification, controlled by **trainable gates**.

The cell state update:
$$C^{(t)} = f^{(t)} \cdot C^{(t-1)} + i^{(t)} \cdot \tilde{C}^{(t)}$$

The gradient flowing back through time:
$$\frac{\partial C^{(t)}}{\partial C^{(t-1)}} = f^{(t)}$$

Since $f^{(t)}$ is a **learned gate** rather than a fixed weight raised to 
power $\tau$, the network can learn $f^{(t)} \approx 1$ to preserve gradients 
over long sequences — acting as a **gradient highway**.

### Full LSTM Equations

$$f^{(t)} = \sigma\!\left(W_f h^{(t-1)} + U_f x^{(t)} + b_f\right) \quad \text{(forget gate)}$$

$$i^{(t)} = \sigma\!\left(W_i h^{(t-1)} + U_i x^{(t)} + b_i\right) \quad \text{(input gate)}$$

$$o^{(t)} = \sigma\!\left(W_o h^{(t-1)} + U_o x^{(t)} + b_o\right) \quad \text{(output gate)}$$

$$C^{(t)} = f^{(t)} \cdot C^{(t-1)} + i^{(t)} \cdot \sigma_c\!\left(W_c h^{(t-1)} + U_c x^{(t)} + b_c\right)$$

$$h^{(t)} = o^{(t)} \cdot \sigma_h\!\left(C^{(t)}\right)$$

| Component | Role |
|---|---|
| Forget gate $f^{(t)}$ | How much of $C^{(t-1)}$ to keep |
| Input gate $i^{(t)}$ | How much new info to write into $C^{(t)}$ |
| Output gate $o^{(t)}$ | How much of $C^{(t)}$ to expose as $h^{(t)}$ |
| Cell state $C^{(t)}$ | Long-term memory highway |
| Hidden state $h^{(t)}$ | Short-term/working memory, used for predictions |

## 8. Gated Recurrent Unit (GRU)

Simpler gated variant with **no separate cell state**:

$$z_t = \sigma\!\left(W_z \cdot [h_{t-1},\, x_t]\right) \quad \text{(update gate)}$$

$$r_t = \sigma\!\left(W_r \cdot [h_{t-1},\, x_t]\right) \quad \text{(reset gate)}$$

$$\tilde{h}_t = \tanh\!\left(W \cdot [r_t * h_{t-1},\, x_t]\right)$$

$$h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$$


| Gate | Role | LSTM Equivalent |
|---|---|---|
| Update gate $z_t$ | How much to update hidden state | Forget + input gate combined |
| Reset gate $r_t$ | How much past state influences candidate | Partial forget |

**GRU vs LSTM:**

| | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| States | Cell state + hidden state | Hidden state only |
| Parameters | More | Fewer |
| Performance | Similar | Similar |
| Preferred when | Long sequences, complex tasks | Compute is limited |

> Both solve vanishing gradients via **learned gating** rather than fixed 
> weight multiplication — allowing gradients to flow selectively across 
> long time horizons.

## 9. Deep RNNs

**Shallow RNN** — one recurrent layer:
$$h^{(t)} = \sigma_r\!\left(W h^{(t-1)} + U x^{(t)} + b\right), \qquad \widehat{y}^{(t)} = \sigma_o\!\left(V h^{(t)} + c\right)$$

**Deep RNN** — stacked recurrent layers (example with 2 layers):
$$h^{(t)} = \sigma_r\!\left(W h^{(t-1)} + U x^{(t)} + b\right)$$
$$z^{(t)} = \sigma_r\!\left(S z^{(t-1)} + T h^{(t)} + a\right)$$
$$\widehat{y}^{(t)} = \sigma_o\!\left(V z^{(t)} + c\right)$$

**Other depth strategies:** stacked hidden/cell states, MLPs for inter-layer connections.


## Key Intuitions Summary

| Concept | Why It Matters |
|---|---|
| Parameter sharing in time | Handles variable-length sequences; generalizes across positions |
| Hidden state | Provides memory — predictions can depend on the full history |
| Gating | Controls information flow to mitigate vanishing/exploding gradients |
| Cell state (LSTM) | Provides a "gradient highway" for long-range dependencies |
| BPTT | Standard backprop applied to unrolled computational graph |
