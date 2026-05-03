# Lecture 11: Seq2Seq, Attention & Transformers

## 1. Sequence-to-Sequence (Seq2Seq) Autoencoders

**Problem:** Input and output sequences may have different lengths, e.g. machine 
translation, text summarisation.

**Given:**
$$\mathbf{x} = (x_1, x_2, \ldots, x_T), \quad \mathbf{y} = (y_1, y_2, \ldots, y_{T'})$$

### Encoder
Processes the input with an RNN (no outputs):
$$h_t = f(x_t, h_{t-1})$$

**Context vector** — summarises entire input:
$$c = q(h_1, \ldots, h_T), \quad \text{simplest choice: } c = h_T$$

**The bottleneck problem:** $c = h_T$ must compress the **entire input sequence 
into a single fixed-size vector** regardless of sequence length. For long sequences, 
earlier tokens' information gets progressively **overwritten** as the hidden state 
processes more tokens — by the time we reach $h_T$, information from $h_1$ may be 
largely lost. This is an **information bottleneck**, not a computational one.

### Decoder
Generates output autoregressively, conditioned on $c$:
$$s_t = g(\hat{y}_{t-1},\, s_{t-1},\, c), \qquad \hat{y}_t \sim P(\hat{y}_t \mid s_t)$$

- $s_0 = c$, initial input $\hat{y}_0 = \langle\text{bos}\rangle$
- Output layer: softmax over vocabulary of size $V$

### Probabilistic Formulation
$$P(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{T'} P(y_t \mid \mathbf{y}_{<t},\, \mathbf{x})$$

**Training loss** (minimise negative log-likelihood):
$$\mathcal{L} = -\sum_{t=1}^{T'} \log P(y_t \vert \mathbf{y}_{<t}, \mathbf{x})$$

## 2. Training: Teacher Forcing

**Problem:** Autoregressive errors compound during training — one wrong prediction 
cascades into worse subsequent predictions.

**Teacher Forcing:** Feed ground-truth token $y_{t-1}$ (not predicted $\hat{y}_{t-1}$) 
during training:
$$s_t = g(y_{t-1},\, s_{t-1},\, c)$$

Converts sequence generation into a collection of **next-token prediction** 
problems → faster, more stable training.

**Caveat — Exposure Bias:** During training the model only ever sees ground-truth 
tokens as inputs — it never experiences the scenario where a wrong prediction feeds 
into the next step. At inference, errors **compound** — one wrong prediction leads 
to a worse next prediction, cascading catastrophically.

**Mitigation — Scheduled Sampling:** Gradually increase the fraction of predicted 
tokens used during training — bridging the gap between training and inference.


## 3. Decoding Strategies

### Greedy Search
$$\hat{y}_t = \underset{y}{\arg\max}\; P(y \mid \mathbf{y}_{<t}, \mathbf{x})$$

- Cost: $O(T'V)$, but can yield **suboptimal** sequences — locally optimal 
  choices may not form a globally optimal sequence.

### Exhaustive Search
- Guaranteed optimal, but cost $O(V^{T'})$ — infeasible.

### Beam Search
- Keep top $k$ (beam width) candidate sequences at each step.
- At each step, expand each of $k$ sequences with all $V$ tokens → prune to top $k$.
- Cost: $O(kVT')$, still linear in $T'$.
- $k = 1$ reduces to greedy search.

**Scoring** (length-normalised log-probability):
$$\text{score}(\mathbf{y}) = \frac{1}{L^\alpha} \sum_{t=1}^{L} \log P(y_t \mid \mathbf{y}_{<t}, \mathbf{x}), \quad \alpha \approx 0.75$$


## 4. Attention Mechanisms

### Motivation
- $c = h_T$ is a bottleneck — poor for long sequences.
- **Average Pooling:** $c = \frac{1}{T}\sum_{i=1}^{T} h_i$ — equal weight to 
  all tokens regardless of relevance (insufficient).

### Attention as Adaptive Pooling

Compute a **different** context vector at each decoding step:
$$c_t = \sum_i w_{t,i}\, h_i$$

**Attention weights** (via softmax over alignment scores):
$$w_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}$$

**Alignment score** $e_{t,i} = a(h_i, s_{t-1})$ — choices:

| Type | Formula |
|------|---------|
| Dot-product | $e_{t,i} = s_{t-1}^\top h_i$ |
| Bilinear (general) | $e_{t,i} = s_{t-1}^\top W h_i$ |
| Additive (Bahdanau) | $e_{t,i} = v^\top \tanh(W_s s_{t-1} + W_h h_i)$ |

### Query-Key-Value (QKV) Framework

Generalises attention as a **library search system**:

```
Query (Q): "I'm looking for information about weather"  ← what decoder wants
   ↓
Keys  (K): [weather: 0.9, food: 0.1, sports: 0.05]     ← what each encoder
                                                           token advertises
   ↓
Weights:   softmax(Q·K) = [0.8, 0.15, 0.05]            ← relevance scores
   ↓
Values (V): actual content of each encoder token        ← what gets retrieved
   ↓
Output:    weighted sum of V                            ← retrieved information
```

$$e_{t,i} = a(k_i, q_t), \qquad c_t = \sum_i w_{t,i}\, v_i$$

- **Key:** $k_i = W_k h_i$ — what each encoder position *advertises*
- **Query:** $q_t = W_q s_{t-1}$ — what the decoder is *seeking*
- **Value:** $v_i = W_v h_i$ — information *retrieved* from encoder

> **Key insight:** Q and K determine *how much* to retrieve; V determines 
> *what* is retrieved. Setting $W_k = W_q = W_v = I$ recovers standard 
> average pooling — QKV generalizes this by making pooling **adaptive**.

**Why different context vector at each step?**

At each decoding step $t$, the query $q_t = W_q s_{t-1}$ changes — the decoder 
is "looking for" different information at each step. Generating "Paris" in 
"The capital of France is Paris" requires attending to "France"; generating 
"capital" requires attending to different tokens entirely.

## 5. Multi-Head Attention

A single attention head may miss different relational aspects simultaneously. 
Use $L$ independent heads — each learns different Q, K, V projections:

$$e_{t,i}^l = a(k_i^l, q_t^l), \quad w_{t,i}^l = \text{softmax}(e_{t,i}^l), \quad C_t^l = \sum_i w_{t,i}^l v_i^l$$

Each head has independent parameters $W_k^l, W_q^l, W_v^l$. Outputs are 
**concatenated** and passed to the decoder.

> **Intuition:** One head might capture syntactic relationships, another 
> semantic ones, another coreference — multiple heads allow the model to 
> jointly attend to information from different representation subspaces.

## 6. Self-Attention (Encoder)

**Motivation:** Remove recurrence from encoder while retaining context-specificity.

### Self-Attention vs Encoder-Decoder Attention

**Encoder-Decoder Attention** — bridges two sequences:
```
Encoder tokens:  [The] [cat] [sat] [on] [mat]
                   ↑     ↑     ↑    ↑    ↑
                  K,V   K,V   K,V  K,V  K,V
                               |
                      attention weights
                               |
Decoder token:            [Le]  ← Q comes from decoder state
```
Q comes from **decoder**, K and V come from **encoder**.

**Self-Attention** — tokens within the same sequence attend to each other:
```
Tokens:  [The] [cat] [sat] [on] [mat]
           ↕     ↕     ↕    ↕    ↕
         Q,K,V Q,K,V Q,K,V Q,K,V Q,K,V

"cat" attends to "sat" and "mat" — building context-aware representation
```
Q, K, V all come from the **same sequence**.

**Why self-attention produces context-dependent representations:**

Standard embedding: "bank" always maps to the **same vector** regardless of context.

Self-attention: "bank" in "river bank" attends strongly to "river"; "bank" in 
"bank account" attends strongly to "account" — **same word, different representation** 
depending on surrounding context.

### Matrix Form
$$Q = XW_q,\quad K = XW_k,\quad V = XW_v$$
$$E = QK^\top, \quad A = \text{softmax}(E), \quad Z = AV$$

**Multi-head self-attention block** = Multi-head self-attention + MLP 
(same MLP shared across all tokens).

## 7. Positional Encoding

Self-attention computes $e_{ij} = q_i^\top k_j$ — a dot product that is 
**order-agnostic**. Shuffling tokens produces identical attention scores. 
Without positional encoding, "cat sat on mat" = "mat on sat cat" to the transformer.

Position information must be injected explicitly:
$$h_i = \text{WordEmbedding}(x_i) + P_i$$

**Sinusoidal Positional Encoding:**
$$P_t = \begin{bmatrix} \sin(\omega_1 t) \\ \cos(\omega_1 t) \\ \sin(\omega_2 t) \\ \cos(\omega_2 t) \\ \vdots \\ \sin(\omega_{d/2}\, t) \\ \cos(\omega_{d/2}\, t) \end{bmatrix}, \qquad \omega_l = \frac{1}{10000^{2l/d}}$$

> **Intuition:** Like a **clock with multiple hands at different frequencies** — 
> each position $t$ gets a unique combination of sine/cosine values across 
> different frequencies, like a unique timestamp. Low $\omega_l$ = slow-changing 
> (coarse position); high $\omega_l$ = fast-changing (fine position).

**Linear shift property:**
$$P_{t+\tau} = M_\tau P_t$$

$$M_\tau = \text{diag}\left( \begin{bmatrix}\cos(\omega_l\tau) & \sin(\omega_l\tau) \\ -\sin(\omega_l\tau) & \cos(\omega_l\tau)\end{bmatrix},\ l = 1\ldots d/2 \right)$$

$M_\tau$ is a fixed rotation matrix depending only on the **gap** $\tau$, 
not the absolute position $t$:

> The relationship between position 3 and 7 ($\tau=4$) undergoes the **same 
> linear transformation** as between position 10 and 14 ($\tau=4$). This enables 
> **shift-invariant attention** — relative relationships between tokens generalize 
> regardless of where in the sequence they appear.

**Properties:** Each $P_t$ is unique and equal-norm; relative displacement $\tau$ 
only depends on the gap.

## 8. Masked Self-Attention (Decoder)

Decoder cannot attend to future tokens — otherwise the model trivially copies 
future tokens instead of learning to predict them.

**Causal mask** $M$:
$$M_{ij} = \begin{cases} -\infty & j > i \\ 0 & \text{otherwise} \end{cases}$$

**Masked attention:**
$$\text{MaskedAttn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top + M}{\sqrt{d_k}}\right)V$$

Adding $-\infty$ before softmax makes $\exp(-\infty) = 0$ — future positions 
get **exactly zero attention weight**.

### Parallelisable in Training, Sequential in Inference

| | Training | Inference |
|---|---|---|
| Tokens available | All at once (teacher forcing) | One at a time |
| Masking | Blocks future positions mathematically | No future tokens exist yet |
| Processing | All positions computed **simultaneously** | Must wait for $\hat{y}_{t-1}$ before computing $\hat{y}_t$ |

> **Key insight:** The mask *simulates* sequentiality during training — 
> allowing full parallelism while preventing information leakage.

## 9. Scaled Dot-Product Attention

If $q, k$ have zero mean and unit variance, then:
$$\text{Var}(q^\top k) = d_k$$

For large $d_k$, dot products grow as $\mathcal{O}(\sqrt{d_k})$ — feeding 
large values into softmax:
$$\text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

The exponential **drastically amplifies differences** — if one $z_i$ is much 
larger than others, $\exp(z_i)$ dominates the sum, pushing softmax toward a 
**one-hot distribution** → vanishing gradients everywhere except the dominant 
position.

**Fix — scale by $\sqrt{d_k}$:**
$$e_{i,j} = \frac{q_i^\top k_j}{\sqrt{d_k}}, \qquad A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

$$\text{Var}\!\left(\frac{q^\top k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

Scaling keeps variance at 1 regardless of $d_k$ — softmax stays **smooth and 
differentiable**, gradients flow properly.

## 10. Encoder-Decoder Transformer Architecture

| Component | Description |
|-----------|-------------|
| Encoder | Stack of Multi-head Self-attention Blocks |
| Decoder | Stack of Masked Multi-head Self-attention Blocks |
| Cross-attention | Queries = decoder states; Keys and Values = encoder outputs |
| MLP / Feed-forward | 2-layer MLP applied token-wise (shared parameters) |
| Add and Norm | Residual connection + Layer Normalisation |

**Pre-norm vs Post-norm:**
- **Post-norm:** tighter control but harder to train
- **Pre-norm:** easier identity mapping, more stable gradients — preferred in 
  deep modern architectures

## 11. Vision Transformers (ViT)

**Key idea:** Treat an image as a sequence of patches (tokens).

**Patch Embedding:**
- Input image: $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$, patch size $P \times P$
- Number of patches: $N = \frac{HW}{P^2}$
- Each flattened patch $x_i \in \mathbb{R}^{P^2 C}$ is projected: 
  $z_i = Ex_i \in \mathbb{R}^D$, where $E \in \mathbb{R}^{D \times P^2C}$
- Equivalent to a convolution with filter size $P \times P$, stride $P$, 
  $D$ output channels

**Class (CLS) token:** Learnable vector $z_\text{cls} \in \mathbb{R}^D$ prepended 
to sequence; aggregates global information for classification.

**Positional Embedding:** Learnable $E_\text{pos} \in \mathbb{R}^{(N+1) \times D}$; 
final input $= [z_\text{cls},\, \mathbf{z}] + E_\text{pos}$

**ViT example (original paper):** $224 \times 224$ image, $16 \times 16$ patches 
$\to N = 196$ tokens.

### CNN vs ViT

| Property | CNN | ViT |
|----------|-----|-----|
| Inductive bias | Strong (translation equivariance, local connectivity) | Weak — no built-in spatial assumptions |
| Receptive field | Grows gradually with depth | Global from first layer |
| Feature extraction | Local convolution | Attention across patches |
| Data requirement | Moderate | Large datasets |
| Small data performance | Better | Worse — must learn spatial structure from scratch |

> **Key insight:** CNN's inductive biases are free assumptions that happen to be 
> correct for images — they don't need to be learned from data. ViT has no such 
> assumptions, so it needs **large amounts of data to discover spatial relationships 
> empirically**. On large enough datasets, ViT matches or exceeds CNN performance.

**Conformer:** Combines convolution (local, short-range) + self-attention 
(global, long-range).

## 12. Transformer Regimes

| Model | Architecture | Pretraining | Best For |
|-------|-------------|-------------|----------|
| **BERT** | Encoder-only | Masked token prediction | Understanding, classification |
| **T5** | Encoder-Decoder | Corrupt span reconstruction | Seq2seq: translation, summarisation |
| **GPT** | Decoder-only | Next-token prediction | Open-ended generation, prompting |

**BERT — Encoder only:**
- Bidirectional context — each token attends to **all** other tokens
- Masked token prediction: randomly mask 15% of tokens, predict from context
- Best for **understanding tasks** where full context matters
- Fine-tune by adding a task-specific head on top of encoder outputs

**T5 — Encoder-Decoder:**
- Text-to-text interface — every task reformulated as string $\to$ string
- Same model for translation, summarization, classification — just change the prompt
- Best for **seq2seq tasks** where input and output are different sequences

**GPT — Decoder only:**
- Causal attention — each token only attends to **previous** tokens
- Next-token prediction: simplest possible pretraining objective
- Best for **generation** — naturally produces text autoregressively
- In-context learning: task examples in prompt guide behavior without fine-tuning

## 13. Key Complexity Summary

| Method | Computational Cost | Notes |
|--------|-------------------|-------|
| Greedy Search | $O(T'V)$ | May be suboptimal |
| Beam Search | $O(kVT')$ | Better quality |
| Self-Attention | $O(N^2)$ per layer | Parallelisable |
| RNN | $O(N)$ per step | Sequential, not parallelisable |

## 14. Tokenisation — General Principle

Transformers are **token sequence processors**. Any modality can be processed 
with appropriate tokenisation:
- **Text:** Word / subword embeddings
- **Vision:** Patch embeddings
- **Audio:** Spectrogram patches
- **Multimodal:** Concatenate tokens from different sources