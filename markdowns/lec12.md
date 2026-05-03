# Lecture 12: Clustering

## 1. Core Idea

**Clustering** groups data so that:
- **Intra-cluster distance** is **small** (similar points together)
- **Inter-cluster distance** is **large** (different groups far apart)

## 2. Distance Between Two Objects

$$D(a, b) = \|a - b\|_2 = \sqrt{\sum_{i=1}^{p}(a_i - b_i)^2}$$

Other options: $\|a-b\|_1$, $\|a-b\|_\infty$, cosine similarity.

## 3. Distance Between Two Clusters

| Linkage | Formula | Description |
|---|---|---|
| **Single** | $\min_{a \in C_1, b \in C_2} D(a,b)$ | Shortest pairwise distance |
| **Complete** | $\max_{a \in C_1, b \in C_2} D(a,b)$ | Longest pairwise distance |
| **Average** ⭐ | $\dfrac{1}{\|C_1\|\|C_2\|} \sum_{a \in C_1, b \in C_2} D(a,b)$ | Average of all pairwise distances |

> ⭐ Average linkage is the most widely used.

## 4. Hierarchical Clustering (Bottom-Up / Agglomerative)

**Algorithm:**
1. Start: each object is its own cluster → $n$ clusters
2. Find the pair of clusters with **smallest** $L(C_i, C_j)$
3. Merge them into one cluster
4. Repeat until 1 cluster remains

**Output:** A **dendrogram** (tree diagram)

```
Height
  |       ┌──────────────┐
8 |       │              │
  |   ┌───┘          ┌───┘
5 |   │              │
  | ┌─┘  ┌─┐       ┌─┘  ┌─┐
2 | │    │ │       │    │ │
  | C    G  A  D   B    E  F
```
- **Low fusion height** → very similar objects
- **High fusion height** → dissimilar objects
- **Horizontal position** along the x-axis means **nothing** about similarity

### Reading the Dendrogram

| Q | How to answer |
|---|---|
| Are X and Y similar? | Check how low they fuse — lower = more similar |
| How different are two clusters? | Read the **height** on the vertical axis |
| Is A more similar to B than to C? | **Cannot** determine from dendrogram alone |

### Identifying Clusters: Horizontal Cut

Make a horizontal cut at height $y = a$:
- Number of vertical lines crossed = **number of clusters**

**Choosing the best cut:** pick the cut with the **largest vertical range** (most room to move up/down without hitting a horizontal branch).

```
Cut at y=6  →  3 clusters: {C,G,A,D}, {B}, {E,F}   (range 1.91) ✓
Cut at y=3.4 →  5 clusters: {C,G}, {A,D}, {B}, {E}, {F}  (range 0.44)
```

## 5. K-Means Clustering

**Objective:** minimise total intra-cluster distance

$$\min_{C_1,\ldots,C_K} \sum_{k=1}^{K} \frac{1}{|C_k|} \sum_{i,j \in C_k} \|x_i - x_j\|^2$$

**Algorithm:**

```
Initialise: place K centroids μ₁, ..., μₖ at random

Repeat until assignments don't change:
  (1) Assign each xᵢ to nearest centroid:
          argmin_k  D(xᵢ, μₖ)

  (2) Recompute centroid of each cluster:
          μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ
```

### Choosing K (Elbow Method)

Plot the objective function vs K. Look for the **"elbow"** — the point of most abrupt decrease.

```
Objective
  |  *
  |    *
  |       *
  |           * * * * *
  +--1---2---3---4---5-- K
              ↑
           elbow → K=2 or K=3
```

## 6. Hierarchical vs K-Means Summary

| | Hierarchical | K-Means |
|---|---|---|
| K specified? | No (cut dendrogram after) | Yes, upfront |
| Output | Dendrogram | Flat cluster assignments |
| Sensitive to init? | No | Yes — run multiple times |
| Key choice | Linkage type + cut height | K, initialisation |
| Overlapping clusters? | No | No |

