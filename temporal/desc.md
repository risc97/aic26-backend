# Temporal search

**Objective.** Given a video of frames $f_1, f_2, \dots, f_n$ and queries $q_1, q_2, \dots, q_m$, we wish to find a subset of $m$ frames $f_{x_1}, f_{x_2}, \dots, f_{x_m}$ that maximizes the total cossim of each frame with its corresponding queries, while maintaining the frame/query order:

$$
\mathcal{S}(\{ x_i \}) = \sum \texttt{cosSim}(f_{x_i}, q_i) - \sum \lambda(\texttt{ts}_{x_i} - \texttt{ts}_{x_{i-1}})
$$

## Dynamic programming

Let $F(i, j)$ be the maximum score if we match the $i$-th query with the $j$-th frame, and queries after the $i$-th query it with some frames after the $j$-th frame. The DP formula is:

$$
\begin{align*}
F(i, j) &= \texttt{cosSim}(i, j) + \max_{j + \texttt{gapMin}(i) \leq k \leq j + \texttt{gapMax}(i)} F(i + 1, k) - \lambda \cdot (\texttt{ts}_k - \texttt{ts}_j) \\
&= \texttt{cosSim}(i, j) + \lambda \cdot \texttt{ts}_j + \max_{l \leq k \leq r} F(i + 1, k) - \lambda \cdot \texttt{ts}_k
\end{align*}
$$

where $1 \leq \texttt{gapMin}(i) \leq  \texttt{gapMax}(i) \leq n$ are the minimum/maximum allowed gap between the $i$-th and $(i+1)$-th query’s frames. We may define $F(m + 1, j) = 0$ to remove the need of caseworking.

## Optimization

Let $\mathbf{F}^{(i)}$ be the array containing $\{F(i, j) - \lambda \cdot \texttt{ts}_j\}$. Let:

$$
\texttt{maxWindow}(i, l, r) = \max_{l \leq k \leq r} \mathbf{F}^{(i)}_k
$$

If the range-queries are asked in a monotonic fashion, we can resolve all queries in $\mathcal{O}(n)$ time.

To transition from $\textbf{F}^{(i + 1)}$ to $\textbf{F}^{(i)}$ we perform sliding window max-queries. The final result is:

$$
\max_{1 \leq k \leq n} \mathbf{F}^{(i)}_k + \lambda \cdot \texttt{ts}_k
$$

## Implementation

Iteration the query IDs $i$ from $m$ down to $1$. Initially, $\mathbf{F}^{(m)} = \mathbf{C}^{(m)} - \lambda \cdot \texttt{ts}$ (both are $n \times 1$ vectors), to transition from $\mathbf{F}^{(i+1)}$ to $\mathbf{F}^{(i)}$, we do the following:

1. For each $j$, query $\texttt{maxWindow}(i + 1, \min(n + 1, j + \texttt{gapMin}(i)), \min(n, j + \texttt{gapMax}(i))$, if we iterate $j$ from $1$ to $n$, we can update the value inplace. We save $\texttt{opt}[i][j]$ as the optimal index for the sliding window query.
2. Then assign $\mathbf{F} \leftarrow \mathbf{F} + \mathbf{C}^{(i)}$.
3. To trace the path, starting from the optimal state $(1, k_{\max})$, we traverse using $\texttt{opt}[][]$ until we reach $(m, k_\text{final})$.