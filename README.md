# Gold Collector

A genetic-algorithm solver for the weight-dependent Travelling Salesman variant
("gold collection"), paired with an analytical **beta optimizer** for
super-linear weight penalties.


## The problem

Cities lie on a plane; city `0` is the depot and holds no gold. Every other city
`i` holds gold `g_i > 0`. Traversing an edge of length `d` while carrying load
`w` costs

    C = d + (alpha * d * w) ** beta

For `beta > 1` the penalty grows super-linearly, so a good plan must decide *when
to return to the depot to unload*. Objective: visit every city, collect all the
gold, minimise total travel cost.

## Solution: genetic algorithm + beta optimizer

`s343540.solution(problem)` returns a feasible path — a list of `(city, gold)`
steps. Internally (`src/solver_framework.problem_solver`):

1. a genetic algorithm evolves the **order** in which cities are visited;
2. a greedy decoder turns each order into depot-anchored trips;
3. for `beta > 1`, the **beta optimizer** splits each trip analytically;
4. a baseline guard ensures the result is never worse than the per-city baseline.

### Genetic algorithm (`src/goldcollector/`)

- **Genotype:** a permutation of the non-depot cities — the *intent* to visit
  them in that order, ignoring weight and depot returns.
- **Greedy split decoder:** scanning the permutation, for each city it compares
  the marginal cost of *extending* the current trip with the cost of *closing*
  it and serving the city on a fresh depot trip, taking the cheaper. Depot
  returns are thus inserted dynamically.
- **Cost model:** the edge-summed cost `d + (alpha*d*w)^beta`, evaluated in O(1)
  per leg via precomputed distance and penalty (`sum d^beta`) matrices over
  all-pairs shortest paths.
- **Operators:** tournament selection (k=3), ordered crossover (OX), mutation
  (20% insertion / 20% inversion), elitism (top 5%); random initialisation.

The package is modular: cost backends (`cost.py`), decoder (`decoder.py`),
pluggable operators (`operators/`, strategy pattern), a typed config
(`config.py`) and the solver loop (`solver.py`) are independent pieces.

### Beta optimizer (`src/beta_optimizer.py`)

When $\beta > 1$ we can exploit the weights' non linearities to subdivide each round trip in $N$ trips, each one picking
$\frac{w}{N}$ (where $w$ is the total gold picked)

To explain the approach we first consider a simpler case, to get used to the notation, then extend to complex paths.

#### Simple case

Let's consider first a single round trip where the gold is picked from only one city.
Let's denote

- $C_{\text{go}}$ as the cost to get to target city (where gold is picked)
    - $P_{\text{go}}$ is such path represented as pairs `[from, to]` representing pairs of each step
- $C_{return}$ as the cost to return from target city to
    - $P_{\text{return}}$ as explained before
    - $C_{\text{ret s}}$ is the static part, depending only on geoemtric distance
    - $C_{\text{ret}}(w)$ is the part depending on weight carried

```math
    C_{\text{go}} = \sum_{(i,j) \in P_{\text{go}}} d_{i,j} \\
    C_{\text{return}} = \sum_{(i,j) \in P_{\text{return}}} d_{i,j} + (d_{i,j} \alpha w)^\beta =
    \sum_{(i,j) \in P_{\text{return}}} d_{i,j} +
    \sum_{(i,j) \in P_{\text{return}}} (d_{i,j} \alpha w)^\beta =  C_{\text{ret s}} + C_{\text{ret}}(w)
```

Finally the total cost of the round trip can be computed ad

```math
    C = C_{\text{go}} + C_{\text{ret s}} + C_{\text{ret}}(w)
```

Now we consider the cost picking $\frac{w}{N}$ gold at each round-trip, repeated $N$ times

```math
    C(N) = N \left( C_{\text{go}} + C_{\text{ret s}} + C_{\text{ret}} \left(\frac{w}{N} \right) \right )
```

We want to find $N^*$ such that it minimizes the cost.

To find such value, we treat $N$ as a continous variable and differentiate w.r.t. $N$

```math
    C'(N) = C_{\text{go}} + C_{\text{ret s}} + (1 - \beta)N^{-\beta}(\alpha w)^\beta \sum_{(i, j) \in P_{\text{ret}}} d_{i,j}^\beta = 0
```

Solving the equation yields

```math
    N^* = \alpha w \left ( \frac{\beta - 1} {C_{\text{ret s}} + C_{\text{go}} } \sum_{(i, j) \in P_{\text{ret}}} d_{i,j}^\beta \right ) ^ {\frac{1}{\beta}}
```

##### Second Order Condition (Proof of Minimality)

To strictly prove that $N^*$ represents a **minimum** cost (and not a maximum or an inflection point), we must examine
the **second derivative** of the cost function, $C''(N)$.

We differentiate $C'(N)$ with respect to $N$ once again. Note that the static costs ($C_{\text{go}} + C_{\text{ret}}$)
are constants, so their derivative is zero. We therefore focus on the other term.

> *(Where $K = (\alpha w)^\beta \sum d^\beta$ is a constant positive term representing weights and distances.)*

Applying the power rule, we obtain:

```math
C''(N) = (1 - \beta)(-\beta) K N^{-\beta - 1}
```

Under the constraint $\beta > 1$: $(1 - \beta) < 0$, $(-\beta) < 0$, and $K, N^{-\beta-1} > 0$, so

```math
C''(N) > 0 \quad \forall N > 0 \; \text{ when } \beta > 1
```

Since the second derivative is strictly positive, $C(N)$ is strictly convex and the stationary point $N^*$ is the
**global minimum**. Note that, given the scale of $w$ in this problem, splitting is almost always beneficial; and the
$\beta > 1$ constraint is essential — for linear or sub-linear penalties the exploit never helps.

#### General case

In the general case we consider a single round trip in which gold can be picked from multiple cities in the same trip.
We subdivide the path into subpaths where gold carried is constant. For example the path `[0, 2, 4, 5(G), 3, 6(G), 2, 0]`
splits into 3 subpaths carrying respectively `[0, gold(5), gold(5)+gold(6)]`.

```math
    C_s = \sum_{(i,j) \in s} d_{i,j} + (d_{i,j} w_s \alpha)^\beta = C_{\text{s static}} + C_{s \text{dyn}}(w_s)
```

```math
    C(N)
    = N \sum_s C_{s \text{static}} + \alpha^\beta N^{-\beta+1} \sum_s w_s^\beta \sum_{(i,j) \in s} d_{i,j}^\beta
```

Differentiating w.r.t. $N$ and setting to $0$ gives the optimal split count

```math
    N^* = \left (
        \alpha^\beta (\beta - 1) \frac{\sum_s w_s^\beta \sum_{(i,j) \in s} d_{i,j}^\beta}{\sum_s C_{s \text{static}}}
    \right)^{\beta^{-1}}
```

which is exactly what `path_optimizer` computes (rounded to the nearest integer).

#### Implementation strategy

The formula for $N^*$ is applied independently to each round trip in the solution:

1. **Decomposition** — the GA decoder naturally produces depot-anchored round trips.
2. **Independent optimization** — each round trip is split with its own $N^*$.
3. **Reconstruction** — the optimized trips are concatenated into the final path.

This is valid because each round trip is independent (starts and ends at the depot) and the formula depends only on the
trip's internal structure. The result is a closed-form, hyperparameter-free, $O(L)$-per-trip optimization that (almost)
always improves cost when $\beta > 1$.

## Usage

```bash
pip install -r requirements.txt

# solve one instance and compare to the baseline
python -c "from Problem import Problem; from s343540 import compare; print(compare(Problem(100, density=0.2, alpha=1, beta=2)))"

pytest        # parity / feasibility tests
```
