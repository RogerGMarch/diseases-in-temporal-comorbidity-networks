# Methodology

This document explains the statistical methods, network analysis approaches, and parameter choices used in this study.

## Table of Contents

1. [Network Construction](#network-construction)
2. [Network Metrics](#network-metrics)
3. [Outlier Detection](#outlier-detection)
4. [High-Mortality Sinks](#high-mortality-sinks)
5. [Critical Bridge Edges](#critical-bridge-edges)
6. [Statistical Methods](#statistical-methods)
7. [Parameter Justification](#parameter-justification)

---

## Network Construction

### Data Source

- **Dataset**: Population-wide health data from 8.9M hospital patients (1997-2014)
- **Population**: Austrian hospital inpatient records
- **Stratification**: Sex (Female/Male) × Age (8 groups) = 16 networks
- **Disease coding**: ICD-10 diagnosis codes

### Adjacency Matrix

**Definition:** An N×N symmetric matrix where N = number of diseases

**Entry (i,j):** Association strength between disease i and disease j

**Calculation:** Based on observed comorbidity patterns in the patient population
- Higher values indicate stronger associations
- Diagonal elements represent self-connections (typically not used)

### Network Filtering

**Isolated nodes**: Nodes with degree = 0 are **filtered out** for most analyses

**Rationale:**
- Isolated diseases provide no network information
- Cannot contribute to path-based metrics (betweenness, closeness)
- Paper states: "We filtered for nodes with at least one neighbor"

**When filtering is NOT applied:**
- Initial data loading (all nodes preserved)
- Some degree distribution analyses

---

## Network Metrics

### Degree (k)

**Definition:** Number of connections a node has

**Formula:** k(i) = ∑ⱼ A(i,j) where A is the adjacency matrix

**Interpretation:**
- Higher degree = more comorbidities
- Degree represents disease "connectivity"
- Used to identify "hub" diseases

**Use in paper:**
- Compared with disease prevalence
- Identifies over/under-connected diseases

---

### Betweenness Centrality (BC)

**Definition:** Fraction of shortest paths between all node pairs that pass through a given node

**Formula:**
```
BC(v) = ∑_{s≠v≠t} (σₛₜ(v) / σₛₜ)
```

Where:
- σₛₜ = number of shortest paths from s to t
- σₛₜ(v) = number of those paths passing through v

**Normalization:** Divided by (N-1)(N-2) for comparability across networks

**Interpretation:**
- High betweenness = "bridge" node connecting different parts of network
- Important for information flow and disease progression pathways
- Nodes with high BC are critical intervention points

**Computational note:**
- Calculated on FULL graph before filtering
- Computationally expensive for large networks (O(N³))

---

### Closeness Centrality (CC)

**Definition:** Reciprocal of average distance to all other nodes

**Formula:**
```
CC(v) = (N-1) / ∑_{u≠v} d(v,u)
```

Where d(v,u) is the shortest path length from v to u

**Interpretation:**
- High closeness = central position, easy to reach from anywhere
- Represents disease "accessibility" in comorbidity network

**Special handling:**
- For disconnected graphs, calculated per component
- Averaged across all connected nodes

---

### Average Path Length (L)

**Definition:** Mean of shortest paths between all pairs of nodes

**Formula:**
```
L = (1 / (N(N-1))) × ∑_{i≠j} d(i,j)
```

**Interpretation:**
- Smaller L = more compact network, diseases more closely related
- Represents average "distance" between diseases
- Indicates how many steps to go from one disease to any other

**Handling disconnected graphs:**
- Uses largest connected component only
- Infinite paths between components excluded

---

### Clustering Coefficient (C)

**Definition:** Probability that neighbors of a node are also connected

**Formula (local):**
```
C(v) = (2 × triangles) / (k(v) × (k(v)-1))
```

**Average (global):** Mean of local clustering coefficients

**Interpretation:**
- High C = nodes form tightly connected groups
- Measures "cliquishness" of network
- Related to transitivity in comorbidity relationships

---

### Modularity (Q)

**Definition:** Strength of division of network into communities

**Formula:**
```
Q = (1/2m) × ∑ᵢⱼ [Aᵢⱼ - (kᵢkⱼ/2m)] × δ(cᵢ,cⱼ)
```

Where:
- m = number of edges
- cᵢ = community of node i
- δ(cᵢ,cⱼ) = 1 if same community, 0 otherwise

**Community detection:** Louvain algorithm (maximizes modularity)

**Interpretation:**
- Q > 0.3: Strong community structure
- Q < 0.1: Weak or no community structure
- Communities may represent disease categories or pathophysiological groups

---

## Outlier Detection

### Degree-Prevalence Relationship

**Hypothesis:** Disease degree should correlate with prevalence

**Why?**
- Common diseases = more opportunities for comorbidity
- Expected: degree ∝ prevalence

**Outliers:**
- **High-degree outliers**: More connected than expected from prevalence
- **Low-degree outliers**: Less connected than expected from prevalence

### Log-Ratio Method

**Step 1: Calculate ratio**
```
Ratio = Degree / Prevalence
```

**Step 2: Log transformation**
```
Log_ratio = log₁₀(Ratio)
```

**Why log₁₀?**
- Handles wide range of ratios (0.01 to 100+)
- Symmetry: log₁₀(10) = -log₁₀(0.1) = 1
- Easy interpretation: log_ratio = 1 means 10× deviation

### Percentile Method

**Thresholds:**
- Lower bound: 20th percentile
- Upper bound: 80th percentile

**Rationale:**
- Percentiles are non-parametric (no distribution assumptions)
- 20th/80th captures tails without being too extreme
- Middle 60% considered "normal"

**Classification:**
- Log_ratio < 20th percentile → Low-degree outlier
- Log_ratio > 80th percentile → High-degree outlier
- Between percentiles → Normal

### Modified Z-Score

**Purpose:** Quantify deviation magnitude (robust to outliers)

**Formula:**
```
M = 0.6745 × (x - median) / MAD
```

Where:
- MAD = Median Absolute Deviation = median(|x - median|)
- 0.6745 = 75th percentile of standard normal distribution

**Why modified instead of standard z-score?**
- Standard z-score uses mean and std (sensitive to outliers)
- Modified z-score uses median and MAD (robust to outliers)
- Better for datasets with extreme values

**Interpretation:**
- |M| > 3.5: Potential outlier (rule of thumb)
- |M| > 2.5: Moderate outlier

**Reference:** Iglewicz & Hoaglin (1993) "How to detect and handle outliers"

---

## High-Mortality Sinks

### Concept

**Definition:** Diseases that are BOTH:
1. Central in network (high betweenness)
2. Associated with high mortality

**Clinical significance:**
- Critical intervention points
- Lie on many disease progression pathways
- Have severe outcomes

### Methodology

**Step 1: Calculate z-scores within each sex-age stratum**
```
z_betweenness = (BC - μ_BC) / σ_BC
z_mortality = (Mort - μ_Mort) / σ_Mort
```

**Why stratify by sex-age?**
- Different disease patterns across demographics
- Ensures comparisons within similar populations
- Controls for age/sex confounding

**Step 2: Compute z-score product**
```
z_product = z_betweenness × z_mortality
```

**Why product?**
- Acts as logical AND (both must be high)
- Negative if either z-score is negative
- Only positive when BOTH are above average

**Step 3: Filter to double-positive nodes**
```
Include if: z_betweenness > 0 AND z_mortality > 0
```

**Rationale:**
- Must be above average in BOTH dimensions
- Strict definition of "high-risk"
- Reduces false positives

**Step 4: Select top percentile**
```
Threshold = Quantile(z_product, 1 - 0.20) = 80th percentile
Keep if: z_product >= Threshold
```

**Default:** Top 20% (can be adjusted in config)

**Step 5: Calculate geometric mean for ranking**
```
z_geom_mean = √(z_betweenness × z_mortality)
```

**Why geometric mean?**
- Interpretable scale (average of the two z-scores on geometric scale)
- Used for ranking/reporting, not selection
- Selection based on z_product (stricter)

---

## Critical Bridge Edges

### Concept

**Definition:** Disease pairs (edges) that are BOTH:
1. Critical connections (high edge betweenness)
2. Have large mortality differences

**Clinical significance:**
- Transitions between diseases with disparate outcomes
- Important for understanding disease progression risk
- May identify critical decision/intervention points

### Methodology

**Similar to high-mortality sinks, but for edges:**

**Step 1: Calculate z-scores**
```
z_edge_betweenness = (EB - μ_EB) / σ_EB
z_mort_diff = (|Mort₁ - Mort₂| - μ_diff) / σ_diff
```

**Step 2: z-product and filtering**
```
z_product = z_edge_betweenness × z_mort_diff
Include if: z_edge_betweenness > 0 AND z_mort_diff > 0 AND |Mort₁ - Mort₂| >= 0.30
```

**Additional threshold:** Minimum 30% mortality difference

**Rationale:**
- Ensures clinical significance (small differences not meaningful)
- 30% chosen based on clinical relevance (e.g., 10% vs 40% mortality)

**Step 3: Select top percentile**
```
Threshold = Quantile(z_product, 1 - 0.05) = 95th percentile
```

**Default:** Top 5% (stricter than nodes due to more edges)

---

## Statistical Methods

### Z-Score Standardization

**Purpose:** Make variables comparable across different scales

**Formula:**
```
z = (x - μ) / σ
```

**Properties:**
- Mean = 0
- Standard deviation = 1
- Preserves relative rankings

**When to use:**
- Comparing variables with different units
- Identifying "above average" vs "below average"
- Before computing products/sums of different metrics

---

### Percentile Thresholds

**Definition:** Value below which P% of observations fall

**Example:** 80th percentile = value exceeded by only 20% of data

**Advantages:**
- Non-parametric (no distribution assumptions)
- Robust to outliers
- Easy to interpret

**Disadvantages:**
- Fixed count, not fixed threshold
- Sensitive to sample size
- May change with data updates

---

### Why Use Both Percentiles and Z-Scores?

**Percentiles:** Used for **selection** (binary: outlier or not)
- Gives fixed proportion of outliers
- Easy to communicate ("top 20%")

**Z-scores:** Used for **quantification** (continuous: how extreme?)
- Magnitude of deviation
- Comparable across strata
- Enables product calculations

**Example workflow:**
1. Calculate log_ratio (continuous scale)
2. Calculate percentiles to define outliers (binary classification)
3. Calculate modified z-score to quantify extremeness (continuous ranking)

---

## Parameter Justification

### Why 20th/80th Percentiles for Outliers?

**Reasoning:**
- Not too strict (10th/90th): Captures meaningful deviations
- Not too lenient (25th/75th): Identifies clear outliers beyond normal variation
- 20%/80% captures distribution tails while maintaining statistical power
- Commonly used in outlier detection literature

**Alternatives considered:**
- 10th/90th: Too few outliers, may miss important cases
- 25th/75th: Too many outliers, less meaningful
- IQR method (Q1-1.5×IQR, Q3+1.5×IQR): Less intuitive for this application

---

### Why Top 20% for High-Mortality Sinks?

**Reasoning:**
- Balances sensitivity (finding critical diseases) vs specificity (avoiding false positives)
- Provides manageable number of targets for intervention
- Consistent with epidemiological risk stratification (quintiles)

**Sensitivity analysis:**
- Top 10%: Very stringent, may miss important diseases
- Top 30%: More lenient, more false positives
- Top 20%: Middle ground, validated in exploratory analysis

---

### Why Top 5% for Bridge Edges?

**Reasoning:**
- Edges are more numerous than nodes
- Want to identify truly critical bridges
- Top 5% gives comparable absolute number to top 20% of nodes

**Example:** 
- 200 nodes → top 20% = 40 nodes
- 1000 edges → top 5% = 50 edges

---

### Why 30% Minimum Mortality Difference?

**Reasoning:**
- Clinical significance: 30% difference is substantial
  - Example: 10% vs 40% mortality is clearly meaningful
  - Example: 20% vs 25% may not be clinically relevant
- Reduces noise from small differences
- Based on expert clinical judgment

**Sensitivity:**
- 20% threshold: Too many edges, less specific
- 40% threshold: Too few edges, may miss important transitions

---

## Reproducibility Considerations

### Stochasticity

**Sources of randomness:**
- Community detection (Louvain algorithm): Multiple runs may give different results
  - Solution: Use consistent random seed (if implemented)
  - Modularity values should be similar even if exact communities differ

**All other analyses are deterministic:**
- Network metrics: Exact calculations
- Percentiles: Fixed for given data
- Z-scores: Deterministic transformations

### Computational Precision

**Floating point considerations:**
- Use 64-bit floating point (Python default)
- Rounding only for display, not calculations
- Store full precision in intermediate steps

### Version Control

**Important for reproducibility:**
- Python version: 3.10.13 (specified in `.python-version`)
- Package versions: Pinned in `pyproject.toml`
- Algorithm versions: NetworkX, community detection library

---

## References

### Statistical Methods

- Iglewicz, B., & Hoaglin, D. C. (1993). "How to detect and handle outliers." ASQC Quality Press.
- Rousseeuw, P. J., & Croux, C. (1993). "Alternatives to the median absolute deviation." Journal of the American Statistical Association.

### Network Analysis

- Freeman, L. C. (1977). "A Set of Measures of Centrality Based on Betweenness." Sociometry, 40(1), 35-41.
- Newman, M. E. J. (2010). "Networks: An Introduction." Oxford University Press.
- Blondel, V. D., et al. (2008). "Fast unfolding of communities in large networks." Journal of Statistical Mechanics.

### Comorbidity Networks

- Hidalgo, C. A., et al. (2009). "A dynamic network approach for the study of human phenotypes." PLoS Computational Biology.
- Jensen, A. B., et al. (2014). "Temporal disease trajectories condensed from population-wide registry data covering 6.2 million patients." Nature Communications.

---

## Validation

### Internal Validation

- **Sensitivity analysis**: Vary thresholds ±10%, check robustness
- **Stratification checks**: Verify patterns consistent across sex/age groups
- **Outlier inspection**: Manual review of top outliers for clinical plausibility

### External Validation

- **Literature comparison**: Compare identified diseases with known high-risk conditions
- **Clinical review**: Expert validation of critical diseases
- **Cross-dataset**: Ideally validate on independent population (future work)
