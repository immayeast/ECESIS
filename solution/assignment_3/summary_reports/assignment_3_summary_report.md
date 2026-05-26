# Bus Mapping via Network Structure:

## Purpose of the latitude and longitude method

My initial approach for Assignment 3 is to use latitude and longitude as the first stage of bus matching. Since buses represent physical electrical locations, two records that describe the same bus or substation should usually appear close to each other geographically. This makes geographic proximity a practical way to narrow the search space before applying more detailed matching logic.

The purpose of this method is not to assume that the nearest point is always the correct match. Instead, latitude and longitude are used as a candidate-generation strategy. For each Dayzer bus, I search for nearby Panorama buses within a configurable radius and keep the nearest Panorama buses as fallback candidates. After those candidate pairs are created, the match is refined using voltage, name similarity, and network topology.

This approach is useful because it turns a very large matching problem into a smaller ranking problem. Rather than comparing every Dayzer bus to every Panorama bus, the method first asks: which Panorama buses are physically plausible matches?

## Why geographic proximity is useful

Latitude and longitude provide strong physical evidence. Bus names may differ across systems because of abbreviations, punctuation, station naming conventions, or different modeling practices. Coordinates can help cut through some of that naming inconsistency.

For example, two buses may have slightly different names but be located at nearly the same point and share the same voltage level. In that case, geographic proximity gives an important signal that the records may refer to the same physical asset.

The coordinate method also supports a clear audit trail. For each candidate pair, the output can show the distance in kilometers, the voltage comparison, and the final score. This makes the matching process easier to explain and review manually.

## Coordinate EDA before matching

Before relying on latitude and longitude, I added an EDA step to check whether the coordinate fields are complete and reliable enough to support spatial matching.

The coordinate diagnostics showed:

| Dataset | Rows / Count | Notes |
|---|---:|---|
| Dayzer buses | 10,799 | 93.69% latitude/longitude coverage |
| Panorama buses | 10,233 | 100.00% latitude/longitude coverage |
| Dayzer branch endpoints | 27,242 | 27,242 resolved to known Dayzer buses |
| Panorama branch endpoints | 22,534 | 22,534 resolved to known Panorama buses |
- Matched Dayzer buses: 5,982
- Unmatched Dayzer buses: 4,817
- High confidence: 865
- Medium confidence: 418
- Low confidence: 1,467
- Ambiguous: 8,049
- Ambiguity rate: 74.5%

The coordinate EDA identified 681 Dayzer buses with missing latitude and longitude. No invalid non-missing coordinate ranges were found.

This means latitude and longitude are useful for most buses, but not for every bus. The missing Dayzer coordinates create a harder matching condition because those buses cannot be matched using spatial proximity as the primary filter.

## Concern about missing values

The main limitation of a latitude/longitude-first method is that it depends on coordinate availability and quality. If a Dayzer bus is missing latitude or longitude, the spatial search cannot identify nearby Panorama candidates. In those cases, the model must fall back to weaker signals such as name similarity and voltage level.

Missing coordinates can make matching harder for several reasons:

- The true match may not be included in the spatial candidate set.
- Name similarity may be weak if the two systems use different abbreviations.
- Multiple buses can share the same voltage level and similar station names.
- Dense substations can contain several buses very close together, creating ambiguity.
- Some buses may be modeled differently between Dayzer and Panorama, such as one system splitting a bus into several records while the other aggregates it.

Because of these issues, I do not want the project to rely only on longitude and latitude. The coordinate method is a strong starting point, but it should be supported by an alternative approach for missing or uncertain cases.

## Alternative approach for missing or difficult cases

For buses with missing coordinates, or for buses where the spatial result is ambiguous, I plan to explore a graph-aware alternative. This is the alternative Codex suggested earlier: use the network structure itself to help infer matches.

In this approach:

- Buses are treated as graph nodes.
- Branches are treated as graph edges.
- High-confidence matches are found first using name, voltage, and available coordinate evidence.
- Those high-confidence matches become seed matches.
- The algorithm then looks at neighboring buses in the Dayzer and Panorama graphs.
- If a Dayzer bus connects to neighbors that already match the neighbors of a Panorama bus, that becomes evidence that the two unresolved buses may correspond.

This method is useful because power grids are networks, not isolated points. Even when a bus is missing latitude and longitude, its electrical neighborhood can still provide identifying information.

## Why this alternative is a better next step

The graph-aware method is a better way to handle missing coordinate cases because it does not require every bus to have complete geographic metadata. Instead, it uses relationships between buses. A bus can still be identifiable if its connected neighbors, voltage profile, and local topology resemble a candidate bus in the other dataset.

This is especially relevant for Assignment 3 because the assignment emphasizes network structure. Latitude and longitude help identify where a bus is, but graph structure helps identify how a bus behaves within the electrical system.

My plan is therefore:

1. Use latitude and longitude as the first-stage candidate-generation method where coordinates are available.
2. Use voltage and name similarity to rank those candidates.
3. Use graph topology to refine the ranking.
4. For missing-coordinate or ambiguous cases, rely more heavily on graph-aware matching.
5. Later, explore a seed-and-propagate network alignment method where confident matches help resolve harder unmatched buses.

## Methodology

### 1. Normalization

Bus names were normalized by lowercasing, removing punctuation, replacing separators with spaces, and standardizing whitespace. Numeric identifiers and voltage-like tokens were intentionally preserved because over-normalization can collapse distinct physical assets.

### 2. Seed Generation

Strict seed matches were selected using strong name similarity, exact or near-exact voltage compatibility, close coordinates, and uniqueness constraints. These seeds act as graph anchors. In the implemented run, 49 strict seeds were accepted before propagation.

### 3. Graph Construction

Separate Dayzer and Panorama graphs were built from the branch lists. Local graph fingerprints include degree, neighbor count, neighbor voltage profile, and neighbor name-token structure.

### 4. Topology-Aware Propagation

The propagation model asks whether already-matched Dayzer neighbors correspond to Panorama neighbors for a candidate pair.

```text
matched neighbors
  -> neighborhood consistency evidence
  -> stronger candidate ranking for unresolved nearby buses
```

For each candidate pair, the model computes propagation support count, propagation support ratio, reverse support, propagation consistency score, and final propagation score. Candidates are iteratively accepted when they have compatible voltage, enough graph support, a strong propagation score, and adequate separation from the next candidate.

### 5. Geographic and Voltage Validation

Geography is used for candidate filtering, tie-breaking, and plausibility validation, but not as the sole definition of identity. Voltage acts as a physical feasibility constraint: incompatible voltage levels are heavily penalized.

### 6. Conflict Resolution

The final mapping enforces one-to-one consistency. If multiple Dayzer buses claim the same Panorama bus, lower-scoring duplicates are flagged as conflicts or left unmatched rather than being accepted. This prevents high-density substations from collapsing many Dayzer buses onto one Panorama bus.

## No Ground Truth Evaluation Strategy

No labeled ground truth is available, so the evaluation does not claim true accuracy. Instead, it measures internal consistency and operational defensibility.

The diagnostics assess:

- voltage compatibility,
- duplicate Panorama assignment rate,
- topology support,
- geographic plausibility,
- confidence labels,
- ambiguity rates,
- manual spot-check samples.

The graph-aware method intentionally prioritizes fewer bad automatic matches over maximizing match count.

## Baseline vs Graph-Aware Diagnostics

The pure latitude/longitude nearest baseline performs poorly on operational consistency metrics. It assigns every Dayzer bus to a nearest Panorama candidate, but produces many duplicate Panorama claims and many voltage conflicts.

| Diagnostic | Pure Lat/Lon Nearest | Full Graph-Aware Final |
|---|---:|---:|
| Matched buses | 10,799 | 5,982 |
| Duplicate Panorama assignments | 7,102 | 0 |
| Exact voltage match rate | 54.2% | 98.8% |
| Voltage conflict rate > 1 kV | 42.2% | 0.18% |
| Median distance km | 1.84 | 3.49 |
| 90th percentile distance km | 6.43 | 12.28 |

The graph-aware model accepts fewer automatic matches, but the accepted matches are much cleaner by voltage and assignment consistency.

## Ablation Study

The ablation study compares progressively richer matching variants.

| Variant | Matched | Duplicates | Exact Voltage | Voltage Conflict > 1 kV | Median Distance km | Ambiguity Rate |
|---|---:|---:|---:|---:|---:|---:|
| Pure lat/lon nearest | 10,799 | 7,102 | 54.2% | 42.2% | 1.84 | 86.2% |
| Name + kV only | 10,799 | 4,220 | 97.6% | 2.3% | 7.42 | 76.2% |
| Name + kV + lat/lon | 10,799 | 5,111 | 97.2% | 2.7% | 2.71 | 89.4% |
| Graph-aware without propagation | 6,285 | 0 | 98.9% | 0.4% | 3.40 | 73.7% |
| Full graph-aware final | 5,982 | 0 | 98.8% | 0.18% | 3.49 | 74.5% |

The ablation shows the value of each layer. Name and voltage drastically reduce voltage conflicts compared with pure geography. Graph-aware scoring and one-to-one resolution eliminate duplicate assignments. Propagation adds explicit neighborhood-consistency evidence and produces accepted graph-supported matches while maintaining zero duplicate final Panorama assignments.

## Sensitivity Analysis

Sensitivity was tested over radius, topology weight, and confidence threshold.

Key observations:

- Increasing radius from 5 km to 20/50 km improves candidate coverage and voltage match rate, but still requires downstream duplicate and ambiguity controls.
- Increasing topology weight from 0.10 to 0.30 increases high-score counts but also raises duplicate pressure and voltage conflict rate in the unpropagated sensitivity view.
- Raising the confidence threshold from 0.65 to 0.85 reduces accepted high-confidence count from 1,288 to 878 while improving exact voltage match rate and eliminating voltage conflicts in that high-confidence subset.

| Setting | Matched / High Count | Duplicate Assignments | Exact Voltage | Voltage Conflict > 1 kV |
|---|---:|---:|---:|---:|
| radius 5 km | 9,116 | 4,420 | 86.5% | 12.7% |
| radius 20 km | 10,766 | 4,487 | 96.7% | 2.7% |
| radius 50 km | 10,799 | 4,490 | 97.0% | 2.5% |
| topology weight 0.10 | 10,799 | 4,456 | 97.4% | 2.4% |
| topology weight 0.30 | 10,799 | 4,576 | 95.5% | 3.6% |
| confidence threshold 0.85 | 878 | 0 | 98.5% | 0.0% |

The most quality-sensitive controls are voltage feasibility and confidence thresholding. Radius mainly affects candidate availability; it does not solve duplicate or voltage-conflict risk by itself.

## One-to-One Assignment Diagnostics

| Metric | Value |
|---|---:|
| Top-candidate duplicate Panorama assignments before resolution | 4,490 |
| Baseline duplicate assignments after baseline resolution | 0 |
| Lower-scoring duplicates marked conflict or unmatched | 2,993 |
| Final duplicate nonblank Panorama assignments | 0 |

One-to-one matching matters because without it, high-density substations may collapse many Dayzer buses onto the same Panorama bus. The conflict output is not simply an error list; it is a useful review queue for areas where the two systems may split or aggregate buses differently.

## Topology Contribution Analysis

| Diagnostic | Value |
|---|---:|
| Accepted topology-supported matches | 1,059 |
| Matched cases with high topology and low/moderate name score | 4,575 |
| Matched candidates changed from lat/lon nearest | 4,528 |
| Matched small-gap, high-topology cases | 2,378 |

Topology is not merely an extra feature. It serves as neighborhood consistency evidence: if a candidate bus connects to already-matched neighbors in a compatible way, the match becomes more defensible even when names are abbreviated or coordinates are not decisive.

## Distance and Voltage Diagnostics

Accepted final matches have:

- mean distance: 4.93 km,
- median distance: 3.49 km,
- 90th percentile distance: 12.28 km,
- maximum distance: 31.57 km,
- high-confidence median distance: 2.97 km.

Voltage consistency by confidence:

| Confidence | Exact Voltage Match Rate | Voltage Conflict > 1 kV |
|---|---:|---:|
| High | 100.0% | 0.0% |
| Medium | 91.9% | 0.24% |
| Low | 99.5% | 0.14% |
| Ambiguous | 99.0% | 0.25% |

Voltage conflicts are rare in the final output, which is a major improvement over pure geospatial nearest matching.

## Manual Review Design

The extended manual review file contains 300 rows:

- 50 high-confidence matches,
- 50 medium-confidence matches,
- 50 ambiguous matches,
- 50 low-confidence or unmatched cases,
- 50 topology-supported matches,
- 50 cases where the lat/lon baseline disagrees with the graph-aware final result.

Manual label convention:

- `1` = likely/correct match,
- `0` = incorrect match,
- `-1` = uncertain.

This creates a path toward supervised calibration without requiring ground truth up front.

## Case Studies

### A. High-confidence clean match

`LPNSW_1_2` matched to `LPNSW_1KV_2`. The voltage matches at 1.0 kV, the name score is strong, topology score is high, and the propagation support count is 3. The nearest-coordinate baseline selected `DESSW_138KV_1`, which is electrically implausible.

### B. Lat/lon baseline failure fixed by graph-aware model

The same `LPNSW_1_2` case shows why geography alone is insufficient. The nearest-coordinate candidate was not the graph-aware final match and had incompatible voltage. The graph-aware model selected a voltage-compatible candidate with stronger neighborhood consistency.

### C. Topology-rescued weak-name match

`VICTPORT_8` matched to `VICTPORT_138KV_1`. Name similarity was moderate, but voltage matched at 138 kV and topology score was high. This is the kind of case where network structure adds evidence that fuzzy matching alone may underweight.

### D. Ambiguous unresolved case

`BTE_2_5` had a plausible candidate `BTE_345KV_2`, with matching 345 kV and good topology, but remained flagged. The ambiguity reflects dense local alternatives and the need for manual review rather than forced certainty.

### E. Voltage-conflict case rejected by graph-aware model

For `CLO_C2`, the nearest-coordinate baseline selected `PRO_13KV_3`, a voltage-conflicting candidate. The graph-aware result avoided accepting that nearest-coordinate candidate as a clean final match and instead selected a voltage-compatible candidate or left the case for review depending on confidence.

## Limitations

- No ground-truth labels are available, so diagnostics measure plausibility rather than true accuracy.
- Coordinate errors and missing coordinates can affect candidate generation.
- Branch modeling may differ between Dayzer and Panorama.
- Some buses may be split in one system and aggregated in the other.
- Propagation can amplify bad seeds if seed thresholds are too loose.
- Local topology is informative, but it is not a complete global graph isomorphism solution.

## Final Conclusion

I decided not to treat longitude and latitude as the full solution. Instead, I use them as the first layer of a broader matching framework and plan to explore a graph-aware alternative for the more difficult cases. This better matches the purpose of the assignment: mapping buses not only by individual attributes, but also by their position within the electrical network.

Graph-aware matching performs better than latitude/longitude alone by operational consistency metrics. It reduces duplicate assignments from thousands to zero, drastically reduces voltage conflicts, and adds topology-supported evidence that names and coordinates alone cannot provide.

The method intentionally prioritizes defensible matches over maximizing match count. That is appropriate for infrastructure data integration: a smaller set of clean, auditable matches is more useful than a larger set of fragile nearest-neighbor assignments. Network structure provides the key additional evidence needed to align buses across Dayzer and Panorama in a way that is technically and operationally defensible.