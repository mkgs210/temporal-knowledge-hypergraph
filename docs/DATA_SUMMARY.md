# Supplied data: descriptive summary

This summary is derived from saved construction metrics and the existing raw-data quality audit. No input correction or new method experiment was performed. Source: `results/metrics.json`, `construction.snapshots[*].snapshot`, and `evidence/data_quality/issues.json`.

The input is an extracted knowledge hypergraph, not full article text. Publication-year cutoffs provide five real, retrospective availability views; they are not an ingestion log.

| Cutoff | Nodes | Hyperedges | New nodes | New edges | Arity mean / median / max | Isolated nodes |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 2022 | 2137 | 526 | 2137 | 526 | 5.827 / 3 / 59 | 21 |
| 2023 | 3376 | 820 | 1239 | 294 | 6.051 / 3 / 59 | 15 |
| 2024 | 4094 | 983 | 718 | 163 | 6.189 / 3 / 65 | 0 |
| 2025 | 4987 | 1239 | 893 | 256 | 6.123 / 3 / 65 | 0 |
| 2026 | 5798 | 1429 | 811 | 190 | 6.248 / 3 / 65 | 0 |

Initial new counts mean the initial population, not a measured update. Exact arity distributions are retained as integer values in metrics. Observation-year and edge fact/source-year distributions are retained separately; they must not be conflated.

| Node type | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: | ---: |
| article | 22 | 32 | 38 | 46 | 52 |
| author | 131 | 222 | 250 | 286 | 318 |
| cited_work | 214 | 327 | 411 | 510 | 580 |
| claim | 246 | 375 | 445 | 608 | 690 |
| component | 309 | 504 | 608 | 737 | 855 |
| dataset | 117 | 172 | 200 | 218 | 259 |
| future_topic | 67 | 102 | 121 | 155 | 178 |
| method | 189 | 294 | 352 | 408 | 448 |
| metric | 36 | 55 | 64 | 72 | 90 |
| problem | 198 | 334 | 415 | 513 | 634 |
| task | 263 | 422 | 517 | 625 | 732 |
| technique | 345 | 537 | 673 | 809 | 962 |

| Relation type | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: | ---: |
| addresses | 22 | 32 | 37 | 45 | 51 |
| authored_by | 22 | 32 | 38 | 45 | 51 |
| cites | 14 | 21 | 25 | 32 | 37 |
| claims | 225 | 360 | 445 | 608 | 690 |
| evaluated_on | 114 | 188 | 221 | 244 | 301 |
| extends | 22 | 32 | 37 | 45 | 50 |
| presents | 22 | 32 | 38 | 46 | 52 |
| proposes_future_work | 19 | 27 | 31 | 39 | 45 |
| solves | 22 | 32 | 37 | 45 | 51 |
| uses_component | 22 | 32 | 37 | 45 | 50 |
| uses_technique | 22 | 32 | 37 | 45 | 51 |

## Raw quality findings

Counts below are issue occurrences, not disjoint records. The same node/edge may contribute to several categories. Full identifiers and observed/expected values are in `evidence/data_quality/issues.csv` and `.json`.

| Issue | Occurrences |
| --- | ---: |
| edge_before_member_observation | 170 |
| fact_year_vs_source | 154 |
| first_seen_vs_sources | 127 |
| last_seen_vs_sources | 120 |
| missing_member_source | 36 |
| origin_after_observation | 3 |

For example, `metr_00002` has a declared first-seen year 2017 while its source-based earliest year is 2018; `metr_00004` has 2021 versus 2024. The derived views admit a node no earlier than both its observation year and an available declared source. They require the original complete endpoint set before admitting an edge. They do not rewrite the raw export or recover historically unavailable wording.

Different edge fact/source years can be legitimate retrospective references; every discrepancy is reported rather than equated automatically with a false scientific statement. Undated aliases and final metadata are excluded from historical inputs.
