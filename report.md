# Bounded Semantic Abstraction of an Evolving Knowledge Hypergraph

## Abstract

We study a bounded hierarchy over extracted scientific records, combining sentence similarity with native multiway relations. An initial balanced partition is followed by persisted, budgeted updates; a counted quotient retains the endpoints and provenance of each original hyperedge. The study separates structural correctness, held-out relational coherence, temporal stability, label faithfulness and retrieval utility. On 5798 records, the principal comparison uses 14 development questions and 63 target occurrences. Graph-assisted BM25 retrieves 28 targets in its first thirty results; adding hierarchy transitions retrieves 28. These numbers measure this run rather than reproduce a previous encoder's results. We also compare actual coarse-first traversal with a flat baseline. Publication-year views, dependent sources and incomplete full-text evidence limit interpretation; unsuccessful tests remain visible.

<!-- ARTICLE BODY -->

## 1. Problem and position

Scientific knowledge arrives as methods, tasks, components, datasets and extracted statements, linked by relations involving several records. A useful abstraction must expose a manageable number of themes while preserving a path to each original record. New papers should extend retained state rather than trigger an independent reconstruction of every previous partition. The research question is whether such an abstraction supplies useful retrieval candidates, subject to hard structural constraints and an explicit cost for changing past assignments.

Sentence-BERT motivates reusable sentence representations and cosine comparison [1]; MiniLM supplies the compact transformer architecture behind our selected checkpoint [2]. We use a pretrained Sentence Transformers model without fine-tuning. Mt-KaHyPar supplies native hypergraph partitioning and fixed-vertex refinement [3]. These are established primitives: our contribution is their explicit integration with chronological state, counted multiway contraction and controlled evaluation. Neither a good cut objective nor a small model proves that a group is a meaningful scientific topic.

Four methodological papers frame the design. Gottesbüren et al. [5] combine multilevel coarsening and native connectivity refinement; we use that solver primitive while imposing our own nested display limits and persisted assignments. Loukas [6] derives restricted spectral and cut guarantees for ordinary-graph reduction. Our quotient instead preserves occupied-part hyperedge connectivity under legal further contraction; we do not inherit those spectral guarantees. Leiden [7] refines communities to guarantee connectivity in its graph setting. Its resolution parameter is useful for exploring scales, but does not itself enforce our fixed counts or native multiway objective. LD-Leiden [8] updates repaired affected regions while retaining aggregation state on dynamic graphs. We adopt the motivation for retained state, not its implementation or graph-modularity guarantees: our editable paths are bounded, while similarity and incidence work may still inspect old records. These distinctions explain why native contraction, semantic evidence and temporal accounting are separate components rather than properties inferred from the partitioner's name.

Credible alternatives include embedding-only clustering, an ordinary-graph projection followed by community detection, and a hierarchy used solely to browse a flat index. Embedding-only clustering leaves supplied multiway relations unused. Projection changes the influence of a relation when it replaces one hyperedge with many pairwise edges. We therefore retain native incidences. The browsing alternative remains a practical deployment option if traversal fails to improve retrieval. We make no claim that this particular combination is a globally novel or optimal algorithm.

## 2. Formal objective and construction

At cutoff t, let V_t denote visible records and E_t their visible hyperedges. Each level partitions V_t: every record has one group and every fine group has one coarse parent. The display limits are 12 coarse and 50 fine groups in total. The finest exported level contains the original records. These constraints are independently checked, rather than inferred from successful solver execution.

For an assignment pi, let lambda_e(pi) be the number of distinct groups occupied by edge e. The structural cost is the sum of w_e times (lambda_e minus one). Semantic neighbour edges have cosine-derived weights. Native scientific relations are normalized within source papers and scaled by gamma=0.03. Provenance-only relations, including citations, have zero construction cost. Article and author containers are attached after scientific partitioning, so their high incidence alone does not determine scientific topics.

For each optimized level, the criterion can be written as follows, where D counts changes from the saved assignment:

$$
J_{t,\ell}=\sum_e w_e(\lambda_e-1)+\eta_t D_{t,\ell}.
$$

The optimization is conditional on fixed outside paths and level nesting. Fine and coarse refinements are staged; the algorithm does not solve an unconstrained joint global optimum. At initialization there is no temporal penalty. Every accepted refinement is checked against its floating-point cost and the reduced counted problem.

Per-paper normalization limits the influence of papers with many extracted relations. Gamma is a fixed historical development operating point, reused without answer-driven tuning for this encoder; it is not a proven optimal trade-off. Its role is explicit and auditable, while sensitivity to other weights remains future work.

The initial builder recursively splits the largest remaining region with the library's native connectivity objective, stopping at the display budgets. Size guards discourage tiny outliers and one giant residual group. Fine groups remain nested inside their coarse parents; a subsequent counted-quotient refinement may adjust coarse assignments while preserving legal nesting and nonempty regions. These guards are engineering choices, not a scientific ontology: a genuinely large field can be divided to satisfy them.

Exact neighbour construction initially compares sentence vectors across the visible scientific records. The stored affinity graph is sparse, but its initial computation is quadratic in record count. Recursive cuts are heuristic; no global joint optimum or semantic guarantee follows. This distinction matters when interpreting a structurally valid but scientifically mixed group.

![Saved hierarchy excerpt](figures/cpu_hierarchy.png)

**Figure 1.** Actual final-cutoff branch, selected by persistent IDs rather than quality. Arrows show membership from coarse group through fine group to an original record. Most branches are omitted; long texts may be shortened in the illustration. Generated names are not expert topic judgments.

## 3. Incremental state and multiway collapse

For later years, the builder reloads its previous checkpoint. New records initially join the nearest existing scientific record's group. New-to-existing similarities are calculated while old-to-old affinity entries and old vectors are retained. The updater selects an affected frontier and fixes assignments outside it. Changes consume one common path budget, including a changed coarse parent, with a maximum fraction 10% of old scientific records per update. A movement penalty discourages gratuitous changes inside the permitted region.

Persisted group identities and event logs distinguish initialization and subsequent growth. The chosen configuration retains a fixed set of group slots; adaptive birth/merge/split proposals are not enabled in this run. Stable identities therefore establish continuity, not discovery of every emerging scientific field. Repeated runs can reuse exact-text embedding cache entries, and changed text is a different cache input. This cache is an optimization over a frozen pretrained encoder, not model training.

If all endpoints of a hyperedge enter one coarse group, it becomes an internal edge with its original endpoint multiplicity. If only some endpoints meet, they become a counted endpoint alongside the other groups. An edge spanning several groups remains multiway. The quotient retains original edge identity, attributes, time and provenance. Consequently it preserves occupied-part connectivity cost under further legal grouping. It loses the individual internal layout at the coarse view; original membership lists provide drill-down. An independent audit compares exported quotients with direct contraction and checks endpoint totals.

Update locality is bounded in editable assignments, not in all computation. With m new records and n existing records, exact new-to-all similarities still cost proportional to m times n times vector dimension. Native normalization and incidence processing can scan old relations, and refinement has data-dependent cost. We report the preserved paths and counted work rather than claim constant-time insertion or a measured production speedup.

## 4. Data and evaluation design

We use the supplied extracted graph, not full articles. A record is admitted only when its declared observation/source years permit it; an edge requires all endpoints and its fact/source years to be visible. Raw date and provenance contradictions are not silently repaired. These retrospective publication-year views are not a historical ingestion log, and year-level dates cannot establish availability by a particular month. The encoder can also contain knowledge learned before its release; cutoff-safe inputs do not certify time-isolated model pretraining.

| Year | Records | Native edges |
| --- | --- | --- |
| 2022 | 2137 | 526 |
| 2023 | 3376 | 820 |
| 2024 | 4094 | 983 |
| 2025 | 4987 | 1239 |
| 2026 | 5798 | 1429 |

The encoder is sentence-transformers/all-MiniLM-L6-v2, with its exact revision and normalization stored in each run. Construction and semantic retrieval use the same own-name representation. Evaluation never calls similarity under that encoder independent semantic validation: that would reuse the geometry optimized by construction. Instead, we evaluate within-citation endpoint-pair retention. Citation incidences have zero construction weight and are not consulted by neighbour selection or metadata attachment. Their linked records' names remain construction inputs, so the result is a held-out relational proxy rather than full-text semantic ground truth.

The citation score averages within each source paper before averaging papers. A type-stratified permutation null preserves group sizes and each group's type composition. We publish the observed score, null mean, excess and conditional permutation probability. No significance threshold selects parameters, and the small permutation reference is exploratory. In particular, the score must not be read as a percentage of correct scientific groups.

| Level | Citation score | Null mean | p (null) |
| --- | --- | --- | --- |
| Coarse | 0.270 | 0.221 | 0.048 |
| Fine | 0.096 | 0.059 | 0.048 |

Perturbation robustness removes the integer floor of one tenth of the initial view's native edge count under five fixed deletion seeds and rebuilds the unchanged initial algorithm. The resulting ARIs summarize sensitivity to that randomization; their Student-t interval is conditional on the fixed initial graph and vectors. They do not constitute a temporal-process confidence interval. Across actual years, shared-record ARI is supplemented by source-component bootstrap diagnostics and a type-stratified random-correspondence significance reference. When source overlap leaves too few independent components, the process interval is unavailable rather than fabricated.

| Transition / level | ARI | Conditional CI | p correspondence |
| --- | --- | --- | --- |
| 2022–2023 / 0 | 1.000 | unavailable | 0.001 |
| 2022–2023 / 1 | 0.996 | unavailable | 0.001 |
| 2023–2024 / 0 | 1.000 | unavailable | 0.001 |
| 2023–2024 / 1 | 0.990 | unavailable | 0.001 |
| 2024–2025 / 0 | 1.000 | unavailable | 0.001 |
| 2024–2025 / 1 | 0.997 | unavailable | 0.001 |
| 2025–2026 / 0 | 1.000 | unavailable | 0.001 |
| 2025–2026 / 1 | 0.997 | unavailable | 0.001 |

| Initial deletion | Mean ARI | Conditional 95% |
| --- | --- | --- |
| Coarse | 0.480 | [0.411, 0.549] |
| Fine | 0.507 | [0.496, 0.518] |

## 5. Labels and faithfulness

The labeller receives each group's exact saved evidence: scientific member names, generation-side member statements and explicitly separated neighbouring claim context. Source-block splitting reserves claim sentences for checking; complete nonclaim scope remains visible so naming does not hide a minority material. The prompt asks for a scientific heading, one explanatory sentence and an explicit coherent/mixed/insufficient status. Generated status is the generator's opinion, not an evaluation result. Requests use schema-constrained JSON and small batches; neither word-count preferences nor punctuation warnings are treated as scientific failures.

A different checkpoint judges a fixed hash-selected sample, blinded to group identity and generator status. The report distinguishes accurate, vague, wrong and insufficient-evidence ratings and records the sample denominator. Sensitivity controls precede the sample. If the controls fail or the run is incomplete, its apparent overclaim fraction is labelled unreliable. Nonclaim names overlap construction and generation, making name-only coverage circular; only withheld claim text supplies a separate factual premise. Groups without such premises are counted explicitly. Both checkpoints are from the Nemotron family, limiting checker independence. This control is narrower than expert validation from full papers, which are unavailable here.

The rubric and synthetic controls are development checks, not unseen calibration data. Earlier model-only failures remain archived. Strictly empty member-and-claim evidence is assigned insufficient-evidence by a deterministic rule, recorded separately from model responses; this rule does not adjudicate nonempty scientific evidence or silently replace a model's substantive rating.

310 group labels were authenticated. The fixed audit sample contains 20 groups; 19 have withheld claim evidence. Labeller: nvidia/nemotron-3.5-lightning-30b-a3b; checker: nvidia/nemotron-3-ultra-550b-a55b.

| Audit outcome | Count |
| --- | --- |
| accurate | 16 |
| vague | 4 |
| wrong | 0 |
| insufficient_evidence | 0 |

Sensitivity controls passed: True. 1 empty-evidence assessments used an explicit deterministic rule, not a model call. Observed wrong fraction in the checked sample: 0.000 over 20 rated labels. These are model-based sample ratings, not population accuracy or human truth.

## 6. Matched retrieval results

Questions and reference answers are development data already inspected during the project. Predictions are serialized and hashed before the evaluator opens the reference file in this run; this prevents direct answer-driven prediction code, but does not manufacture an untouched final test. Methods are matched by normalized exact surface form or explicit graph alias. Unmapped references remain in the denominator, with their identities retained for audit. Reference lists are incomplete relevance judgments, so we do not interpret unlisted results as proven false positives.

The primary outcome is micro Recall@30 across Type A target occurrences. For both lexical and lexical/dense seeds, we compare the same native/source walk with and without added group transitions. This pair isolates the hierarchy overlay; it is distinct from tree traversal. The tree arms first select coarse groups and their fine children, then score and walk within the induced region. The all-branch arm is the corresponding traversal reference. Saved traces count evaluated groups and leaves; the experiment also computes flat comparators, so reduced per-arm leaf work is not an end-to-end benchmark speed claim.

| Search | Found / targets | Recall@30 |
| --- | --- | --- |
| BM25 | 16/63 | 25.4% |
| BM25 + MiniLM | 8/63 | 12.7% |
| BM25 + MiniLM + native/source | 22/63 | 34.9% |
| BM25 + MiniLM + native/source + hierarchy | 22/63 | 34.9% |
| BM25 + native/source | 28/63 | 44.4% |
| BM25 + native/source + hierarchy | 28/63 | 44.4% |
| MiniLM | 7/63 | 11.1% |
| Tree 2 coarse / 8 fine | 8/63 | 12.7% |
| Tree 8 coarse / 32 fine | 22/63 | 34.9% |
| Tree all branches | 22/63 | 34.9% |

| Comparison | Delta recall | Paired 95% |
| --- | --- | --- |
| BM25 + MiniLM + native/source | +0.000 | [-0.037, 0.047] |
| BM25 + native/source | +0.000 | [0.000, 0.000] |
| coarse_first_2_8 | -0.222 | [-0.386, -0.119] |
| coarse_first_8_32 | +0.000 | [-0.132, 0.081] |

Equal hit totals can conceal replacements: we retain gained and lost target identities as well as aggregate differences. Paired question bootstrap intervals condition on these inspected questions and fixed method choices. They do not correct development selection or imply equivalence when they include zero. A useful hierarchy may recover a missed target and still harm total recall by excluding several others.

![Retrieval quality](figures/cpu_retrieval.png)

**Figure 2.** Recall@30 from this run's predictions and common evaluator. The horizontal scale spans the full possible recall range.

The study does not establish a reliable retrieval gain from the hierarchy. We would retain the stronger flat/native-source search as the retrieval default and expose the bounded hierarchy for inspection while treating traversal as experimental. This is a measured negative result for the tested configurations; the comparison was performed and no positive utility is claimed.

## 7. Limits and next work

Two predeclared component controls reuse the exact CPU vectors, cutoffs, budgets and search. The zero-native-cost control removes the scientific hyperedge cut term; provenance attachment and retrieval relations remain. Unbounded refinement removes the old-scientific edit cap and movement penalty, while retaining warm starts, fixed topic slots and saved affinities. Its initial partition is byte-identical to the full method. These are conditional component comparisons, not a new search for a winning configuration.

| Variant | Citation excess (fine) | Mean yearly ARI (fine) | Tree hits@30 |
| --- | --- | --- | --- |
| Full | 0.038 | 0.995 | 22/63 |
| No native cost | 0.048 | 0.995 | 22/63 |
| Unbounded edits | 0.036 | 0.902 | 22/63 |

Citation excess is the final fine-level score minus its matched null. Yearly ARI is the descriptive mean across shared-old-record transitions, not independent replications. Tree retrieval uses the same 8-coarse/32-fine route in every row. Per-cutoff scores, five deletion rebuilds per control, movement audits, overlays and paired question intervals are saved in metrics. Changed controls receive no transferred labels or faithfulness scores; T5 applies only to the full method. We retain the preselected full method for bounded edits, without claiming that these controls establish a universally optimal trade-off. Removing the native term does not lower fine citation excess or routed recall in this run; its inclusion therefore has no demonstrated advantage on these point estimates. Native relational fidelity remains an explicit design requirement, not a quality claim inferred from this comparison. Removing the edit cap and penalty lowers chronological agreement. This supports their role in retaining old memberships, but imposed stability does not establish scientifically correct adaptation.

Hard budgets, laminarity, cutoff membership, checkpoint persistence and counted contraction are executable invariants. Topic meaning, informative labels and useful retrieval are empirical outcomes. The study cannot claim that preserving old paths makes those paths correct, or that a low cut cost establishes a scientific taxonomy. Source dependence also limits confidence statements, and availability views do not reconstruct exact extraction history.

No untouched final test is available; these questions were inspected during development.

Citation coherence is a relational proxy; full-text semantic ground truth is unavailable.

Perturbation robustness covers the initial view; process-level temporal confidence remains unavailable.

With four more weeks, we would collect independently annotated topic and relevance judgments, improve entity/alias matching under a shared predeclared evaluator, test context representations with full-paper evidence where licensed, and evaluate adaptive topic events on a genuinely later stream. The highest priority would be diagnosing lost retrieval targets and whether routing should retain multiple representative vectors per topic. These are future experiments, not improvements credited to the present run.

The repository regenerates numerical tables and figures from metrics, validates deliberately broken fixtures and records prompts, raw model responses and source hashes. Procedural scientific-writing and critical-thinking skills informed reporting and are disclosed [4]. Their use is not evidence that our method is valid. The accountable submitter must review the generated article and the explicit outstanding requirements before submission.

## References

[1] Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP. https://aclanthology.org/D19-1410/

[2] Wang, W. et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. https://arxiv.org/abs/2002.10957

[3] KaHyPar project. Mt-KaHyPar: native balanced graph and hypergraph partitioning, fixed vertices and deterministic refinement. https://github.com/kahypar/mt-kahypar

[4] Kassis, T., Agarwal, V., He, Y., Patel, D. and Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065

[5] Gottesbüren, L., Heuer, T., Maas, N., Sanders, P. and Schlag, S. (2023). Scalable High-Quality Hypergraph Partitioning. https://arxiv.org/abs/2303.17679

[6] Loukas, A. (2019). Graph Reduction with Spectral and Cut Guarantees. JMLR 20(116), 1–42. https://jmlr.org/papers/v20/18-680.html

[7] Traag, V. A., Waltman, L. and van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports 9, 5233. https://doi.org/10.1038/s41598-019-41695-z

[8] Bokov, G. et al. (2026). LD-Leiden: Local Parallel Community Detection in Large Dynamic Networks. Revised preprint, v2. https://arxiv.org/abs/2502.18497
