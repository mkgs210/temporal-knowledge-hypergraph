# Bounded Native Semantic Hierarchies for an Evolving Knowledge Hypergraph

## Abstract

Scientific hypergraphs need compact, meaningful views that retain multiway relations as new literature arrives. We combine frozen text affinities with native hyperedge connectivity, construct nested partitions under permanent display budgets, and update saved state through a bounded old-record frontier. Counted contraction preserves native endpoint multiplicities; fixed terminals and temporal anchors constrain local refinement. Adaptive births, donor merges and genuine old-topic splits are separately controlled interventions. We evaluate independent-family coherence against a matched null, actual chronological changes, edge-deletion rebuilds, explanatory-label support and paired retrieval. The experiments establish structural invariants and expose limits of semantic interpretation and routing. Positive downstream benefit and expert-validated label faithfulness remain open; a stable or well-formed hierarchy alone does not settle either question.

**Keywords:** temporal hypergraphs; semantic abstraction; native coarsening; bounded updates; evaluation validity.

<!-- ARTICLE BODY -->

## 1. Problem and prior work

A useful abstraction must represent scientific ideas while preserving structure and history. We seek coarse and fine partitions C and F, followed by original-node singletons, with each fine group assigned one coarse parent. The hard limits are twelve coarse groups and fifty fine groups in total. They persist after every update; adding papers never increases them.

Native multilevel partitioning supplies optimization primitives [1]. Graph coarsening with spectral/cut guarantees [2] motivates stating exactly what contraction preserves, without transferring ordinary-graph theorems to this hypergraph. Leiden [3] motivates explicit refinement and cautions against assuming connectivity from an optimization score. LD-Leiden [4] maintains affected regions and hierarchy state for dynamic graphs; our bounded path edits and multiway criterion address different constraints. Projected Leiden is a credible simpler alternative, but projection changes the native cost. Unrestricted rebuilding and semantic-only grouping respectively omit localized persistence and native relations.

## 2. Data and hierarchy construction

We use the supplied export, with cumulative publication-availability cutoffs. A hyperedge enters only when its fact/source years and every original endpoint are visible. Raw records remain unchanged. Historical wording and ingestion order are unavailable, and date/provenance contradictions limit temporal interpretation. Full article bodies are absent. Scientific partitioning uses entity, task, claim and related records; article/author records receive metadata paths rather than drive topic formation.

**Table 1.** Snapshot size, native arity and growth. Counts concern the derived cutoff-safe views, not reconstructed ingestion history.

{{SNAPSHOTS}}

### 2.1 Semantic and native criterion

Frozen E5-base-v2 vectors encode each record's own surface form. Initial positive-cosine top-fifteen affinities are symmetrized and normalized by their initial total. Later updates append new-node queries while preserving old-old affinities. Thus the index retains history rather than recomputing a global neighbor graph.

For partition P, the native stages use

$$
J_t(P)=\sum_{u<v}a_{uv}\mathbf1[P(u)\ne P(v)]+\sum_e b_e[\lambda_e(P)-1].
$$

Here lambda counts distinct occupied parts of a scientific hyperedge. Eligible edges from each paper share equal total native weight, scaled by gamma and the initial eligible-paper count. Provenance-only relations have zero scientific cut weight but remain in exported original-edge accounting. Semantics can connect related names across sparse papers; native cost discourages fragmenting a joint relation. Gamma is a declared development choice, not an independently proven optimum or a fixed percentage of influence as channel masses grow.

Initial construction repeatedly bisects the largest region using the library's native objective, first to the coarse budget and then to the fine budget. A child holds at most `ceil(.55 * parent_size)` scientific records. Children inherit their coarse parent; counted quotient refinement then revisits coarse placement. These size guards prevent degenerate allocations but can divide a genuinely large scientific field. Recursive solves are a heuristic, not a joint optimum over hierarchy levels.

![Native hierarchy, counted edges and bounded update flow](figures/repair_structure.png)

**Figure 1.** Original records remain the leaves. Fine groups inherit coarse parents; original hyperedges retain counted multiway endpoints. Updates reuse previous state and expose only a budgeted region to movement.

### 2.2 Collapse actually used by optimization

Each original edge retains its ID, relation and provenance, with count c(e,g) of endpoints landing in group g. If all n endpoints share one group, an internal edge with count n remains. If only m share a group, that group receives count m and other occupied groups retain their counts. Three or more occupied groups remain a multiway edge; no binary replacement is invented.

Counts add under repeated contraction, preserving total endpoints. For any assignment of quotient groups, occupied final parts equal those obtained by mapping original endpoints directly; connectivity cost is therefore identical. The solver consumes this quotient, and tests compare original and contracted costs. Endpoint identities within a displayed group are lost and require the saved original graph. A four-endpoint clique projection illustrates the distinction: native cost is equal for 3+1 and 2+2 splits, whereas pairwise crossing counts differ.

## 3. Temporal updates and adaptive events

Define an old record's complete path as its coarse and fine IDs. With S_old previous scientific records and O all previous records, bounded updates enforce

$$
\sum_{v\in O}\mathbf1[p_t(v)\ne p_{t-1}(v)]\leq\lfloor .1|S_{\rm old}|\rfloor.
$$

Both scientific and metadata changes consume this budget; changing both levels counts once. New records consume no old-record slots. The bound applies per update, not for the lifetime of the hierarchy.

Changed semantic/native incident mass ranks old scientific nodes. The frontier is capped without expanding over whole touched topics. Outside vertices contract to separate fixed terminals for their occupied parts. Unary temporal penalties are represented exactly by edges to fixed old-label anchors. Fine and coarse stage penalties are additive; the hard path budget counts their union. Anchor masses are disclosed, incumbent-feasible capacities avoid balance-induced false stability, and anchors cannot substitute for occupied scientific groups. Floating checks reject numerical regressions after integer-weight optimization. Whole-fine-group coarse moves charge every affected old path.

Optional event transactions use another explicit criterion: fine-plus-coarse spherical dispersion, native connectivity at both levels, and a penalty for the fraction of changed old paths. Spherical dispersion is the sum of group size minus norm of its unit-vector sum, normalized by scientific node count. This supplies a semantic-fit reason to split, absent from positive cut cost alone. It is a two-stage acceptance scheme, not an undisclosed single global objective.

First-view leave-source-out similarities provide novelty/cohesion alarms. Shared-source records are excluded from that fold's reference pool. These dependent thresholds are empirical heuristics, not guaranteed novelty detection. A cohesive new cohort can acquire a slot only within budget; at capacity an accepted same-parent donor merge must free one. Atomicity prevents half-merges or dangling metadata IDs.

The separate old-split intervention examines one dispersed touched previous topic. Each proposed child needs multiple original old members and must pass the construction-space cohesion gate. One child retains the old ID, the other receives a new ID under the same parent. At capacity, another same-parent donor pair must merge. Full criterion checks and all old metadata/scientific movements apply. Split capability is distinct from an observed beneficial split, and its experiment is not substituted for the selected default.

Laminarity, coverage, group budgets, counted contraction and bounded outside-region persistence are enforceable properties. Meaningful topics, useful adaptation and faithful explanations remain empirical. Exact neighbor queries, validation and quotient checks still scan global inputs. A candidate split needs quadratic pair storage within its topic; the solver is heuristic, with no global optimality or end-to-end sublinear update guarantee.

## 4. Testing and comparative evidence

### 4.1 Composition and temporal response

The selected default is balanced initialization with bounded refinement. Append-only placement and event-enabled updates are controls; historical construction is retained separately. Append-only placement tests whether apparent stability merely reflects freezing. Comparisons share visible inputs, frozen vector history and native weights. Different encoders or rebuilt affinity graphs would confound an isolated update-policy comparison.

**Table 2.** Method comparison under matched display budgets. Independent cosine and excess over the null are descriptive scores, not percentages of scientifically correct topics.

{{COMPARISON}}

![Independent quality against null and chronological adaptation](figures/repair_quality.png)

**Figure 2.** Quality and stability answer different questions. Coherence uses a different encoder family; old-path changes and chronological agreement quantify persistence. Neither alone proves useful scientific abstraction.

Nemotron supplies a representation distinct from E5; type-stratified size-preserving random memberships supply the null. However, this evaluator and corpus were inspected during development. Encoder separation controls direct geometric circularity, not repeated-selection leakage. Changed partitions receive fresh measurements rather than inherited blind judgments.

**Table 3.** Real chronological transitions, complete old-path movement and conditional alignment references. Statistical significance concerns random correspondence, not the temporal fitting process.

{{TRANSITIONS}}

ARI is exact for the two fitted finite partitions. Overlapping provenance joins source papers into dependence components; too few effective components prevent a defensible process-level confidence interval. Source-deletion sensitivity and within-type label permutations address narrower conditional questions. Frozen assignments can also reject random correspondence. They do not establish responsive adaptation.

**Table 4.** Fresh initial-view edge-deletion rebuilds. Seed-based intervals describe the fixed corruption experiment and do not replace uncertainty about future corpus updates.

{{ROBUSTNESS}}

Structural verification independently reconstructs paths, counts memberships/budgets and compares exported endpoint counts. Broken overlap, overflow and dangling-parent fixtures must fail. Update tests cover no-ops, widely spread weak changes, remote coarse moves, exact saved vectors, beneficial changed evidence, cumulative article batches and atomic capacity-saturated transactions. Passing these tests establishes implementation properties, not semantic validity.

### 4.2 Explanatory labels and faithfulness

Labels explain a common scientific theme rather than list member names. Exact prompts, cutoff-safe evidence and outputs are cached. Source-separated generation/support pools exclude shared members; claim context is included only when its provenance belongs to the appropriate side. This does not make support independent of clustering, which saw the member names.

**Table 5.** Label coverage and support judgments with explicit denominators. Unsupported, vague and insufficient-evidence cases remain separate; insufficient evidence is not counted as faithful.

{{LABELS}}

A local entailment checker is tested against corrupted and generic controls. Its mismatch with a separate blinded LLM audit limits interpretation; neither is expert ground truth. In the initial-cutoff audit, only a minority of own labels were rated accurate: insufficient evidence predominated, and some were wrong. Cross-topic corruptions were rejected, showing sensitivity to blatant errors without establishing reliable own-label support. Measured subsets must not be extrapolated to every group or later cutoff. Generated explanations therefore do not establish T5 success; automatic strings and expert-validated faithfulness remain different outcomes.

### 4.3 Extrinsic retrieval and actual work

We compare identical BM25 or BM25/E5/native/source systems with and without added hierarchy access. Micro Recall@30 uses the same target occurrences and filters. Positive effects of original source/hyperedge links cannot be attributed to the added topic hierarchy. Fixed branch widths are reported without choosing a winner from answer inspection.

**Table 6.** Paired retrieval results. A non-improvement is a negative result for the tested policy; it neither fulfills positive downstream utility nor disproves every hierarchy method.

{{RETRIEVAL}}

![Retrieval with and without the added hierarchy](figures/repair_retrieval.png)

**Figure 3.** Matched Recall@30 comparisons separate original graph assistance from the intervention contributed by topic groups.

A separate coarse-first search scores group documents/prototypes, then only selected children and leaves. Leaf BM25 preserves corpus weights while slicing selected rows before query scoring. Induced native/source propagation runs inside those candidates. This tests genuine navigation rather than filtering a previously scored full corpus. Preprocessing, group work and leaf work are counted; fewer evaluated leaves alone do not establish faster end-to-end retrieval.

**Table 7.** Coarse-first work/quality trade-off. These development measurements are not a guaranteed speedup or an untouched retrieval test.

{{TREE_WORK}}

## 5. Discussion and reproducibility

Balanced bounded refinement is the default for a reviewable structure with explicit edit limits. Optional transaction controls are not promoted merely because their construction criterion decreases; independent quality and task utility can disagree. Hard budgets can defer real corrections, and local changes can preserve wrong themes. Metadata attachment and size guards remain pragmatic assumptions.

No positive hierarchy retrieval advantage or expert-level label faithfulness is established by structural correctness. Further limits include incomplete text, extraction/date errors, inspected development evaluators, source dependence, history-dependent affinities and missing prospective process uncertainty. The original scientific task therefore retains substantive open outcomes.

With four more weeks, we would collect unseen source-balanced queries and expert label/group judgments, calibrate event decisions under declared error costs, study arrival-batch sensitivity and evaluate navigation with complete time/memory accounting. Full paper evidence would require a separate documented acquisition/extraction study. These are proposed studies, not measured results.

The repository preserves per-snapshot hierarchies, original/coarsened inputs, event lineage, checkpoints, prompts, measurements and manifests. The reproduction command builds, validates, measures and regenerates numerical displays from metrics. Frozen model outputs enable replay without repeated neural inference; cached assistant judgments are disclosed as such. `AI_USAGE.md` records assistance and corrected failures. Verification, assumptions and unfulfilled scientific claims remain distinct.

## References

[1] L. Gottesbüren, T. Heuer, P. Sanders, and S. Schlag. Scalable Shared-Memory Hypergraph Partitioning. ALENEX, 2021. https://doi.org/10.1137/1.9781611976472.2

[2] A. Loukas. Graph Reduction with Spectral and Cut Guarantees. JMLR, 20(116):1–42, 2019. https://jmlr.org/papers/v20/18-680.html

[3] V. A. Traag, L. Waltman, and N. J. van Eck. From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports, 9:5233, 2019. https://doi.org/10.1038/s41598-019-41695-z

[4] G. Bokov et al. LD-Leiden: Local Parallel Community Detection in Large Dynamic Networks. Updated preprint, 2026. https://arxiv.org/abs/2502.18497
