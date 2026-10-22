# PhD Candidate Assessment Task — Multi-Resolution Semantic Abstraction over an Evolving Knowledge Hypergraph

Tags: #assessment #hiring #SLoD #hypergraph #temporal #internship
Created: 2026-08-17

**Group:** Constructor Knowledge Labs, Knowledge Discovery project
**Track:** PhD-level research internship / doctoral project
**Estimated effort:** 5–7 days
**Language of deliverables:** English


---

## 1. Background

Our group builds the **Temporal Knowledge Hypergraph (TKH)**, a machine-readable
representation of scientific literature. Nodes are entities, concepts, methods and
datasets. Edges are **hyper-edges**: a single edge may connect three or more nodes
at once (e.g. a method-comparison table connects several methods, one dataset and
one metric). Every node and edge carries temporal and provenance attributes, and
the graph **grows over time** as new papers enter the corpus.

A TKH over a few thousand papers already contains tens of thousands of nodes.
Showing all of them is useless to a human and expensive for a retrieval agent. We
need a **multi-resolution semantic abstraction**: a hierarchy of increasingly
coarse views of the same graph, so a reader starts from ~a dozen macro-concepts
and drills down to individual methods and papers — and, crucially, so that this
hierarchy **stays coherent as the corpus evolves**.

The naive solution (sample, or threshold by degree) produces a smaller view, not a
more meaningful one. We want aggregations where every super-node stands for a
coherent scientific idea a domain expert would recognise, that respect the
hyper-edge structure rather than a lossy graph projection, and that do not
reshuffle every time the corpus is updated.

## 2. Problem definition

We deliberately give you **less scaffolding than the MSc version**. Part of what we
assess is whether you can turn the following informal desiderata into a precise
problem statement and a defensible method.

A multi-resolution semantic abstraction over a temporal hypergraph
*H(t) = (V(t), E(t))* is, at each snapshot *t*, a sequence of partitions
*P_0, P_1, …, P_K* of *V(t)* (P_K finest, one node per cluster; P_0 coarsest),
together with labels, satisfying:

- **P1. Laminar refinement.** Each partition refines the one above it (a node in
  super-node *S* at level *k* lies in exactly one descendant of *S* at *k+1*).
  Overlapping clusters are out of scope.
- **P2. Size budget.** Level *k* has at most *N_k* super-nodes, *N_0 ≈ 10–15*. A
  hard display constraint, not a soft objective.
- **P3. Semantic coherence.** Members of a super-node are close in **meaning**, not
  merely in citation/co-occurrence topology. Structure-only and content-only
  signals will disagree; **how you reconcile them is a central research question,
  not a detail** — state the trade-off explicitly and justify your resolution.
- **P4. Hyper-edge fidelity.** The abstraction operates on the hypergraph, not a
  pairwise projection. Your coarsening rule for hyper-edges (Task T4) must be
  implemented and actually used by the method, not only described.
- **P5. Temporal stability.** As *H(t) → H(t+1)*, the hierarchy must evolve
  smoothly: stable regions persist, and change is localised to where the corpus
  actually changed. Super-node identity must be trackable across snapshots
  (birth / growth / merge / split / death).
- **P6. Interpretable, faithful labels.** Every super-node needs a short label and a
  one-sentence gloss, generated automatically, that assert **nothing the member
  nodes do not support** (temporal honesty: no claim beyond the snapshot's cutoff).

**Deliverable 0 (part of the report): a formal statement.** Write the objective
your method optimises (or the criterion it satisfies), the trade-off it makes
between structural and semantic signal, its complexity, and which of P1–P6 it
guarantees *by construction* versus *empirically*. Defend it against at least one
credible alternative you rejected.

## 3. Literature grounding (re-introduced vs the MSc task)

Read and synthesise **3–4 papers** spanning: hierarchical / multi-resolution
community detection, **hypergraph** clustering or spectral hypergraph partitioning,
graph coarsening, and dynamic / evolutionary community detection. Recent
(2022–2026) venue papers (KDD, NeurIPS, ICLR, WWW, TKDE) are welcome. In the
report, position your method against this prior art — what you borrow, what you do
differently, and why the hypergraph/temporal setting needs it.

## 4. Data

Use **one** of the following. Option A is preferred if provided with your task.

**Option A.** A **temporal** TKH export we provide: nodes (id, type, surface form,
year, provenance) and hyper-edges (id, member node ids, relation type, year),
sliced into **at least three time snapshots** (e.g. ≤2020, ≤2022, ≤2024) so the
evolution is real, not simulated.

**Option B.** Build your own small temporal hypergraph: ~300–500 arXiv abstracts
from one category over a fixed window (e.g. `cs.LG`, 2019–2024), extract
method/dataset mentions with any approach (LLM allowed), and connect mentions
co-occurring in the same abstract as a **hyper-edge** (arity = number of distinct
mentions in that abstract). Cut into ≥3 yearly/biennial snapshots. State clearly
that these are **co-occurrence** hyper-edges, not verified relations.

Keep it small enough to iterate in minutes, not hours. We evaluate reasoning, not
compute budget.

## 5. Tasks

**T1. Load and describe the evolving graph (≈0.5 day).**
Per snapshot: node/edge counts, node-type distribution, **hyper-edge arity
distribution**, time span, and how much the graph grows/changes between snapshots.
Note data-quality problems.

**T2. Method and formal statement (≈2 days).**
Design and implement your multi-resolution abstraction. You must:
- Operate on the hypergraph directly (hyperedge-aware objective / spectral
  hypergraph method / hypergraph coarsening). If you also try a pairwise
  projection, **quantify what the projection loses** relative to the native method.
- Reconcile structural and semantic signal explicitly (P3). A single joint
  objective, a principled two-stage scheme, or a consensus method are all fine —
  but the choice must be argued, not defaulted.
- Produce a laminar hierarchy satisfying P1, P2 at every snapshot.
- Deliver the formal statement (Deliverable 0).

**T3. Temporal coupling (≈1 day).**
Extend the method across the ≥3 snapshots so the hierarchy is **temporally
stable** (P5): persistent super-node identity, and a tracked event log of
birth/growth/merge/split/death. Say what mechanism enforces stability (warm-start
from the previous snapshot, temporal regularisation, matching, …) and what it costs.

**T4. Hyper-edge collapse — design *and* implementation (≈0.5 day).**
When *m* of a hyper-edge's *n* endpoints land in the same super-node, the edge
partly becomes internal. Decide what an arity-*n* hyper-edge becomes at a coarser
level for: *m = n*; *1 < m < n*; and endpoints landing in **three or more**
distinct super-nodes. Unlike the MSc task, this is **not only a written answer** —
your coarsening rule must be the one your method actually applies. Justify it and
name what it loses.

**T5. Labelling with measured faithfulness (≈0.5 day).**
Auto-generate label + one-sentence gloss per super-node at levels 0 and 1 (LLM
call acceptable). Report the exact input you gave the labeller and, critically,
**how you keep labels faithful and quantify it** (see T6 on over-claim rate). Note
the P6 temporal-honesty constraint.

**T6. Evaluation — and a proof it is trustworthy (≈1.5 days).**
Define and compute metrics for **coherence, temporal stability, and label
faithfulness**, plus **one extrinsic, task-grounded** evaluation.

> **Read this carefully — it carries the most weight, and it contains a subtle
> methodological hazard you are expected to identify and neutralise.**

- **Coherence — beware circularity.** If you form clusters using embedding *X* and
  then measure "semantic coherence" with the *same* embedding *X*, high coherence
  is guaranteed by construction — you would be grading the method with its own
  ruler. Your coherence measure must use a signal **independent** of the one that
  drove clustering (a different embedding family, held-out relations/citations, or
  blind LLM/human judgement), and you must compare against a **null model**
  (degree/arity-preserving hypergraph shuffle, or random labels) so a number means
  something relative to chance. The same leakage applies to labels: if the surface
  forms used to cluster are also the labeller's only input, faithfulness is
  circular — control for it.
- **Stability:** rebuild after perturbation (remove 10% of hyper-edges; five
  seeds) **and** measure real cross-snapshot stability (T3), reporting Adjusted
  Rand Index (or a laminar-hierarchy analogue) **with confidence intervals /
  significance**, not point estimates.
- **Label faithfulness:** quantify an **over-claim rate** — e.g. an NLI/entailment
  check of each gloss against its members, or a blind rating of accurate / vague /
  wrong over a sample — rather than eyeballing ten.
- **Extrinsic utility:** show the abstraction *helps a downstream task*. Minimal
  version: a retrieval/agent that starts at a coarse level and drills down to find
  target nodes, versus a flat baseline, scored on hit-rate or steps-to-target.
  Intrinsic metrics alone are not enough at this level.

Compare your method variants on these metrics and say which you would ship and why.

**T7. Write-up (≈1 day).**
See deliverables. Separate explicitly what you **verified** from what you
**assumed**.

## 6. Deliverables

1. A git repo (public or zip) with runnable code, a README with exact reproduction
   steps, and a pinned dependency list.
2. `hierarchy.json` **per snapshot**: for each super-node — id, level, parent id,
   member ids, label, gloss; plus a `temporal_events.json` (or equivalent) with the
   birth/merge/split/death log.
3. `metrics.json`: the numbers behind T6 (coherence vs null model, stability with
   CIs, over-claim rate, extrinsic-task scores).
4. `report.pdf`/`report.md`: **3–5 pages** of prose + figures, covering: the formal
   statement (Deliverable 0), the method and its literature positioning, the T4
   coarsening decision, the evaluation **including your handling of the coherence
   circularity hazard**, and what you would do with four more weeks.
5. `AI_USAGE.md`: major prompts and where you used AI assistants, with a line on
   what you verified (see §9).
6. Optional: a static visualisation of levels 0–2 across snapshots (a screenshot is
   enough). Do **not** build an interactive frontend.
7. **Mandatory computational component — pass/fail gate, see §6a.**

### 6a. Mandatory computational component

Small (≈2–4 hours) but **not optional**: a submission without it is not scored
further, however strong the report. It exists so we see how you work with code
and data, not only what you conclude.

1. **`validate.py`** — a standalone checker, run as `python validate.py out/`
   (exit 0 = pass, non-zero = fail, one line per violation). For every snapshot it
   verifies:
   - each `hierarchy.json` matches a JSON Schema shipped in the repo
     (`schema/hierarchy.schema.json`);
   - **P1 laminarity**: every node sits in exactly one super-node per level, and
     every super-node at level *k+1* has exactly one parent at level *k*;
   - **P2 budget**: level *k* has ≤ *N_k* super-nodes;
   - every member id exists in that snapshot's node list (no ids from a later
     snapshot — the P6 cutoff applied to membership);
   - `temporal_events.json` refers only to super-node ids that exist in the
     snapshots it names.
2. **Proof the validator bites.** At least three hand-made broken fixtures in
   `tests/` (e.g. a node in two clusters, a budget overflow, a dangling parent id),
   each of which `validate.py` must reject. A validator that passes everything
   proves nothing.
3. **`make_tables.py`** — regenerates **every number table in the report** from
   `metrics.json`. A number in the report not produced by this script counts as an
   unsupported claim (§7).
4. **One command end-to-end.** `make all` (or one script) runs
   build → validate → metrics → tables on a clean checkout. We will run it.

In `AI_USAGE.md`, also describe one concrete case where an AI assistant produced
wrong code or a wrong claim for this task, how you caught it, and what you
changed. "None" is acceptable only if you say how you checked.

## 7. How we assess

| Criterion | Weight |
|---|---|
| Problem formulation & method design (Deliverable 0: objective, structure/semantics trade-off, hypergraph-native, guarantees vs empirical) | 25% |
| Evaluation validity — metrics measure what you claim, **circularity neutralised**, null models, CIs, extrinsic utility | 30% |
| Temporal treatment (real cross-snapshot stability + identity tracking, not a static snapshot) | 15% |
| Hyper-edge collapse reasoning **and** its implementation (T4) | 10% |
| Reproducibility from a clean environment | 10% |
| Clarity of the report, incl. explicit separation of verified vs assumed | 10% |
| **Gate (§6a):** `make all` runs on a clean checkout; validator rejects all broken fixtures; report tables come from `make_tables.py` | pass / fail |

Two things weigh more than they look. First, we would rather see a modest method
**honestly and independently evaluated** than an ambitious method whose numbers we
cannot trust. Second, unsupported claims cost points: if a metric is noisy, a label
guessed, or a choice made for convenience, write it down. **Flagging your own
uncertainty is a strength.**

## 8. Out of scope

No production infrastructure, no graph database, no scaling beyond the sample
corpus, no model fine-tuning, no interactive UI. Use of LLMs and coding assistants
is allowed and expected — note where you used them (§9).

## 9. AI assistance (allowed & encouraged)

You are encouraged to use AI coding/writing assistants. We evaluate your
**judgment, verification habits, and experimental rigor**, not whether you wrote
every line unaided. Include an `AI_USAGE.md` listing: tool used, task/module, the
**major prompts**, a 1–2 line note on what you accepted/modified, and any important
verification you performed (tests, manual checks, ablations). You remain
responsible for correctness and for citing external sources.

## 10. Questions

Ask early rather than guessing at an interpretation. Questions about the task are
not held against you; questions sent on the last day are hard to answer in time.

---

## What we're looking for

This task sits at the intersection of **hypergraph learning**, **multi-resolution
representation**, **dynamic/temporal structure**, and **experimental rigor**. We
are not looking for a large system — we are looking for a clean, well-formalised
method whose evaluation we can trust, over an evolving corpus. We value candidates
who:

- Turn an informal desideratum into a precise, defensible problem statement.
- Recognise and neutralise evaluation confounds (e.g. coherence circularity)
  **without being told they are the key issue**.
- Distinguish what their results show from what they would like them to show.
- Make the temporal dimension real rather than decorative.
- Write code that is easy to reproduce and are honest about negative or ambiguous
  results.

**Rigor over breadth.** A simple, well-controlled, honestly-evaluated method beats
five uncontrolled experiments.
