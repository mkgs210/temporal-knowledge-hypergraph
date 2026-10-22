# Official assessment coverage — current CPU study

The official assessment remains authoritative. Current measurements use CPU
MiniLM, the native balanced builder and bounded saved-state updates. Historical
E5/Nemotron diagnostics elsewhere are not measurements of these partitions.

| Requirement | Current evidence | Result / limit |
|---|---|---|
| T1: temporal graph description | `docs/DATA_SUMMARY.md`; `results/metrics.json` → `construction.snapshots[*].snapshot`; `evidence/data_quality/` | Five publication-year views; node/edge counts, types, exact arities, year distributions, growth and quality discrepancies; retrospective cutoffs are not ingestion logs |
| T2 / Deliverable 0: native semantic/structural objective and bounded laminar hierarchy | Formal report; native KM1 solver, semantic neighbours and size guards | Implemented heuristic; 12/50 budgets are hard, semantic topic correctness empirical; gamma is fixed development configuration |
| T3: localized temporal state | Published per-cutoff event logs and traces; metrics; checkpoints regenerated under `OUT/hierarchy/` | Old vectors and old-old affinity entries preserved; bounded old-path edits; adaptive topic-event discovery disabled |
| T4: actual multiway collapse | Counted quotients and independent endpoint/objective audits | Fully internal, partial and multi-group cases implemented; internal endpoint layout is lost in coarse view |
| T5: explanatory labels and measured faithfulness | `annotations/nvidia_labels.tar.gz`: exact inputs, requests, responses, blinded packet, controls and judgments | 310 authenticated labels; 20 sampled: 16 accurate, 4 vague; controls passed, including 1 separately logged empty-evidence rule; 0% observed wrong is a model sample estimate |
| T6: coherence independent of construction and versus null | Held-out citation topology; type/size-preserving permutations | Positive excess on both final levels; this is a relational proxy, not independent full-text semantic truth |
| T6: perturbation and chronological stability | Five genuine 10%-edge-deletion rebuilds; ARI intervals and cross-snapshot significance | Moderate deletion robustness; chronological process CIs unavailable because sources are dependent; conditional significance is reported |
| T6: downstream comparison | Same BM25/native/source and mixed search with/without hierarchy; true coarse-first arms | Negative result: no matched Recall@30 gain. The evaluation was performed; positive utility is not claimed |
| T6: method component comparison | `results/metrics.json` → `components`; detailed runs regenerated under `OUT/components/` | Fixed zero-native-cost and unbounded-edit controls; same representations, cutoffs, caps and retrieval; fresh citation/null, deletion and chronological checks; no transferred T5 ratings |
| T7: report and disclosure | Generated English `report.md` / `report.pdf`, figures, AI_USAGE, four methodological papers | Article within the 3–5 page budget; real branch illustration and comparison tables generated from saved metrics; limits and negative findings retained |
| Standalone computational gate | `validate.py`, JSON Schema and deliberate invalid fixtures | Schema, nesting, budgets, complete membership, cutoff and event checks; invalid fixtures rejected in tests |
| One-command computational gate | `make all`; automatically created isolated `.venv-cpu`; authenticated annotation archive | Clean source-copy run, installation and all twelve stages recorded in `results/verification.json` and `docs/CPU_RUN.md`; real CPU model, fresh vectors, no new API calls |

The primary quality measure is Type A micro Recall@30 across all reference
occurrences. Query-development exposure, incomplete reference relevance judgments,
small audit sampling and model-based ratings limit the scientific conclusions.
A complete computational study is not a claim that the hierarchy improves search
or that every scientific desideratum was achieved. A negative retrieval result
is a legitimate measured finding. The conservative `full_spec_satisfied: false`
also avoids claiming adaptive topic discovery or universally faithful semantic
groups, which the selected configuration and limited audits do not establish.

## Complete specification check before publication

| Specification item | Evidence / assessment |
| --- | --- |
| English deliverables, §0 | README, article and submission documentation are English |
| Option A and at least three real temporal views, §4 | Supplied TKH; five publication-year cutoffs, 2022–2026; source limitations disclosed |
| Small enough to iterate in minutes, §4 | Measured complete `make all`: 411.10 seconds including environment setup; downloaded model files and pip wheels already cached |
| P1: a partition at every level and laminar parents | Standalone schema/partition/parent checks over every published cutoff |
| P2: hard level budgets | At most 12 coarse / 50 fine groups total; tests add 100 source papers and reject output-authorized budget inflation |
| P3: reconcile meaning and structure | Explicit cosine-neighbour plus native KM1 objective, gamma 0.03 and component control; scientific topic correctness remains empirical |
| P4: operate on actual hyperedges | Native multiway solver, counted quotient used in construction/refinement; direct-contraction and endpoint-count checks |
| P5: identities, localized updates, event tracking | Checkpoint reuse, fixed outside paths, bounded old edits and actual event logs; chosen run keeps fixed topic slots, so adaptation to newly emerging fields is unestablished |
| P6: readable labels/glosses and temporal honesty | Automatic L0/L1 labels, cutoff-filtered member/claim evidence and audit; universal factual support is not guaranteed by a small model-rated sample |
| Deliverable 0: criterion and structural/semantic trade-off | Report §2: weighted connectivity cost plus temporal movement penalty; explicit heuristic and engineering-choice limits |
| Deliverable 0: complexity and guarantees | Report §§2–3: initial quadratic neighbour computation, new-to-all update cost, hard invariants versus empirical outcomes |
| Deliverable 0: credible rejected alternative | Report §1: embedding-only clustering, pairwise projection and browsing-only hierarchy |
| Literature: synthesize 3–4 relevant papers, §3 | Four methodological papers positioned in report §1; borrowed mechanisms and non-inherited guarantees distinguished |
| T2: projection-loss comparison if projection is tried | The submitted construction uses native hyperedges; the zero-native-cost control removes a term, not a pairwise projection experiment |
| T4: full, partial and three-or-more-group collapse | Report §3 and implemented counted contraction; coarse views lose internal endpoint layout while preserving drill-down membership |
| T5: exact labeller input | Archived group evidence, prompt, guided JSON requests and raw responses; generation-side and checking-side statements distinguished |
| T6: independent coherence and null | Citation topology excluded from construction cost and neighbour/metadata selection; type/size-preserving permutation null; names remain visible, so this is a limited relational proxy |
| T6: avoid circular label assessment | Withheld claim sentences, separate checkpoint, blind sample and controls; visible member names overlap generator input, limiting independence |
| T6: perturbation rebuilds | Five actual 10%-edge-deletion initial rebuilds, with ARI and conditional confidence intervals |
| T6: real chronological stability and uncertainty | Shared-record ARI over four actual updates, conditional significance; process confidence intervals marked unavailable under source dependence |
| T6: overclaim assessment | Fixed blind 20-group sample, accurate/vague/wrong ratings, denominator and control gate; 0 wrong is not population accuracy |
| T6: downstream coarse-first versus flat | Explicit coarse-to-fine traversal arms plus matched flat/graph/overlay comparisons; Recall@30 and question-conditional intervals; no gain established |
| T6: compare variants and select deployment | Full, zero-native-cost and unbounded-edit controls; report retains bounded updates and recommends the stronger graph-assisted search as retrieval default; controls have no transferred T5 score |
| T7: verified versus assumed | Report §§4–7 and this checklist distinguish executable invariants, empirical proxies and unestablished claims |
| Deliverable 1: runnable repo, exact steps, pins | `make all`, README, Python 3.12 CPU lockfiles; automatic isolated environment creation |
| Deliverable 2: per-view hierarchies and event logs | `results/labelled_hierarchy/snapshots/YEAR/`; id, level, parent, members, label and gloss, plus temporal events |
| Deliverable 3: numbers behind T6 | `results/metrics.json`; full raw intermediate artifacts recreated by the public command |
| Deliverable 4: 3–5-page article and four-week plan | Generated PDF has five pages; formal criterion, literature, collapse, evaluation, diagrams and next-work plan included |
| Deliverable 5 / §9: AI use and a concrete corrected AI error | `AI_USAGE.md`: major prompts, accepted/modified work, checks and documented mistakes |
| Deliverable 6: optional static illustration | Actual saved coarse/fine/leaf excerpt and retrieval chart; no interactive frontend |
| §6a: independent `validate.py` and JSON Schema | CLI verifies schema, complete partitions, nesting, budgets, cutoff membership and event references; exit status communicates violations |
| §6a: validator rejects broken inputs | Handwritten overlap, budget-overflow and dangling-parent fixtures, exercised through the CLI |
| §6a: every report number table comes from metrics | Exact report text matches `cpu_report.render_report(metrics)` invoked by `make_tables.py` |
| §6a: one-command clean execution | Twelve stages, actual CPU model and new vectors; archived model judgments authenticated without new API calls; complete timing transcript retained |
| §8: exclusions | No model fine-tuning, production infrastructure, graph database or interactive UI |

This checklist is a coverage audit, not a claim that all empirical desiderata are
proven. In particular, held-out citations do not establish a domain-expert
scientific taxonomy, and preserved old paths do not prove correct adaptation.
