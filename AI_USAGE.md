# AI assistance and verification

## Current submission

OpenAI Codex assisted with implementation, experiment design, debugging, analysis
and writing. The submitter remains responsible for scientific decisions. This
file describes the current CPU study; earlier model pilots and E5 experiments
are historical evidence, not measurements of the present MiniLM partitions.

The full current command is `make all`, documented in README.md. It computes real
CPU MiniLM vectors, builds five chronological native-hypergraph views, evaluates
matched retrieval and component controls, checks labels, and generates the report.
No model fine-tuning or participant solution code is used. Earlier public solutions
were inspected and executed for research comparison; current code does not import
or invoke them. The scope of the historical source-overlap audit is disclosed in
docs/CODE_ORIGIN.md.

## Major prompts and accepted or modified work

| Request or major prompt | Accepted or modified work | Verification |
|---|---|---|
| Explain and implement the official multi-resolution temporal hypergraph task | Bounded nested views, native counted contraction and saved-state updates | Independent validator, deliberately broken fixtures, endpoint audits and checkpoint comparisons |
| Preserve earlier work when adding papers; test the 12/50 limits | Old vectors and old-old affinities retained; at most 10% old scientific paths editable in the chosen mode | Actual five-year updates and tests adding 100 synthetic source papers |
| Compare identical retrieval with and without our hierarchy, using Recall@30 | Common channels, graph/source walk, group overlay and separate coarse-first routing | Predictions locked before reference access; exact gained/lost targets and paired conditional intervals |
| Use a small Sentence Transformers model on CPU and start from raw data | Pinned all-MiniLM-L6-v2 with an exact-text SQLite cache | Real model inference and extracted-archive reproduction; synthetic tests replace only external neural boundaries |
| Name each group with a scientific heading and explanatory sentence | Prompt in prompts/group_labels.md, adapted from the user's own topic-title prompt | Exact evidence, guided JSON requests and raw NVIDIA responses authenticated before publication |
| Check label faithfulness without long local-model runs; do not use Gemini | NVIDIA Nemotron Ultra checks a fixed blinded sample and sensitivity controls | Raw judgments replayed and authenticated; failed prior checkers retained; empty evidence uses a separately logged deterministic rule |
| Finish with component comparisons, a real hierarchy example and consistent documentation | Offline zero-native-cost and unbounded-edit controls; deterministic saved branch illustration | Same vectors/cutoffs/budgets/search; independent structural/temporal audits; no transferred label ratings |

## Concrete AI errors, detection and correction

1. An early assistant claim that the hierarchy did not help was based on an
   incomplete set of baseline comparisons. The user identified the missing strong
   graph/source baseline. Matched comparisons were added; current results remain
   negative, with conditions and target swaps retained.
2. Prompt-only requests were initially treated as if they guaranteed JSON.
   Actual malformed responses exposed the error. Naming now uses the provider's
   guided structured generation, bounded retries and saved raw responses.
3. Earlier label checks confused name coverage with factual faithfulness and
   some judges accepted deliberately bad controls. Support claims are now split,
   a fixed sample is blinded, controls precede interpretation, and failed checker
   attempts are archived rather than counted as successful final assessments.
4. A previous reproduction description depended on saved E5 vectors. Reading the
   actual clean-start path exposed that mismatch with the requested raw-data
   workflow. The current command downloads the pinned small model when needed,
   performs real inference and authenticates exact-text caches.
5. The assistant described a negative downstream result too broadly as failure
   to perform the evaluation. The comparison was performed; the correct
   conclusion is that no gain was established under the tested settings. This
   is distinguished from unimplemented adaptive topic events.
6. A report-generation stage was started before its asynchronous illustration
   preparation finished. Checking metrics against the actual rendered file
   exposed the missing illustration. Generation was repeated after preparation;
   the end-to-end Makefile keeps these dependent stages sequential.

7. Finishing integration exposed that re-authenticating unchanged labels rewrote
   JSON key ordering and invalidated report byte hashes. A regression reproduced
   it; publication now leaves semantically unchanged metrics untouched.
8. A fresh-context code review found that component reuse lacked reference-answer
   and main-baseline authentication, and packaging could accept modified control
   artifacts. Failing changed-reference, changed-baseline and tampered-package
   cases were added. Read-only verification now authenticates all of those inputs;
   reference hashing starts only after prediction locking. Controls were rerun
   after the source change rather than credited to a version that never ran.

## Labels and model-based checking

Generator: `nvidia/nemotron-3.5-lightning-30b-a3b`. Checker:
`nvidia/nemotron-3-ultra-550b-a55b`. The generator receives scientific member names,
selected member statements and separated neighbouring context. Exact full inputs
are saved in the annotation archive and restored to `OUT/labels` during reproduction. The prompt requests a 2–12-word heading and one
explanatory sentence. Word count and punctuation preferences are warnings, not
semantic failure conditions. Generation uses at most three groups per request,
512 output tokens and no reasoning output.

The main study authenticates 310 labels. Its 20 hash-selected audit groups yielded
16 accurate and four vague model ratings. Nineteen have withheld claim evidence.
These are descriptive sample ratings, not population accuracy. Both checkpoints
belong to Nemotron; shared lineage limits independence. Nonclaim names overlap
construction and naming; only withheld claim text supplies a separate factual
premise. Six model-assessed controls and one deterministic strictly-empty-evidence
control passed. The deterministic result has distinct provenance and is not
presented as an API completion.

Gemini is not used by the final workflow. Old failed/cancelled checker attempts
remain under `labels/audit` inside the annotation archive; their ratings are excluded from the final
sample. Keyless reproduction replays authenticated responses only for identical
groups, evidence, prompts and settings. It does not pretend to perform new model
calls. Changed component-control groups have no inherited labels or T5 score.

## Scientific scope and disclosure

The primary retrieval outcome is Type A micro Recall@30. The questions and
reference answers were inspected during development; prediction locks prevent
run-time answer access, not historical selection effects. Citation coherence
holds out link topology, not cited names, and is a limited relational proxy.
Chronological agreement is partly imposed by constrained editing. Conditional
significance is available, but source dependence prevents a defensible temporal
process confidence interval. Fixed topic slots do not demonstrate discovery of
new scientific fields. No positive retrieval gain is claimed.

The final component controls remove one mechanism at a time. Zero native cut
weight still retains provenance for metadata and search. Unbounded edits remove
the old-scientific cap and movement penalty, while keeping warm starts, fixed
slots and stored affinities. These are exploratory diagnostics, not an unseen
method-selection experiment. The illustration selects persistent IDs rather
than the most coherent-looking branch and preserves actual membership.

Scientific Agent Skills informed design, visualisation and critical reporting;
the verified current paper is cited in the report. Skills and AI judgments are
procedural assistance, not evidence that scientific claims are true. Primary
sources consulted for these finishing changes are linked in
`docs/COMPONENT_CONTROLS.md`.

## Evidence and release boundary

`results/verification.json` and its logs distinguish actual clean CPU execution
from mocked transport/neural tests. The release includes code, data, per-cutoff
hierarchies, prompts, raw responses, predictions, metric inputs and reports.
Credentials, environments, pretrained weights and text caches are excluded.
Standard libraries and their public APIs are attributed in docs/CODE_ORIGIN.md;
no participant repository is imported into the implementation. Previous research
notes are marked historical. No publication or external submission is automatic.
# Technical release cleanup and timing

The assistant removed obsolete embedding assets and repeated generated solver
states from the Git distribution, compressed the exact NVIDIA response evidence,
and added a standard-library runner behind `make all`. The runner bootstraps the
pinned CPU environment when absent, records subprocess and total wall-clock
timings, and explicitly replays saved annotations without new LLM requests.
Technical regression tests cover timing on success/failure, archive integrity,
relocated results and validation of the compact published hierarchy. A full
real-model local reproduction is recorded separately from mocked unit tests.

