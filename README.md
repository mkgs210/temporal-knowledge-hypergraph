# Bounded temporal knowledge-hypergraph abstraction

Native hypergraph partitioning with sentence embeddings, persisted chronological
updates, counted hyperedge contraction and separately checked labels.

The current method uses **all-MiniLM-L6-v2 on CPU**, without fine-tuning. Every
snapshot has at most **12 coarse and 50 fine groups in total**, followed by
original-node leaves. Updates reuse saved vectors and affinities; at most 10% of
old scientific paths may change in the affected frontier. Original multiway edges
retain endpoint counts and provenance when contracted.

## Measured retrieval outcome

The main outcome is Type A micro **Recall@30**, on 14 development questions and
63 reference target occurrences. Each overlay uses the same seed search, original
relations/source transitions, matching rules and denominator as its baseline.

| Search | Found / targets | Recall@30 |
|---|---:|---:|
| BM25 + original relations/source | 28/63 | 44.4% |
| Same search + hierarchy | 28/63 | 44.4% |
| BM25 + MiniLM + original relations/source | 22/63 | 34.9% |
| Same mixed search + hierarchy | 22/63 | 34.9% |
| Coarse-first, 2 coarse / 8 fine | 8/63 | 12.7% |
| Coarse-first, 8 coarse / 32 fine | 22/63 | 34.9% |
| All branches | 22/63 | 34.9% |

**The hierarchy has not demonstrated a retrieval improvement.** Narrow traversal
loses targets. Citation coherence exceeds a type/size-preserving random null, but
is a held-out relational proxy, not a percentage of scientifically correct topics.
Deletion robustness is moderate. See the generated [report](report.md) and
[PDF](report.pdf) for uncertainty, actual label-audit results and failed checks.
A successful program run is not proof that every scientific desideratum passed.

The study also removes two components one at a time: the native scientific cut
term, and the old-edit cap plus movement penalty. These controls share vectors,
cutoffs, budgets and search. Their results and limitations are in the report and
[COMPONENT_CONTROLS.md](docs/COMPONENT_CONTROLS.md). A saved branch illustration
shows actual generated group names and member records, selected by persistent IDs.

The complete [specification checklist](docs/REQUIREMENTS.md) distinguishes
implemented checks from scientific limits. The [data summary](docs/DATA_SUMMARY.md)
describes per-year node/relation types, hyperedge arities, growth and raw quality
discrepancies without altering the supplied data.

## Reproduce from raw data

Python **3.12**, Linux x86_64. Use a fresh isolated environment; a system-site
CUDA torchvision installation conflicts with CPU PyTorch. No GPU is required.

```bash
make all
```

This one command creates `.venv-cpu` and installs the pinned CPU dependencies when
the environment does not exist, then runs every stage through ZIP packaging. An
existing environment is reused. Each stage and the complete wall-clock duration
are recorded in `out/cpu-timing.json`; console output is retained in `out/cpu.log`.
If `out/cpu` already exists, a new timestamped run directory is used. `OUT=...` and
`PYTHON=...` can select another output directory or an existing environment.

Measured clean run: **6 min 51 s including environment setup**, or **5 min 49 s
for the pipeline itself**. Downloaded model files and pip wheels were cached;
text vectors were computed afresh. See [per-stage timings](docs/CPU_RUN.md).

The first run downloads the pinned Sentence Transformers checkpoint and computes
normalized 384-dimensional vectors from raw own `surface_form` text. Exact-text
SQLite cache entries survive interruption. Full article texts are not provided.
Old E5/Nemotron assets and participant repositories are not construction inputs.

The full command uses the authenticated [annotation archive](annotations/README.md)
and makes no new LLM API requests, even if a key exists locally. Replay is allowed
only if graph, partition, evidence, prompts and settings match exactly. These are
archived judgments, not fresh model calls. Changed inputs need new annotations.
For an explicit live run, put `NVIDIA_API_KEY` in the ignored `.env` and use the
separate `make labels` target with its `OUT`. `make offline` omits T5 and cannot
certify a complete label evaluation.

The checker also uses NVIDIA, with a separate Nemotron Ultra checkpoint.
Gemini checking and its automatic selection are disabled.
Archived NVIDIA audits retain their exact model on resume. Strictly empty evidence is
classified by an explicit rule with separate provenance; the model handles the
remaining cases. Earlier failed controls and unavailable API attempts are retained.

The full command performs build → standalone validation → matched retrieval →
citation/null, deletion and temporal checks → offline component controls → batched
naming → blind audit → validated labelled views → saved hierarchy illustration →
scientific article, figures and local review ZIP.
Generation uses strict guided JSON, at most three groups per batch and 512 output
tokens, with no artificial RPM cap. Actual API failures have bounded retries.
An independent judge must pass sensitivity controls before its rates are trusted.

## Outputs and resume

- `results/metrics.json`: shipped measured results; fresh runs write `OUT/metrics.json`.
- `results/labelled_hierarchy/`: published per-year JSON views, without duplicated vectors or checkpoints.
- `annotations/`: compressed exact LLM evidence; no API key or model weights.
- `OUT/hierarchy/`: raw chronological structures, checkpoints and event logs.
- `OUT/labelled_hierarchy/snapshots/YEAR/hierarchy.json`: authenticated scientific labels.
- `OUT/labels/`: exact prompts, API requests/responses and blinded audit evidence.
- `OUT/science/`, `OUT/assets/`, predictions and locks: reproducible measurement inputs.
- `OUT/components/`: two validated chronological controls, frozen predictions and independent assessments.
- `report.md`, `report.pdf`, retrieval and hierarchy figures: generated from current metrics.
- `OUT-name-submission.zip`: code, supplied data, this run and report; credentials excluded.

The full command creates a new run rather than overwriting a completed one. After an API interruption, use `make labels`,
then `make label-audit`, `make illustrate`, `make pdf` and `make package`, with the same `PYTHON` and
`OUT`; completed requests are reused. Other separate stages are `make build`,
`make validate`, `make metrics`, `make science` and `make components`. See [CPU_RUN.md](docs/CPU_RUN.md).

```bash
make test PYTHON=.venv-cpu/bin/python
.venv-cpu/bin/python validate.py results
PYTHONPATH=src .venv-cpu/bin/python make_tables.py results --report report.md --pdf report.pdf
```

The standalone validator checks schemas, complete partitions, nesting, hard
budgets, cutoff membership and event references. Intentional overlap, overflow
and dangling-parent fixtures must fail. Growth tests add 100 articles in batches
and verify bounded groups, preserved old vectors and saved-state reuse.

## Method and provenance

[Mt-KaHyPar](https://github.com/kahypar/mt-kahypar) is the MIT-licensed native
hypergraph partitioning library, not another participant's solution. We implement
our own nesting, chronological constraints, evidence accounting and evaluation
around its public API. [CODE_ORIGIN.md](docs/CODE_ORIGIN.md) attributes libraries,
models and literature; [AI_USAGE.md](AI_USAGE.md) records assistance and corrected
errors. The report positions the approach against four methodological papers.

Generated vector caches, solver checkpoints, historical embedding assets and old
review ZIPs are excluded from Git and reconstructed locally. The repository keeps
raw input data, published JSON views, measured metrics, report and annotation
evidence. See [CPU_RUN.md](docs/CPU_RUN.md) for the measured execution scope.

Core code: `cpu_embeddings.py`, `cpu_pipeline.py`, `balanced_hierarchy.py`,
`local_update.py`, `divisive_solver.py`, `cpu_science.py`, `cpu_label_audit.py`,
`cpu_report.py` and `cpu_package.py` under `src/tkh/`. Standard dependencies are
pinned in `requirements-cpu.lock`. Older E5 diagnostics in documentation describe
historical experiments and must not be read as MiniLM results. Nothing is
published or submitted automatically.
