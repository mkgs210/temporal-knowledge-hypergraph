# Current CPU workflow

Run `make all` with Python 3.12 and GNU Make available. It creates the isolated CPU
environment and installs pinned dependencies if needed. The current default encoder
is sentence-transformers/all-MiniLM-L6-v2 at revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41, using Sentence Transformers 5.2.0,
CPU, normalized 384-dimensional float32 outputs and a 256-token input limit.
Documents use exact own surface_form text; questions use their exact text. No E5
prefixes, fine-tuning or full article text are introduced.

The exact-text SQLite cache under .cache/tkh/embeddings is keyed by model settings
and revision. Completed batches survive interruption; hits need no model
initialization. Model weights may be downloaded on the first run. Old vectors and
old-old affinities are separately persisted in chronological checkpoints.

Run the complete study with a new output directory:

```bash
make all
```

Stages: raw-data vectors/build, standalone validation, matched retrieval,
independent citation/null and stability checks, component controls, NVIDIA naming
and checking, labelled validation, hierarchy example, tables/figures/PDF, ZIP.

The main hierarchy keeps at most 12 coarse and 50 fine groups in total. Its chosen
update mode permits changes to at most 10% of old scientific paths. Adaptive topic
birth/split/merge is disabled; fixed IDs and node growth must not be described as
measured discovery of new scientific fields.

The two offline component controls use the same vectors, years and search:
no_native_cost removes the native scientific cut term; unbounded_edits removes
the old-scientific edit cap and movement penalty. The latter keeps checkpoint
warm starts. Both preserve provenance, input hyperedges and original search
relations. They receive no transferred explanatory labels. See COMPONENT_CONTROLS.md.

The complete reproduction always uses the compressed archive under annotations/
with exact authenticated responses, even if an API key exists locally. Fresh
labels are an explicit separate `make labels` operation, using NVIDIA_API_KEY in
the ignored .env. Current final checking is NVIDIA-only; Gemini is disabled. Strictly
empty evidence uses a distinct deterministic rule. Changed groups require fresh
annotations, so an old E5 label archive cannot label new MiniLM partitions.

Available stages: make build, validate, metrics, science, components, labels,
label-audit, illustrate, tables, pdf and package, always with the same PYTHON and
OUT. A full build selects a timestamped directory if OUT already exists. After a label/API interruption, continue with
labels, label-audit, illustrate, pdf and package. Successful requests are reused.
After a component failure, inspect its log and recorded state before rerunning;
completed component results authenticate their inputs and artifacts on reuse.

make offline runs the local main study without T5; it does not claim complete
label evaluation. Historical E5 vectors are not shipped or used by the command.
The archive is written next to OUT and excludes .env, model weights and caches.

The runner records every subprocess duration, exit status, total wall-clock time
and environment setup duration in OUT-timing.json, with its transcript in OUT.log.
Measurements distinguish reuse of downloaded model files from reuse of text
vectors. Clean reproduction uses fresh text/vector caches and real CPU inference;
downloading weights and installing packages depends on network availability.

The completed clean measurement took 411.10
seconds for the complete command including environment setup; 348.84 seconds
after setup. No new model API request was made.

| Stage | Wall seconds |
| --- | ---: |
| Create environment | 2.06 |
| Install dependencies and project | 60.16 |
| Authenticate archived annotations | 1.00 |
| Vectors and five hierarchy cutoffs | 55.92 |
| Validate raw hierarchies | 2.14 |
| Retrieval | 11.21 |
| Scientific checks | 23.95 |
| Both component controls | 138.06 |
| Replay saved labels | 27.01 |
| Replay blind label audit | 33.28 |
| Validate labelled hierarchies | 2.34 |
| Hierarchy figure | 3.40 |
| Report | 1.78 |
| Verify and create ZIP | 48.76 |

The Git checkout keeps the full measured transcript and machine-readable timings
under results/verification/. Each new execution independently writes OUT.log and
OUT-timing.json next to its generated run directory.
