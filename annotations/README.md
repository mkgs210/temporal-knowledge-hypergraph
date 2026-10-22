# Archived NVIDIA annotations

`nvidia_labels.tar.gz` contains the exact saved prompts, requests, raw responses,
310 completed group labels, the 20-item blind audit and its controls. Earlier
failed controls are retained. These are measured annotation records, not model
weights or precomputed construction vectors. The adjacent SHA-256 file is checked
before extraction into the ignored `.cache/tkh/annotations/` directory.

`make all` always replays these records without contacting an LLM API. It rebuilds
the graph hierarchy and sentence vectors from the supplied data first, then
requires annotation evidence and group membership to match exactly. Changed
groups cannot inherit these labels. Fresh annotations require an explicit
`make labels` invocation with an NVIDIA key; they are outside the timed release
reproduction. Gemini is not used by that reproduction.
