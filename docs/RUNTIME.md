# Measured one-command execution

The measured public command was `make all`, executed in a clean copy of the Git-visible files on 22 October 2026. All twelve stages exited with status 0.

- Complete shell wall time, including environment creation and installation: **411.10 seconds**.
- Pipeline after environment setup: **348.84 seconds**.
- New isolated Python environment, fresh exact-text/vector cache, real pinned MiniLM CPU inference.
- The pinned model files and pip wheel cache already existed. Downloading model weights on a new machine is outside this measurement.
- No NVIDIA or Gemini key was present. Naming and checking replayed exact authenticated archived responses; no new LLM calls.

| Setup step | Wall seconds |
| --- | ---: |
| create_environment | 2.06 |
| install_dependencies | 59.24 |
| install_project | 0.92 |

| Pipeline stage | Wall seconds |
| --- | ---: |
| Authenticate/unpack saved annotation evidence | 1.00 |
| CPU vectors and five hierarchy cutoffs | 55.92 |
| Validate raw hierarchies | 2.14 |
| Common flat, graph and tree retrieval | 11.21 |
| Coherence/null, deletion and temporal checks | 23.95 |
| Both fixed component controls | 138.06 |
| Authenticate/replay 310 saved labels | 27.01 |
| Authenticate/replay blind label audit | 33.28 |
| Validate published labelled hierarchies | 2.34 |
| Draw actual hierarchy excerpt | 3.40 |
| Generate tables, figures, Markdown and PDF | 1.78 |
| Authenticate and write full local ZIP | 48.76 |

These are measurements on this Linux x86_64 machine with four numerical worker threads, not runtime guarantees for arbitrary hardware or network conditions. The full command also builds the two existing controls and packages all generated intermediate evidence; it is broader than construction alone.

Raw evidence: [timings](../results/verification/local-timing.json), [complete transcript](../results/verification/local-timing.log), [external shell timer](../results/verification/local-wallclock.txt), [source hashes and scope](../results/verification.json).

The full test suite separately passed 472 tests in 37.23 seconds. The regenerated retrieval reports exactly match the previous release. No new method/parameter experiment was added.
