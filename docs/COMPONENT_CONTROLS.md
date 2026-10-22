# Controlled component comparisons

These diagnostics extend the existing CPU study. They do not replace its method,
retune parameters against answers, or promise a positive result.

## Fixed design

All variants share source data, five cutoffs, exact cached MiniLM representations,
12/50 display caps, solver seeds, cutoff policy, initial balanced builder and
retrieval formulas. The settings were saved before the control predictions were
generated. The questions are inspected development data, not an unseen test.

| Variant | Scientific native cut term | Old-scientific updates |
|---|---|---|
| Full | Historical gamma 0.03 | At most 10% editable; movement penalty 0.05 |
| No native cost | Zero gamma | Same edit cap and penalty |
| Unbounded edits | Same positive gamma | No old-scientific cap; zero movement penalty |

Zero native cost does **not** delete hyperedges from the supplied graph. Their
provenance still determines metadata attachment, and the same source/native links
remain available to retrieval. This removes one construction objective term,
not every structural signal in the pipeline.

Unbounded edits do **not** rebuild each year from scratch. Warm starts, persisted
vectors/affinities, fixed topic slots and old metadata paths remain. The initial
partition must exactly equal the full method's initial partition. This control
removes the cap and penalty together as the chosen temporal-control mechanism;
it cannot identify their individual effects or interactions with other settings.

## Measurements and provenance

`python -m tkh.cpu_components RUN` builds the controls with no model/API call.
Cached document IDs/texts, query ordering, encoder and vector normalization must
match. Every cutoff is validated and its multiway quotient independently checked.
Old vectors, affinity blocks, editable domains and actual moved paths are audited.
The held-out citation/null score uses the main study's same rule and null seeds.
Five 10%-edge-removal initial rebuilds are performed per control. For zero native
cost, scientific membership insensitivity to such deletion is expected: the
removed relation term already has zero weight, so ARI 1 is not evidence of a
better scientific taxonomy.

Common predictions cover BM25/source/native plus group transitions, mixed seeds
plus the same transitions, and true coarse-first 2/8, 8/32 and all-branch search.
All control predictions are serialized under one lock before opening reference
answers. Paired differences retain gained/lost targets and conditional question
bootstrap intervals. Shared questions and one chronology are not independent
experimental replications. Neither control inherits the full method's generated
labels or their T5 ratings.

The generated report table shows final fine citation excess, descriptive mean
cross-year fine ARI and 8/32-tree hits out of all reference occurrences. These
measures answer different questions; none is a percentage of correct topics.
Complete results and immutable inputs are in `RUN/components/`, with the summary
also in `RUN/metrics.json`. Reuse checks source/input and artifact hashes; an
incomplete directory is not silently accepted as a successful experiment.

## Hierarchy illustration

The example uses the last cutoff's authenticated main labels: the first coarse
persistent ID, the first two fine child IDs and the first method ID per child
(typed fallback if no method is present). Real parent/member relationships are
checked. The selection is independent of group quality. The small figure omits
other branches and may shorten long strings; full strings/IDs and source hashes
are saved in `metrics.json`. It illustrates the actual partition, including
unintuitive members, rather than manually presenting a polished taxonomy.

## Primary references consulted

- [Fostiropoulos and Itti, ABLATOR, AutoML 2023](https://proceedings.mlr.press/v224/fostiropoulos23a.html): component comparisons and persistent experimental evidence; no ABLATOR code or dependency is imported.
- [Matplotlib, drawing boxes](https://matplotlib.org/stable/gallery/shapes_and_collections/fancybox_demo.html): public drawing API used for the static original illustration.
- [Scientific Agent Skills](https://arxiv.org/abs/2609.00065): procedural experimental-design, visualisation and evidence-review guidance, disclosed in AI_USAGE.md and the report.

The current code and results remain authoritative; these references do not
certify validity, independence or scientific superiority of the method.
