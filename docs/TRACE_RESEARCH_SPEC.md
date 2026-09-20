# TRACE Research Specification

## 1. Identity and research question

TRACE is an evidence-led image forensic analysis engine. It estimates whether an image contains learned forensic evidence associated with synthetic imagery; it does not establish provenance or authenticity.

The primary research question is:

> Does fusing independent spatial and frequency-domain evidence improve detection and cross-generator generalization compared with either signal alone?

The secondary question is:

> How stable are those signals under common image transformations?

This keeps TRACE focused on one image-forensics problem while making the evaluation stronger than a single classifier accuracy number.

The claim is deliberately narrow: TRACE is a controlled study of spatial and frequency-domain forensic evidence, their fusion, cross-generator generalization, and transformation stability. It is not a universally reliable AI-image detector.

## 2. Final architecture

```text
Uploaded image
    |
    v
Validated preprocessing
    |-----------------------|----------------------|
    v                       v                      v
Spatial branch        Frequency branch       Perturbation harness
RGB / learned         FFT or DCT statistics   JPEG, resize, blur,
visual features       and learned features    noise, controlled replay
    |                       |                      |
    |                       |              repeated branch inference
    |                       |                      |
    +----------- structured evidence ------------+
                            |
                            v
                   Evidence fusion
              baseline + learned ablation
                            |
                            v
            calibration / confidence estimate
                            |
                            v
              visual evidence and report API
```

The perturbation harness is deliberately separate from the classifier. It measures robustness and evidence stability; it is not treated as a third proof signal and it does not receive a hand-written weight in the final score.

## 3. Forensic branches

### Spatial branch

Use a pretrained lightweight vision backbone or a documented CNN baseline. Fine-tune only after the data split is fixed. Produce:

- a spatial logit and branch score;
- a Grad-CAM or equivalent activation map from the final convolutional layer;
- optionally, top suspicious regions after thresholding the normalized map.

The map is a model attribution, not a ground-truth manipulation mask. The report must say that clearly.

### Frequency branch

Treat frequency evidence as a hypothesis, not an assumed improvement. Start with deterministic grayscale FFT or block-DCT statistics and a simple classifier. Only add a learned frequency encoder if the deterministic baseline shows a meaningful, repeatable signal that merits more capacity.

Candidate statistics include:

- radial-band energy ratios;
- low/mid/high-frequency energy;
- spectral slope or residual statistics;
- a compact frequency encoder if the ablation justifies it.

Frequency analysis earns a place in the final system only if it beats trivial/statistical baselines and provides a repeatable gain or complementary cross-generator behavior. Do not display an FFT image as evidence without reporting the statistics used.

### Perturbation analysis

Keep this as a first-class report section and a robustness experiment, but not as an independent classifier branch. Run a fixed small protocol:

- original;
- JPEG Q90, Q70, Q50;
- resize down/up;
- optional mild blur or noise.

For each transformation, record both branch scores and the fused score. Report absolute score change, rank stability, and which branch degraded. A score change under JPEG is not evidence of synthesis by itself.

## 4. Evidence fusion

Implement two fusion methods:

1. **Fixed baseline:** concatenate or average standardized branch logits using weights selected on a validation set only. Never hand-tune weights against the test set.
2. **Learned fusion:** concatenate spatial and frequency embeddings plus selected frequency statistics, then train a small MLP or logistic-regression head. Keep the perturbation features out of the first learned fusion experiment; use them for robustness analysis.

Compare spatial-only, frequency-only, fixed fusion, and learned fusion. Report mean and standard deviation over at least three seeds when compute permits. The contribution is the comparison and generalization evidence, not the novelty of an MLP.

## 5. Dataset and split design

The unit of splitting is the source or generator, not an individual filename. Remove duplicate or near-duplicate images before splitting.

Minimum defensible design:

```text
Train:       real_train + Generator A + Generator B
Validation:  real_val + held-out samples from A/B
Test seen:   real_test + A/B test samples
Test unseen: real_test + Generator C only
```

Generator C must not influence model selection, threshold selection, normalization statistics, fusion weights, hyperparameter tuning, architecture selection, early stopping, feature selection, or any other development decision. It is opened only once for the final held-out evaluation. Record generator names, licenses, counts, resolution, and preprocessing in a manifest committed without image files.

Dataset confounding is a primary research risk. Match real and synthetic data by resolution, aspect ratio, file format, compression quality, content category, and collection process wherever possible. Keep source metadata and acquisition details in the manifest. Run shortcut checks before trusting model results: a metadata-only classifier, a low-level image-statistics classifier, and a source-folder or compression-quality probe should not achieve suspiciously high performance. If they do, rebalance or redesign the dataset before interpreting the neural model.

The current `train/{real,FAKE}` and `test/{REAL,FAKE}` dataset can support a binary baseline, but it cannot support the unseen-generator claim until the fake class is separated by generator provenance.

## 6. Score language, calibration, and confidence

Use three distinct concepts:

- **Synthetic evidence score:** a normalized 0-100 presentation of fused model evidence.
- **Model prediction:** the thresholded class selected using validation data.
- **Confidence/calibration:** reliability measured with validation calibration, such as temperature scaling, expected calibration error, and reliability diagrams.

Until calibration is implemented and reported, do not call the score a probability. The UI should say `Synthetic evidence score: 76 / 100`, not `76% probability`.

A score near the decision boundary should be reported as ambiguous, not forced into an authoritative label.

## 7. Research order and decision gates

Implement and validate the research pipeline in this order:

1. Dataset manifest, provenance records, duplicate checks, matching checks, and fixed train/validation/seen-test/unseen-test boundaries.
2. Trivial and statistical baselines, including metadata/source probes and simple image-statistics classifiers.
3. Deterministic FFT/DCT statistics with a simple frequency classifier.
4. Spatial baseline.
5. Fixed fusion using validation-only decisions.
6. Learned fusion.
7. Final Generator C evaluation with no further development decisions afterward.
8. Transformation robustness experiments.
9. Calibration and confidence reporting.
10. Explainability and structured report generation.
11. Frontend/reporting improvements after the research outputs are trustworthy.

Every major component needs a keep/remove test:

| Component | Keep only if the experiment shows |
| --- | --- |
| Frequency statistics | Better than trivial baselines or complementary cross-generator performance |
| Learned frequency encoder | Repeatable validation gain over deterministic statistics without a seen-generator-only gain |
| Spatial backbone | Meaningful improvement over statistical and frequency baselines on held-out data |
| Fixed fusion | Better or more stable generalization than the strongest single branch |
| Learned fusion | Improvement over fixed fusion on validation and held-out generators, not just training data |
| Perturbation suite | It reveals measurable stability/degradation patterns useful for model assessment |
| Grad-CAM/frequency maps | Attributions are technically faithful and do not claim ground-truth localization |
| Calibration | Reliability improves over raw logits and remains valid on the declared evaluation protocol |

If a component fails its gate, remove it or report the negative result. Do not tune the system until it produces a positive result.

## 8. Explainability without hallucination

The model must emit a structured evidence record before any prose is generated:

```json
{
  "spatial_score": 0.84,
  "frequency_score": 0.71,
  "frequency_statistics": {"high_band_ratio": 0.42},
  "perturbation_curve": [{"transform": "jpeg_q70", "score": 0.73}],
  "prediction": "synthetic_evidence",
  "confidence": "moderate",
  "spatial_regions": [],
  "frequency_bands": []
}
```

The deterministic report generator should be the default. It may select from templates whose claims are directly tied to populated fields. An LLM is optional and must receive only this structured record, cite the fields it used, and never invent regions, artifacts, or causal explanations.

## 9. Report contract

The final report should contain:

- overall synthetic evidence score and ambiguity band;
- spatial, frequency, and fusion scores;
- evidence agreement, such as `2 of 2 branches support synthetic evidence`;
- calibration status and confidence label;
- spatial attribution map and frequency spectrum/band chart;
- perturbation curve and stability summary;
- training distribution and held-out generator context;
- a short evidence-grounded explanation;
- limitations and an explicit statement that the result is not proof of provenance.

## 10. Evaluation matrix

At minimum, publish:

| Question | Comparison |
| --- | --- |
| Does frequency help? | spatial-only vs frequency-only vs fused |
| Does fusion generalize? | seen-generator vs Generator C |
| Is it robust? | original vs JPEG/resize/blur/noise |
| Is confidence meaningful? | calibration curve and ECE |
| Is the result stable? | multiple seeds and confidence intervals where feasible |

Accuracy alone is insufficient. Include balanced accuracy, precision, recall, ROC-AUC, and per-generator results. For the unseen test, report the performance drop from seen to unseen rather than hiding it.

## 11. Scope by milestone

### MVP

- dataset manifest and confound audit;
- trivial/statistical baselines;
- curated real plus two synthetic sources;
- deterministic FFT/DCT frequency baseline;
- spatial baseline;
- fixed fusion;
- held-out generator split;
- JPEG Q90/Q70/Q50 robustness curve;
- balanced accuracy, ROC-AUC, and confusion matrix.

### Research version

- learned fusion ablation only if fixed fusion earns its keep;
- learned frequency encoder only if deterministic frequency evidence earns its keep;
- pretrained spatial backbone comparison;
- calibration with a validation set;
- Grad-CAM and frequency-band visualization;
- three-seed summary and per-generator reporting;
- structured report API and stored hashes.

### Stretch goals

- resize/blur/noise perturbation suite;
- region-level attribution quality checks;
- PostgreSQL history and experiment dashboard;
- cloud deployment and CI experiment jobs.

## 11. What not to build now

Do not add video, audio, EXIF, camera fingerprints, multi-modal fusion, an LLM detector, or RAG at this stage. RAG is explicitly deferred until the core experiments establish that the measured evidence is trustworthy; it is not part of the detector.

The defensible contribution is a controlled study of whether independent spatial and frequency evidence improves cross-generator generalization, with transformation stability and calibrated reporting. A negative result is valid: if frequency or fusion does not improve cross-generator performance, TRACE should say so and remove or demote that component rather than engineering a positive result.
