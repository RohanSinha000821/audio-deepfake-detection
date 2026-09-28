# Cross-Dataset Audio Deepfake Detection

This repository is the working codebase for my research on **generalizable audio deepfake detection**. The main question is simple: if a detector is trained on several deepfake datasets, how well does it perform on a dataset that it has never seen during training or model selection?

The project is being developed in two stages:

1. build a clean cross-dataset evaluation pipeline and reproduce strong baselines under the same protocol;
2. implement **Language-Anchored Cross-View Forensics (LACF)**, which combines WavLM and CLAP through a shared forensic concept space.

The current repository has the data/LODO/evaluation infrastructure in place and a complete **WavLM weighted-average baseline**. The first leave-one-dataset-out experiment (F1, with SpeechFake held out) has been completed. LACF itself is not yet implemented on `main`.

---

## 1. Experimental setup

The main benchmark currently uses four datasets:

- **ASVspoof 2019 LA**
- **ASVspoof 5 Track 1**
- **CFAD**
- **SpeechFake**

The experiments follow a **leave-one-dataset-out (LODO)** protocol. For each fold, three datasets are used as source domains and the fourth dataset is kept completely unseen until final evaluation.

| Fold | Source datasets | Held-out target |
|---|---|---|
| F1 | ASVspoof 2019, ASVspoof 5, CFAD | SpeechFake |
| F2 | ASVspoof 2019, ASVspoof 5, SpeechFake | CFAD |
| F3 | ASVspoof 2019, CFAD, SpeechFake | ASVspoof 5 |
| F4 | ASVspoof 5, CFAD, SpeechFake | ASVspoof 2019 |

The held-out target is not used for training, source validation, early stopping, hyperparameter selection, checkpoint selection, or threshold calibration.

Labels are fixed throughout the project as

```text
0 = bona-fide
1 = spoof
```

and larger detector scores always mean **more spoof-like**.

---

## 2. Data manifests

The original datasets use different directory layouts, metadata files, and protocol formats. To keep the training and evaluation code dataset-independent, each dataset is converted to a common manifest representation.

The core fields are:

```text
path, label, dataset, split, utterance_id
```

Additional information such as speaker, attack type, generator, language, codec, or condition is retained when it is available.

Manifest creation follows dataset protocols rather than blindly scanning folders. In particular:

- ASVspoof 2019 is built from the official LA protocols;
- ASVspoof 5 uses the Track-1 protocol entries;
- CFAD preserves its dataset split/condition structure;
- SpeechFake is built from its supplied metadata, with duplicate-path handling and train/dev overlap protection.

Build manifests with:

```bash
python scripts/data/build_manifests.py --dataset asvspoof2019 --data-root "$DATA_ROOT"
python scripts/data/build_manifests.py --dataset asvspoof5    --data-root "$DATA_ROOT"
python scripts/data/build_manifests.py --dataset cfad         --data-root "$DATA_ROOT"
python scripts/data/build_manifests.py --dataset speechfake   --data-root "$DATA_ROOT"
```

The generated manifests are written under `manifests/`.

---

## 3. Data loading and sampling

The shared data layer handles audio loading, resampling, cropping, padding, attention masks, batching, and reproducibility.

For the current WavLM baseline:

- training uses a random continuous **4 s crop**;
- development and final evaluation use the **full utterance**;
- audio is provided to WavLM at **16 kHz**;
- cropping is reproducible across epochs/workers;
- padded regions are excluded during temporal pooling through the attention mask.

The WavLM baseline uses a `DatasetBalancedSampler`. A source dataset is sampled uniformly first, and an utterance is then sampled from that dataset. This prevents a large source corpus from dominating training while preserving the natural class distribution inside each source dataset.

This is intentionally different from the planned LACF sampler, where dataset balancing and class balancing will both be used.

---

## 4. Implemented baseline: WavLM weighted-average backend

The first implemented baseline uses `microsoft/wavlm-base` with a learned weighted combination of the projected convolutional representation and the 12 Transformer-layer representations.

For an utterance, the model uses 13 WavLM representations:

```text
projected CNN representation + Transformer layers 1 ... 12
```

A softmax-normalized scalar weight is learned for each representation. The weighted representation is temporally pooled using the valid attention-mask positions and passed to a `768 -> 2` classifier.

The weighted-average backend contains **1,551 parameters**, but the current experiment also fine-tunes the WavLM encoder; it is therefore not a frozen-feature experiment.

Main training choices:

| Setting | Value |
|---|---:|
| WavLM checkpoint | `microsoft/wavlm-base` |
| Training crop | 4 s |
| Physical batch size | 8 |
| Gradient accumulation | 4 |
| Effective batch size | 32 |
| WavLM learning rate | `2e-5` |
| Backend learning rate | `5e-3` |
| LR decay per epoch | `0.95` |
| Class weights `[bona-fide, spoof]` | `[9, 1]` |
| Early-stopping patience | 5 epochs |
| Precision | BF16 |

Checkpoint selection is based on the **unweighted mean of the source-domain development EERs** rather than a pooled development EER. This keeps a large development set from dominating model selection.

---

## 5. Training

Install the project in a Python 3.12 environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Set the dataset root before building manifests or starting an experiment:

```bash
export DATA_ROOT=/path/to/audio-deepfake-datasets
```

Train the WavLM baseline on F1 with:

```bash
python scripts/train/train_wavlm_wa.py \
  --fold configs/lodo/f1.yaml \
  --config configs/sota/wavlm_wa.yaml \
  --output-dir outputs/wavlm_wa/f1
```

The training script writes both `last.pt` and the source-selected `best.pt`. Training can be resumed from `last.pt` using `--resume`.

---

## 6. Evaluation

Evaluation first scores the source development domains and freezes the source-derived thresholds. Only after this step is the held-out target loaded and scored.

```bash
python scripts/evaluate/evaluate_wavlm_wa.py \
  --fold configs/lodo/f1.yaml \
  --config configs/sota/wavlm_wa.yaml \
  --checkpoint outputs/wavlm_wa/f1/best.pt \
  --output outputs/wavlm_wa/f1/evaluation.json
```

The evaluator reports:

- EER
- AUROC
- APCER
- BPCER
- ACER
- transferred source operating points at 1%, 5%, and 10% APCER

Per-utterance scores are also saved as TSV files for later error and distribution analysis.

---

## 7. Current F1 result

F1 uses ASVspoof 2019, ASVspoof 5, and CFAD as sources, with **SpeechFake completely held out**.

The source-training set contains **246,137** utterances. The three source development sets contain **180,194** utterances in total. The final SpeechFake test set contains **346,307** utterances: **37,242 bona-fide** and **309,065 spoof**.

Training stopped after epoch 9. The best checkpoint was selected at **epoch 4**.

### Source development

| Dataset | EER |
|---|---:|
| ASVspoof 2019 dev | 0.0392% |
| ASVspoof 5 dev | 0.2144% |
| CFAD dev | 0.0417% |
| **Macro source EER** | **0.0984%** |

### Held-out SpeechFake

| Metric | Result |
|---|---:|
| **EER** | **15.63%** |
| **AUROC** | **0.9203** |

Source-calibrated operating points transferred unchanged to SpeechFake:

| Source APCER operating point | SpeechFake APCER | SpeechFake BPCER | SpeechFake ACER |
|---:|---:|---:|---:|
| 1% | 52.44% | 0.17% | 26.31% |
| 5% | 63.79% | 0.046% | 31.92% |
| 10% | 75.15% | 0.008% | 37.58% |

The main observation from F1 is the large gap between source-domain performance and the unseen target. The very small target BPCER together with the much larger APCER indicates that most genuine SpeechFake samples remain easy for the model, while many unseen spoof samples shift toward the bona-fide side of the source-calibrated score space.

This is exactly the failure mode the later LACF experiments are intended to study.

---

## 8. Planned model: LACF

**Language-Anchored Cross-View Forensics (LACF)** is the proposed model for the next stage of the project.

The idea is to avoid relying only on generator-specific spoof artefacts. LACF will compare two independently derived views of the same audio:

- **WavLM Base+** for speech-acoustic evidence;
- **CLAP** for language-grounded audio evidence.

Both views are expressed relative to the same fixed bank of eight forensic text concepts (three bona-fide and five synthetic-generation concepts). Their two concept distributions are then compared through agreement/disagreement features, Jensen--Shannon divergence, and entropy. With eight concepts, this produces a **35-dimensional cross-view relation vector**, followed by a small `35 -> 64 -> 16 -> 1` classifier.

The current `main` branch only contains the LACF package/configuration scaffold. The full LACF model, training code, ablations, and experiments are still to be implemented.

A useful distinction when reading the repository is:

```text
WavLM baseline : WavLM Base, 4 s training crop, dataset-balanced sampling
LACF V1 plan   : WavLM Base+ + CLAP, common segment up to 10 s,
                 dataset- and class-balanced sampling
```

The baseline should therefore remain unchanged while LACF is added as a separate model family.

---

## 9. Repository layout

```text
configs/
  lodo/                 F1--F4 leave-one-dataset-out definitions
  sota/                 baseline configurations
  lacf/                 LACF configuration/ablation scaffold

scripts/
  data/                  manifest generation
  train/                 training entry points
  evaluate/              final evaluation entry points
  analysis/              analysis scripts (planned)

src/audio_deepfake_detection/
  data/                  manifests, datasets, audio loading, samplers
  protocols/             LODO protocol handling
  metrics/               EER, AUROC, APCER/BPCER/ACER, thresholds
  sota/wavlm_backend/    implemented WavLM weighted-average baseline
  lacf/                  LACF implementation scaffold
  training/              reproducibility utilities

tests/
  data/
  metrics/
  protocols/
  sota/
  training/

manifests/               generated canonical metadata
```

---

## 10. Current status

| Component | Status |
|---|---|
| Canonical manifest generation | Done |
| Shared audio/data pipeline | Done |
| Manifest validation | Done |
| LODO F1--F4 definitions | Done |
| Dataset-balanced sampler | Done |
| Dataset + class-balanced sampler | Done |
| EER/AUROC/APCER/BPCER/ACER | Done |
| Source-only threshold calibration | Done |
| WavLM weighted-average baseline | Done |
| WavLM F1 training | Done |
| WavLM F1 SpeechFake evaluation | Done |
| WavLM F2--F4 experiments | Pending |
| AASIST / other comparison baselines | Pending |
| LACF V1 implementation | Pending |
| LACF ablations and multi-seed evaluation | Pending |

---

## 11. Tests

The repository includes unit tests for dataset parsing, manifest validation, audio loading, collation, samplers, LODO target isolation, metrics, threshold selection, reproducibility, and the WavLM backend/evaluator.

Run the test suite with:

```bash
pytest -q
```

---

## 12. Notes on reproducibility

A few choices in the WavLM reproduction are explicitly fixed in the configuration rather than silently assumed:

- WavLM input normalization is disabled;
- SpecAugment is disabled because it is not reported as part of the baseline fine-tuning setup being reproduced;
- the 13 learned layer weights are softmax-normalized;
- layer-wise learning-rate decay and L2 regularization toward the initial WavLM weights are currently disabled because the required coefficients were not specified;
- the target domain is never used to choose checkpoints or thresholds.

These choices are kept explicit so that later experiments can be compared against the same baseline without changing its training recipe.

---

## 13. Research direction

The repository is intentionally being built from the evaluation protocol upward. The immediate goal is to complete the remaining baseline folds under exactly the same leakage-free setup. The next model-development step is then LACF, followed by controlled ablations to determine whether the WavLM--CLAP cross-view relation improves generalization to unseen datasets, generators, languages, and recording conditions.

The F1 WavLM result already shows why this matters: near-perfect source-domain performance does not guarantee reliable behavior on an unseen deepfake corpus.
