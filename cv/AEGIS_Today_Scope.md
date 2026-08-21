# AEGIS AUV — Today's ML Build Scope

**Purpose of this document:** context for an agentic coding session (Claude Code) building and pushing
Kaggle notebooks. This is a deliberately trimmed slice of the full AEGIS ML program (see the uploaded
`AEGIS_AUV_Authentic_Dataset_Strategy.pdf` and `AEGIS_AUV_ML_Training_Specification.pdf` for the complete
long-term plan). **Only what's below is in scope for today.** Do not add tracking, segmentation, sensor
fusion, enhancement, sonar, or additional datasets/classes beyond what's specified here, even though the
source documents describe them — those are explicitly deferred to a later phase.

**Time budget:** one day. **Compute budget:** ~60 GPU-hours across two Kaggle accounts (see Compute Plan).

---

## 1. Scope: exactly two models, nothing else

| Model | Task | Architecture | Status today |
|---|---|---|---|
| **Model A** | Object detector | YOLO11n (Ultralytics, ~2-6M params) | Build + train |
| **Model B** | Condition classifier | MobileNetV3-Small (timm, ImageNet-pretrained) | Build + train |

Explicitly **not** built today: multi-object tracker, damage segmentation model, temporal evidence
aggregation, sensor-fusion risk engine, image-enhancement module, sonar branch. These depend on Model A/B
existing first and are separate future sessions.

---

## 2. Frozen taxonomy for today

Do not expand these lists during implementation. If a dataset offers more classes than this, map the
extras to the closest class below or drop them — do not add new classes on the fly.

### Model A — detector classes (4)
| Class | Rationale |
|---|---|
| `fish` | R1 (marine organism detection) |
| `pipe` | R3 (pipeline detection — core AEGIS use case) |
| `debris` | R2/R5 (general object + marine debris) |
| `structure` | R2/R7 (wreck / underwater structure / infrastructure member) |

Everything else in source datasets is background/unlabeled, not a 5th class.

### Model B — condition classes (3)
| Class | Rationale |
|---|---|
| `normal` | No visible defect |
| `damaged` | Any visible defect (crack, corrosion, deformation, fracture — not sub-typed yet) |
| `unknown` | Insufficient visual evidence — used instead of guessing, per the spec's labeling rule |

The full 7-class condition taxonomy from the spec (corroded/cracked/deformed/fractured/biofouled/severe/
unknown) is the correct long-term target but requires the project's own AEGIS-DAMAGE dataset, which does
not exist yet. Today's 3-class version is the honest starting point given only proxy data is available.

**Labeling rule preserved from the spec:** object identity and condition are separate models/outputs.
Never fold condition into a detector class like `damaged_pipe` — the detector finds `pipe`; Model B
separately says `damaged`.

---

## 3. Datasets for today (3 sources, not the full 13-source catalog)

| Dataset | Maps to class(es) | Access | License note |
|---|---|---|---|
| **FathomNet** | `fish`, `structure` | Public API (`fathomnet` Python client), query by concept | Item-level licensing — check per downloaded subset; fine for prototype use |
| **SeaClear** (TU Delft, 8,610 images, 40 raw categories) | `debris`, `structure` | `research.tudelft.nl` / `github.com/adjuras/seaclear-dataset` | CC BY 4.0 — permissive, attribution required |
| **SubPipe-Mini** (Zenodo subset, ~6.1GB) | `pipe` | `zenodo.org/records/12666132` | Explicit reuse/copyright text tied to OceanScan-MST — fine for research; flag for legal review before any commercial use |

**For Model B (`damaged` proxy):**

| Dataset | Maps to | Access | License note |
|---|---|---|---|
| **SDNET2018** | `damaged` (crack) / `normal` (uncracked) | Kaggle-native: `aniruddhsharma/structural-defects-network-concrete-crack-images` | Open dataset (Dorafshan, Thomas & Maguire 2018) |

**Deliberately excluded today** (from the full catalog, with reason):
- JODD — academic-only CC BY-NC-SA license adds friction for no MVP benefit today
- Fish4Knowledge — video-derived, "High" processing burden per the dataset strategy doc's own table
- Trash-ICRA19 / TrashCan — redundant with SeaClear; both are J-EDI-family (same leakage-control concern the strategy doc flags for JODD)
- MIMIR-UW, VDD-C, AQUALOC — 45GB+ / 100k+ images / ROS-bag processing respectively; not a one-day task
- CODEBRIM (structural damage, multi-label) — real and valuable, but its multi-label format needs the official dataloader to parse correctly; wrong auto-parsing would silently mislabel data. Deferred, not skipped.

---

## 4. Non-negotiables (kept from the spec despite trimming everything else)

1. **Source-aware splits.** Split by dataset/site/video, never by random frame shuffling. Near-identical
   adjacent frames leaking across train/test inflates metrics — this applies even at reduced dataset scope.
2. **Object identity separate from condition.** See taxonomy section above.
3. **Record the license alongside every downloaded source**, even with only 4 sources. One column in a
   simple manifest (dataset name, source, license, class mapping) is enough — no need for the full
   AEGIS dataset-manifest system from the long-term spec today.
4. **Held-out test split, never touched during training**, for honest final metrics on both models.
5. **`unknown` is a valid label, not a bug.** Insufficient evidence maps to `unknown`, not a forced guess.

---

## 5. Training configuration

### Model A — YOLO11n
- Input size: 384px (edge-inference budget, per spec's 2-6M param / low-latency target)
- Epochs: 100-150 (multi-session across Kaggle's ~12h cap is expected and fine)
- Checkpointing: every 3 epochs, `CONTINUE_FROM_WEIGHTS` pattern for cross-session continuation
  (not `resume=True` — that needs the same run folder to persist, which Kaggle doesn't guarantee)
- Metrics: mAP@0.5, mAP@0.5:0.95, precision, recall, per-class recall, held-out test split eval

### Model B — MobileNetV3-Small
- Input size: 224px
- Epochs: 40+ per session, same multi-session checkpoint pattern as Model A
- Metrics: macro F1, per-class precision/recall, confusion matrix

---

## 6. Compute plan — two Kaggle accounts, run in parallel

- **Account 1:** Model A (detector) training
- **Account 2:** Model B (condition classifier) training, run at the same time as Account 1

Rough GPU-hour estimate: detector ~4-8 hrs, classifier ~2-3 hrs. Well inside the 60-hour budget; the
budget buys retries/tuning room, not more scope.

**Note on running your own training on a second account:** if the second account belongs to a friend, it's
fine for them to train their own things on it — using it to extend your own personal compute quota is the
part that sits in a greyer area of Kaggle's terms. Flagging this, not blocking it — it's your call.

---

## 7. Definition of done for today

- [ ] Model A: 4-class YOLO11n, exported ONNX, with mAP/precision/recall/per-class-recall on a genuinely
      held-out (source-isolated) split
- [ ] Model B: 3-class MobileNetV3-Small, exported ONNX, with macro F1 + confusion matrix on a held-out split
- [ ] Both models checkpointed such that training can resume in a later session if needed
- [ ] Simple dataset manifest (source, license, class mapping) recorded for the 4 datasets used
- [ ] Everything else from the two source PDFs remains explicitly out of scope for today