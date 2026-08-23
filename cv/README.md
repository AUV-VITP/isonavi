# Optical perception (AEGIS)

Optical object detection and condition classification for underwater
inspection: a YOLO11n detector and a MobileNetV3 condition classifier, with
trained ONNX weights and evaluation reports.

**This work is deliberately not part of the isonavi proposal.** That is a
scoping decision about the proposal, not a judgement of the work, and the
reasoning belongs on the record rather than in anyone's head.

## Why it is out of scope here

The proposal rests on a single claim: in monsoon flood conditions the water
carries around 3.2 g/L of suspended sediment, visibility is effectively zero,
and therefore the vehicle has to perceive acoustically. Every design decision
in the report follows from that, from the sonar payload to the geometric
detector that needs no labelled imagery.

Presenting an optical detector as a capability of the same vehicle would
contradict the argument that justifies the vehicle. A reviewer would be right
to ask why a system built because cameras cannot see is being sold partly on
cameras, and there is no good answer while the mission is a flooded river.

Two further reasons, both smaller:

- The acoustic detector claimed in the report reaches 99.0 percent mAP at IoU
  0.5 on a held out split of **real** sonar. The optical models sit
  substantially below that on their own datasets, so adding them lowers the
  weakest number in the submission rather than raising anything.
- Nothing connects these models to the vehicle. There is no code path from the
  autonomy stack, the mission logic or the payload to either network, so they
  could not be demonstrated on the bench even if they were in scope.

## Where it does become relevant

Clear water. The statutory driver in the report, IRC:SP:35 and the Dam Safety
Act 2021, mandates periodic inspection of submerged structures, and much of
that inspection happens outside flood season or during reservoir drawdown,
when optical imaging is not only viable but preferable for surface condition:
cracking, spalling, exposed reinforcement, biofouling. That is a real adjacent
market and a genuine second mission profile for the same hull.

Making it part of a proposal would need three things this work does not yet
have: a defined clear-water mission with its own success criteria, evaluation
on imagery representative of Indian reservoir conditions rather than general
underwater datasets, and an integration path into the vehicle's payload bay
and compute budget.

## What is here

| Path | Contents |
| --- | --- |
| `notebooks/` | training notebooks for the detector and the classifier |
| `push/` | Kaggle kernel packaging for both |
| `results/Detector/` | trained detector weights and evaluation report |
| `results/Classifier/` | trained classifier weights and evaluation report |
| `AEGIS_Today_Scope.md` | the scope the work was built against |

The models load and run. They are kept in the repository because they are real
work and because the clear-water case above is worth returning to, not because
the submission depends on them.
