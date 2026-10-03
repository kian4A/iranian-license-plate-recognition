# Iranian License Plate Recognition with CNN + CTC

A PyTorch project that reads **cropped Iranian license plates** using a custom convolutional network and Connectionist Temporal Classification (CTC). The workflow includes label cleaning, identity-based splitting, augmentation, training, validation, inference benchmarking, and mistake analysis.

**Author:** Kian Aghmashe  
**Notebook:** [license_plate_recognition_ctc.ipynb](license_plate_recognition_ctc.ipynb)

## Repository contents

| File | Description |
| --- | --- |
| `license_plate_recognition_ctc.ipynb` | Complete notebook with code, explanations, plots, and saved outputs |
| `README.md` | Project overview and Google Colab instructions |

Dataset archives, trained weights, and generated reports are not included. The notebook is preserved as supplied, with its filename renamed.

## Model and workflow

- Normalize label variants and remove invisible joining characters.
- Validate the eight-character format: two digits, one letter, and five digits.
- Split 90%/10% by **unique cleaned plate identity**, keeping images of the same plate in one split.
- Resize RGB crops to **64 × 256** and normalize them; augment training images with mild geometric and color changes.
- Extract features using Inception-style residual branches, depthwise convolution blocks, and channel attention.
- Pool over image height, process the horizontal feature sequence with 1D convolutions, and predict character log probabilities.
- Train with CTC loss without requiring individual character bounding boxes.
- Decode using greedy CTC decoding and measure exact plate accuracy and character error rate (CER).

The model uses convolutional sequence processing rather than an RNN or LSTM. It recognizes already-cropped plates; vehicle or plate detection is a separate task.

## Saved validation results

These values are taken from the notebook's stored outputs and have not been independently reproduced.

| Item | Recorded value |
| --- | ---: |
| Total images | 24,561 |
| Unique cleaned plate identities | 20,905 |
| Training images | 22,161 |
| Validation images | 2,400 |
| Vocabulary size, including CTC blank | 32 |
| Selected checkpoint epoch | 49 |
| Exact validation plate accuracy | 88.54% |
| Validation CER | 2.51% |
| Incorrect validation predictions | 275 / 2,400 |

The validation split is used for checkpoint selection as well as evaluation. These are **validation results**, not independent test-set results. Rare letter classes are highly imbalanced, and some are absent from one split.

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and upload `license_plate_recognition_ctc.ipynb`.
2. Enable a GPU runtime through **Runtime → Change runtime type** (T4 if available).
3. Run cells in order. The dataset cell uses `gdown` to download `LPR.zip` and extracts it into `/content/LPR_FILES`. If `gdown` is unavailable, run `%pip install gdown` first.
4. Verify the extracted files include:
   - `/content/LPR_FILES/detections/` — cropped plate images.
   - `/content/LPR_FILES/valid_samples.csv` — CSV with `image_path` and `label` columns.
5. Review the split and vocabulary checks, then run the training section. The default configuration trains for up to 50 epochs with batch size 32.
6. Run the learning-curve, prediction, speed, and mistake-analysis cells.

The dataset link is supplied in the notebook. Access depends on the source owner's sharing settings. If the link is unavailable, obtain the dataset from its owner and upload/extract it into the expected layout. Dataset provenance and redistribution terms are not documented here.

### Keep checkpoints in Google Drive

By default, `save_folder` is `/content/LPR_checkpoints_accuracy_v2`. Files under `/content` are temporary. To persist model checkpoints and reports, mount your Drive before configuring training:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then change `save_folder` in the training configuration cell to:

```python
save_folder = '/content/drive/MyDrive/LPR_checkpoints_accuracy_v2'
```

The notebook creates that directory. Copy `train.csv` and `val.csv` from `/content/LPR_FILES` to Drive as well if you want to retain the exact split.

## Training configuration

| Setting | Default |
| --- | --- |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0001 |
| Maximum epochs | 50 |
| Gradient clipping | Global norm of 5 |
| Scheduler | ReduceLROnPlateau on validation plate accuracy |
| Scheduler activation | Epoch 8 |
| Early stopping | 12 epochs without a better accuracy/CER checkpoint |
| Checkpoint selection | Highest plate accuracy; lower CER breaks ties |

Training starts a new run and does not automatically resume checkpoints.

## Generated outputs

The notebook saves the following files during execution:

- `train.csv` and `val.csv` in the dataset directory.
- `best_ctc_model.pt` and `best_loss_model.pt`.
- `training_curves.png` and `learning_rate.png`.
- `inference_speed.csv`.
- `validation_mistakes.csv` and `mistaken_samples.png`.

Model weights are separate files; saved notebook outputs do not embed a usable checkpoint.

## Evaluation notes

- Exact plate accuracy requires every character to match.
- CER counts character insertions, deletions, and substitutions relative to target length.
- Greedy decoding does not enforce the expected plate format.
- Inference timing covers the model alone or model plus decoding. It excludes image loading, preprocessing, input transfer, and plate detection.
- The split seed is fixed, but the entire training run is not guaranteed deterministic.
