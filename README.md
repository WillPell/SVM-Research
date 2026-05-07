# Swallowing Aspiration Detection (SVM)

An audio-based pipeline for detecting aspirated swallows from clinical recordings. The pipeline converts raw audio to mono, builds annotated HDF5 label files from CSV timestamps, segments out individual swallow / non-swallow clips, extracts Spectral Subband Centroid (SSC) features, and trains a Support Vector Machine to classify each swallow as aspirated (`1`) or non-aspirated (`0`).

The model is trained on the **VFSS** (paediatric videofluoroscopic swallow study) dataset and evaluated on the **CFE** (cervical / clinical) dataset.

---

## Pipeline overview

```
Raw stereo .wav  ──►  mono .wav  ──►  labels.h5  ──►  per-swallow .wav clips  ──►  SSC features  ──►  SVM
   (cfe / vfss)       (Convert_*)     (h5gen_*)        (swallow_segment_*)        (ssc / SVM.py)
```

| Step | Script | Purpose |
| --- | --- | --- |
| 1 | `Convert_single_channel (cfe).py` | Walk per-participant folders in `cfe/` and convert each `.wav` to mono. |
| 1 | `Convert_single_channel (paed).py` | Convert flat-folder paediatric VFSS recordings to mono. |
| 2 | `h5gen_16k_final (cfe).py` | Read the CFE swallow-timestamp CSV, locate each mono `.wav`, and write start/stop times, sample indices, raw audio and a per-sample class array into `labels.h5`. |
| 2 | `h5gen_16k_final (paed).py` | Same as above for the VFSS dataset. Handles `mm:ss.sss` and `mm.ss.ms` timestamp formats and common typos. |
| 3 | `swallow_segment (3).py` | Using `labels(cfe).h5`, cut individual swallow segments out of each recording. Extends each swallow's end into the following gap (up to 0.5 s) so trailing acoustic events are preserved. Also produces a concatenated non-swallow file per participant. |
| 3 | `swallow_segment_nonSwallow.py` | Stripped-down variant that only emits non-swallow audio. |
| 4 | `ssc (4).py` | Single-dataset SVM trainer (CFE-only, train/test split). |
| 4 | `SVM.py` | Cross-dataset trainer: trains on VFSS, tests on CFE, and writes predictions back into a copy of the CFE labelling spreadsheet. |
| - | `autoCompare.py` | Diagnostic — prints byte sizes of swallow / non-swallow segments per participant. |

---

## Repository contents

```
Scripts/
├── Convert_single_channel (cfe).py     # stereo → mono (CFE, per-participant folders)
├── Convert_single_channel (paed).py    # stereo → mono (VFSS, flat folder)
├── h5gen_16k_final (cfe).py            # build labels.h5 from CFE CSV
├── h5gen_16k_final (paed).py           # build labels.h5 from VFSS CSV
├── swallow_segment (3).py              # cut swallow / non-swallow .wav clips
├── swallow_segment_nonSwallow.py       # cut non-swallow only
├── ssc (4).py                          # SVM, single-dataset (CFE)
├── SVM.py                              # SVM, train VFSS / test CFE + write predictions to xlsx
├── autoCompare.py                      # segment-size diagnostic
├── labels.h5                           # generated: VFSS labels + raw audio
├── labels(cfe).h5                      # generated: CFE labels + raw audio
├── test_with_updated_predictions.xlsx  # output of SVM.py
├── step_by_step.txt                    # design notes on the SSC feature extraction
└── README.md
```

---

## Expected directory layout

The scripts use absolute paths under `…/Stephen's Research/`. Either keep that layout or edit the paths at the top of each script.

```
Stephen's Research/
├── cfe/                              # raw CFE recordings (one folder per participant)
│   └── study 2_clean_swallow sounds.csv
├── paed_vfss/                        # raw VFSS recordings (flat folder of .wav)
│   └── vinod_edited_spreadsheet-VFSSdatset-22Aug2022.csv
├── Swallowing1.csv                   # CFE aspiration labels
├── mono_audio(cfe)/                  # output of Convert_single_channel (cfe).py
├── mono_audio_vfss/                  # output of Convert_single_channel (paed).py
├── swallowing_segments_cfe/          # output of swallow_segment (3).py
├── swallowing_segments_vfss/         # (same script, run against VFSS)
├── non_swallowing_segments_cfe/
└── Scripts/                          # this folder
```

---

## CSV / label conventions

**VFSS labels CSV** (`vinod_edited_spreadsheet-VFSSdatset-22Aug2022.csv`):

- `Participant` — e.g. `P9`
- `Total swallows` — integer
- `Swallow Number {i} Start` / `Swallow Number {i} Stop` — timestamps (numeric seconds, `mm:ss.sss`, or `mm.ss.ms`)
- `Aspirating` — `0` / `1`
- `Which Swallows` — comma-separated swallow numbers that aspirated, e.g. `2, 5`

**CFE labels CSV** (`Swallowing1.csv`):

- `Participant` — e.g. `P64`
- `Aspirating` — comma-separated swallow numbers that aspirated (different schema from VFSS)
- `Classifier's Pred` — column overwritten by `SVM.py` with model predictions

**HDF5 layout** produced by `h5gen_16k_final*.py`:

```
raw/{participant}_1                    # int16 audio samples
labels/{participant}/start/times       # float seconds
labels/{participant}/stop/times
labels/{participant}/start/samples     # int sample indices
labels/{participant}/stop/samples
labels/{participant}/class             # int8 per-sample 0/1 mask
```

---

## Feature extraction

Per `step_by_step.txt`, each swallow clip is reduced to a fixed-length feature vector using **Spectral Subband Centroids**:

- Frame the signal at 25 ms windows with a 12.5 ms hop (`winlen=0.025`, `winstep=0.0125`).
- Compute the FFT (`nfft=2048`) and divide the spectrum into 26 mel-spaced subbands (`nfilt=26`, `lowfreq=0`, `highfreq=fs/2`).
- One centroid per subband per frame.
- Aggregate frame-wise centroids into a single vector by concatenating per-subband **mean** and **std** → 52-dim feature vector.

---

## Model

`SVM.py` (cross-dataset):

- Train on VFSS swallow segments, test on CFE swallow segments.
- 80 / 20 train / validation split on VFSS, stratified by class.
- `SMOTE` oversampling on the training fold to address class imbalance.
- `StandardScaler` fit on the (resampled) training features.
- Classifier: `SVC(C=1, gamma='auto', kernel='poly', class_weight='balanced')`.
- Outputs validation + test confusion matrices and classification reports, lists files predicted as aspirated, and writes per-participant predictions into the `Classifier's Pred` column of `test_with_updated_predictions.xlsx`.

`ssc (4).py` is a simpler single-dataset variant (CFE-only train/test split).

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy h5py librosa soundfile \
            python_speech_features scikit-learn imbalanced-learn openpyxl
```

A `.venv/` and `myenv/` are present in the folder but are local environments and not committed/portable.

---

## Running the pipeline

End-to-end, for each dataset:

```bash
# 1. Stereo → mono
python "Convert_single_channel (cfe).py"
python "Convert_single_channel (paed).py"

# 2. Build HDF5 labels
python "h5gen_16k_final (cfe).py"
python "h5gen_16k_final (paed).py"

# 3. Cut per-swallow .wav clips (edit paths inside the script to switch dataset)
python "swallow_segment (3).py"

# 4. Train + evaluate
python SVM.py
```

`SVM.py` prints validation accuracy / confusion matrix / classification report, then the same on the held-out CFE test set, and finally writes `test_with_updated_predictions.xlsx`.

---

## Notes & caveats

- All script paths are hard-coded absolute paths — edit them before running on a different machine.
- The two datasets use **different label schemas** (`Which Swallows` in VFSS vs. `Aspirating` as a list in CFE). Both `h5gen_*` and `SVM.py` parse them differently.
- VFSS participant IDs in audio filenames take the form `MBS {n}_mono.wav`; CFE filenames are `P{n}_mono.wav` after conversion. The segmenter strips the prefix and uses `{n}_{swallow_index}.wav` for output clips.
- `swallow_segment (3).py` extends each swallow's end into the following silence (max 0.5 s) so cut-off acoustic tails are retained — the corresponding non-swallow audio is shortened by the same amount to avoid overlap.
- Class imbalance is severe (most swallows are non-aspirated). `SMOTE` + `class_weight='balanced'` are both applied; tune `C`, `gamma`, and the kernel if the operating point needs to shift.
