# Multimodal Emotion Recognition (Audio + Video + EEG)

Training, evaluation, and reliability-audit code for a multimodal emotion recognition paper (Results in Engineering, under revision). Dataset: EAV (EEG-Audio-Video), 42 subjects, 5 emotions.

## Setup

1. **Get dataset access** — see "Dataset access" below. Request the raw EAV files directly from the original providers.
2. **Upload your data to Google Drive**, organized however you like (e.g. `MyDrive/THESIS/Audio Data/`, `MyDrive/THESIS/Video Data/`, `MyDrive/THESIS/EEG Data/`). If your raw files are zipped, keep the zip in Drive — no need to unzip it yourself.
3. **Open a notebook in Google Colab**: `File → Open notebook → GitHub`, paste this repo's URL, pick the notebook.
4. **Turn on a GPU**: `Runtime → Change runtime type → GPU`.
5. **Run the first few setup cells** of the notebook. Each notebook handles input differently — check its first 2-3 cells before running the rest:
   - `Audio.ipynb`, `Video.ipynb`, `EEG.ipynb`, `Fusion.ipynb` mount Google Drive (`drive.mount('/content/drive')`) and read paths from there. `EEG.ipynb` additionally copies its data from Drive into Colab's local `/content` folder once per session for faster I/O (you'll see a "Drive extraction already complete" message on reruns) — if your zip isn't extracted yet, the notebook extracts it automatically the first time.
   - `Testing.ipynb` does **not** use Drive — it expects you to upload a few specific files directly into the Colab session via the "Choose Files" prompt: `audio_features.h5`, `subject_split.json`, `scalers.pkl`, `best_audio_model.pt`. These four are produced by running `Audio.ipynb` first.
6. **Update any hardcoded Drive paths** in the notebook's setup cells to match wherever you placed your own data.
7. Run the rest of the cells top to bottom.

## Files

| File | What it does | Depends on |
|---|---|---|
| `Audio.ipynb` | Trains the Audio branch (UC1) — acoustic + Wav2Vec2 + semantic (Whisper/RoBERTa, emotion-word-scrubbed) streams, attention-fused. Fixed 30/6/6 subject split, seed=42. Exports `uc1_predictions.npz`. | EAV audio files |
| `Video.ipynb` | Extracts video features (ViT + landmarks + optical flow) and trains the MS-TAP model (UC2) via 7-fold subject-disjoint CV across all 42 subjects. Exports fold checkpoints + `uc2_predictions_shared_subjects.npz`. | EAV video files |
| `EEG.ipynb` | Trains the EEG branch (UC3) — EEGNet (raw) + DE + PSD streams, attention-fused. Fixed 30/6/6 split (independent of Audio's split — different seed-shuffle ordering). Exports `uc3_predictions_v2.npz`, `uc3_results_v2.json`. | EAV EEG files |
| `Fusion.ipynb` | Loads all three branches' saved predictions, aligns them on the 4 subjects shared by all three test sets (15, 16, 18, 41), computes decision-level fusion (UC4–UC7, static + confidence-weighted). | Outputs of the three notebooks above |
| `Testing.ipynb` | The Section 3B reliability audit — branch ablations, standalone single-branch baselines, the TF-IDF content-only control, the Wav2Vec2-vs-TF-IDF agreement test, bootstrap confidence intervals, majority-vote baseline, and the EEG stream ablation. | Outputs of Audio.ipynb and EEG.ipynb |

## Run order

1. `Audio.ipynb`, `Video.ipynb`, `EEG.ipynb` — can run independently/in any order (each trains one branch from raw EAV data).
2. `Fusion.ipynb` — needs all three branches' saved predictions.
3. `Testing.ipynb` — needs Audio's and EEG's saved model/predictions; run last.

## Data

- `subject_split.json` — Audio's exact saved 30/6/6 subject split (seed=42), included for reproducibility.
- `uc3_results_v2.json`, `uc3_cv_results_v3.json` — EEG's saved evaluation results, including per-stream attention weights.
- `uc1_predictions.npz`, `uc3_predictions_v2.npz`, `uc2_predictions_shared_subjects.npz` — saved test-set predictions for each branch.

## Dataset access

Raw EAV recordings are **not** included — the dataset is access-restricted under a Data Usage Agreement / NDA (Lee et al., 2023; Zenodo DOI: 10.5281/zenodo.10205702). Request access directly from the original providers. The prediction arrays and result JSONs in this repo are derived, aggregate outputs and are **not** covered by that NDA.

## License

MIT — see `LICENSE`.
