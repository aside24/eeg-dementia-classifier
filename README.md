# EEG Based Classification of Dementia (Alzheimer's / FTD) vs. Healthy Controls

The onset and presence of Alzheimer's disease and frontotemporal dementia are associated with changes in resting state brain rhythms that can be measured with electroencephalograms (EEG). When a patient is suffering from either disease, a pattern known as "EEG slowing" will frequently appear in their EEG readings. With EEG slowing, power shifts away from faster alpha oscillations toward slower theta and delta waves as neurodegeneration becomes more severe (Babiloni et al., 2020, *Neurobiology of Aging*; Babiloni et al., 2004, *NeuroImage*).

I built a pipeline that takes raw, 19 channel resting state EEGs, extracts spectral power features, and trains a Random Forest classifier to distinguish Alzheimer's disease (AD), frontotemporal dementia (FTD), and healthy control subjects (CN), using the [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504/versions/1.0.9) dataset.

[**Full write up**](https://docs.google.com/document/d/1XijgZJPhEBfPMK6wvMmIFlJXkWANYW_B4j1-LF1HFpo/edit?usp=sharing): methodology, implementation challenges, results, and comparison to published work on the same dataset.

## Results

| Model | Classes | Mean CV accuracy | Naive baseline |
|---|---|---|---|
| Random Forest, 95 features | AD / FTD / CN | **62.5%** | ~41% |
| Random Forest, 95 features | Dementia / Control (binary) | **77.3%** | ~67% |

The 3 class model distinguishes healthy controls well (86% recall) and Alzheimer's reasonably well (72% recall), but struggles specifically with FTD (17% recall), a weakness that matches literature describing FTD's EEG signature as less consistent than Alzheimer's slowing pattern. The binary reframing (dementia vs. healthy) trades diagnostic granularity for a cleaner separation.

A single subject spectral comparison illustrates the underlying effect: a healthy control subject's EEG is dominated by alpha power (50.6% of total), while both an Alzheimer's and an FTD subject show power concentrated almost entirely in delta (~70%), with alpha nearly absent, visible directly in the power spectral density before any classifier is involved.

## Pipeline

```
Raw EEG (.set)
  → Filtering (0.5 to 45 Hz bandpass)
  → Bad channel detection
  → Bad segment detection
  → ICA (eye artifact removal, via proxy EOG correlation)
  → Rereferencing (average reference)
  → Epoching (4 second fixed length windows)
  → Power spectral density (Welch's method)
  → Relative band power extraction (delta/theta/alpha/beta/gamma × 19 channels)
  → Random Forest classification
```

Every step's parameters were chosen and validated against the data itself. See the [full write up](https://docs.google.com/document/d/1XijgZJPhEBfPMK6wvMmIFlJXkWANYW_B4j1-LF1HFpo/edit?usp=sharing) for the rationale behind each choice.

## Repository structure

```
├── data/
│   ├── raw/          # downloaded EEG (not tracked in git, see Quickstart)
│   └── processed/     # preprocessed epochs + final feature table (regenerable)
├── src/eeg_dementia/
│   ├── preprocessing.py   # the 6 step pipeline, as a single reusable function
│   └── features.py        # Welch's method PSD + relative band power extraction
├── scripts/
│   └── build_dataset.py   # batch processes all subjects, with per subject error handling
├── notebooks/
│   ├── 01_explore_raw_data.ipynb    # raw signal exploration, artifact identification by eye
│   ├── 02_preprocessing.ipynb       # the 6 step pipeline, built and validated interactively
│   ├── 03_feature_extraction.ipynb  # PSD, band power, the 3 subject comparison
│   └── 04_classification.ipynb      # Random Forest, cross validation, error analysis
└── tests/
    └── test_preprocessing.py
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"

# Download the dataset (88 subjects, 19 channel resting EEG)
openneuro-py download --dataset=ds004504 --target-dir=data/raw

# Run the full pipeline across all downloaded subjects
python scripts/build_dataset.py

# Then open notebooks/04_classification.ipynb to reproduce the classification results
```

## Design decisions 

- **Relative, not absolute, band power.** Absolute EEG power is confounded by non neural factors (skull thickness, electrode contact, cap fit) that vary subject to subject. Switching to relative power (each band as a fraction of that channel's own total power) revealed a much cleaner three way separation than absolute power showed. See the full write up for the investigation that led to this choice.
- **Subject level, not epoch level, train/test splitting.** Each subject contributes one row to the final feature table (band power averaged across that subject's own clean epochs), which prevents a subject's data from leaking across the train and test.
- **A manual fallback for ICA artifact detection.** MNE's automatic find_bads_eog() missed a component for one subject because a second, moderately correlated component compressed the relevant z score just under the automatic threshold. The pipeline includes a fallback for this.
- **Per subject error isolation in batch processing.** scripts/build_dataset.py wraps each subject's processing in its own try/except with incremental saving, so one problematic subject can't crash a time intensive batch run.

## Limitations

- **Sample size.** 88 subjects against 95 features is a small sample regime for a 3 class problem. Several attempts to improve on the baseline model (constraining tree complexity, narrowing to literature motivated feature subsets) did not outperform it.
- **No dedicated muscle artifact removal.** The ICA step targets eye related artifacts only. Some residual high frequency (beta/gamma) power in frontal/temporal channels is likely muscle related (EMG) rather than neural, and was not specifically corrected for.
- **No true EOG channel.** This dataset has no dedicated eye movement sensor. Fp1 (the frontopolar channel closest to the eyes) was used as a proxy.
- **FTD classification is weak and inconsistent** in the 3 class model, reflecting a less consistent EEG signature for FTD relative to Alzheimer's.

## Acknowledgments

Dataset: Miltiadous et al., *A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG*, OpenNeuro ds004504.
