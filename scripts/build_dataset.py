"""
Batch process multiple subjects: preprocessing + feature extraction, with
per subject error handling 

Usage:
    python scripts/build_dataset.py

Reads data/raw/participants.tsv for group labels, processes every subject
folder found under data/raw/ that has a corresponding EEG file, and writes
the resulting feature table to data/processed/features.csv

Progress is saved after every subject so an interruption
partway through does not lose already-completed work. running it again
picks up from an empty output file, but there are completed persubject logs make it
easy to see what succeeded before a crash.
"""

import sys
import time
from pathlib import Path

import pandas as pd

# Make src/ importable when running this directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eeg_dementia.preprocessing import preprocess_subject
from eeg_dementia.features import build_feature_row

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "features.csv"


def get_available_subjects():
    """
    Cross reference which subjects have both a raw EEG file downloaded AND a
    group label in participants.tsv. Returns {subject_id: group_label}.
    """
    participants = pd.read_csv(RAW_DIR / "participants.tsv", sep="\t")
    labels = dict(zip(participants["participant_id"], participants["Group"]))

    available = {}
    for sub_dir in sorted(RAW_DIR.glob("sub-*")):
        sub_id = sub_dir.name
        eeg_file = sub_dir / "eeg" / f"{sub_id}_task-eyesclosed_eeg.set"
        if eeg_file.exists() and sub_id in labels:
            available[sub_id] = labels[sub_id]
    return available


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    subjects = get_available_subjects()
    print(f"Found {len(subjects)} downloaded subjects with labels.")

    rows = []
    failures = []

    for i, (sub_id, group) in enumerate(subjects.items(), start=1):
        eeg_path = RAW_DIR / sub_id / "eeg" / f"{sub_id}_task-eyesclosed_eeg.set"
        print(f"[{i}/{len(subjects)}] Processing {sub_id} (group {group})...", end=" ")

        start = time.time()
        try:
            epochs, info = preprocess_subject(str(eeg_path))
            row = build_feature_row(sub_id, group, epochs)
            rows.append(row)
            elapsed = time.time() - start
            print(
                f"OK ({elapsed:.0f}s) — {info['n_epochs_kept']}/{info['n_epochs_total']} "
                f"epochs kept, ICA excluded {info['excluded_ica_components']}"
            )
        except Exception as e:
            elapsed = time.time() - start
            print(f"FAILED ({elapsed:.0f}s) — {type(e).__name__}: {e}")
            failures.append({"subject_id": sub_id, "group": group, "error": str(e)})

        # Save progress after every subject, not just at the end 
        # interruption partway through still leaves a usable partial dataset.
        if rows:
            pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone. {len(rows)} succeeded, {len(failures)} failed.")
    if failures:
        print("Failed subjects:")
        for f in failures:
            print(f"  {f['subject_id']} ({f['group']}): {f['error']}")
        failures_path = PROCESSED_DIR / "failed_subjects.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Failure details saved to {failures_path}")

    print(f"Feature table saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()