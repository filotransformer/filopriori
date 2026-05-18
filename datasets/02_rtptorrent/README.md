# RTPTorrent Dataset

## Description

Open-source dataset from MSR 2020 containing test execution histories from 20 Java projects on GitHub with over 100,000 build logs from Travis CI.

## Citation

```bibtex
@inproceedings{mattis2020rtptorrent,
  title={RTPTorrent: An Open-source Dataset for Evaluating Regression Test Prioritization},
  author={Mattis, Toni and Rausch, Thomas and Rinard, Martin},
  booktitle={Proceedings of the 17th International Conference on Mining Software Repositories},
  pages={558--562},
  year={2020},
  organization={ACM}
}
```

## Source

- **Paper:** MSR 2020
- **DOI:** https://doi.org/10.1145/3379597.3387458
- **Download:** https://zenodo.org/records/3712290
- **License:** CC BY 4.0

## Projects Included

The dataset includes 20 open-source Java projects:

| Project | Description |
|---------|-------------|
| apache/commons-math | Math library |
| apache/commons-lang | Language utilities |
| apache/commons-io | IO utilities |
| google/guava | Google core libraries |
| junit-team/junit4 | Testing framework |
| ... | And 15 more projects |

## Download Instructions

### Option 1: Automatic Download

```bash
cd /home/acauan/ufam/iats/sprint_07/filo_priori_v9
python scripts/preprocessing/download_rtptorrent.py
```

### Option 2: Manual Download

```bash
# Download from Zenodo (4.1 GB)
cd datasets/02_rtptorrent/raw/
wget https://zenodo.org/records/3712290/files/rtp-torrent-v1.zip

# Extract
unzip rtp-torrent-v1.zip
```

## Preprocessing

After downloading, run the preprocessing script:

```bash
python scripts/preprocessing/preprocess_rtptorrent.py
```

This will:
1. Parse the raw RTPTorrent data
2. Extract test execution information
3. Convert to Filo-Priori format
4. Split into train/test sets
5. Save to `processed/` folder

## Data Format After Preprocessing

| Field | Type | Description |
|-------|------|-------------|
| `Build_ID` | string | `<project>_<commit_sha[:8]>` |
| `TC_Key` | string | Test class + method name |
| `TE_Summary` | string | Test method name (cleaned) |
| `TC_Steps` | string | Empty (not available) |
| `TE_Test_Result` | string | "Pass" or "Fail" |
| `commit` | string | Commit SHA |
| `Build_Test_Start_Date` | datetime | Build timestamp |

## Statistics (After Preprocessing)

Statistics will be generated after preprocessing and saved here.

## Usage

```bash
# Train with RTPTorrent
python main.py --config configs/experiment_rtptorrent.yaml
```

## Directory Structure

```
02_rtptorrent/
├── raw/                      # Original downloaded data
│   └── rtp-torrent-v1/       # Extracted archive
├── processed/                # Converted to Filo-Priori format
│   ├── train.csv
│   ├── test.csv
│   └── statistics.json
└── README.md                 # This file
```

## Notes

- Contains only test execution results (no detailed test descriptions)
- Limited semantic information compared to Industry dataset
- Good for evaluating structural features (failure history, co-failure)
- Multiple projects allow for cross-project evaluation
