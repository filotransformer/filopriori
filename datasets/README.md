# Datasets for Filo-Priori

This directory contains datasets used for evaluating the Filo-Priori test case prioritization approach.

## Directory Structure

```
datasets/
├── 01_industry/              # Industrial QTA Dataset (original)
│   ├── train.csv
│   ├── test.csv
│   └── README.md
│
├── 02_rtptorrent/            # RTPTorrent Dataset (open-source)
│   ├── raw/                  # Original downloaded data
│   ├── processed/            # Converted to Filo-Priori format
│   │   ├── train.csv
│   │   └── test.csv
│   └── README.md
│
└── README.md                 # This file
```

---

## Dataset 1: Industrial QTA Dataset

**Location:** `01_industry/`

### Description
Industrial dataset from Qodo Test Automation containing test execution data from a real CI/CD pipeline.

### Statistics

| Statistic | Value |
|-----------|-------|
| Total Executions | 52,102 |
| Unique Builds | 1,339 |
| Builds with Failures | 277 (20.7%) |
| Unique Test Cases | 2,347 |
| Pass:Fail Ratio | 37:1 |

### Fields

| Field | Description |
|-------|-------------|
| `Build_ID` | Unique build identifier |
| `TC_Key` | Test case identifier |
| `TE_Summary` | Test execution summary/description |
| `TC_Steps` | Detailed test case steps |
| `TE_Test_Result` | Pass/Fail verdict |
| `commit` | Associated commit message |
| `CR` | Change request number |
| `Build_Test_Start_Date` | Build timestamp |

---

## Dataset 2: RTPTorrent Dataset

**Location:** `02_rtptorrent/`

### Description
Open-source dataset from MSR 2020 containing test execution histories from 20 Java projects on GitHub with over 100,000 build logs from Travis CI.

### Source
- **Paper:** Mattis et al., "RTPTorrent: An Open-source Dataset for Evaluating Regression Test Prioritization", MSR 2020
- **DOI:** https://doi.org/10.1145/3379597.3387458
- **Download:** https://zenodo.org/records/3712290

### Projects Included

The dataset includes 20 open-source Java projects:
- Apache Commons projects (math, lang, etc.)
- Google Guava
- JUnit
- And others

### Download Instructions

```bash
# Download RTPTorrent (4.1 GB)
cd datasets/02_rtptorrent/raw/
wget https://zenodo.org/records/3712290/files/rtp-torrent-v1.zip

# Extract
unzip rtp-torrent-v1.zip
```

### Preprocessing

After downloading, run the preprocessing script to convert to Filo-Priori format:

```bash
python scripts/preprocessing/preprocess_rtptorrent.py
```

---

## Usage

### Running with Industry Dataset (default)

```bash
python main.py --config configs/experiment_industry.yaml
```

### Running with RTPTorrent Dataset

```bash
python main.py --config configs/experiment_rtptorrent.yaml
```

### Running Cross-Dataset Evaluation

```bash
# Train on Industry, Test on RTPTorrent
python main.py --config configs/experiment_cross_dataset.yaml
```

---

## Data Format Requirements

All datasets must be converted to the following CSV format for Filo-Priori:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `Build_ID` | string | Yes | Unique build identifier |
| `TC_Key` | string | Yes | Test case identifier |
| `TE_Summary` | string | Yes | Test description (for semantic features) |
| `TC_Steps` | string | No | Detailed test steps |
| `TE_Test_Result` | string | Yes | "Pass" or "Fail" |
| `commit` | string | No | Commit message |
| `Build_Test_Start_Date` | datetime | No | Build timestamp |

---

## Adding New Datasets

1. Create a new folder: `datasets/XX_<dataset_name>/`
2. Download raw data to `raw/` subfolder
3. Create preprocessing script in `scripts/preprocessing/`
4. Convert to standard format in `processed/` subfolder
5. Create dataset-specific config in `configs/`
6. Update this README

---

## License

- **Industry Dataset:** Proprietary (Qodo)
- **RTPTorrent:** CC BY 4.0
