# Industrial QTA Dataset

## Description

Industrial dataset from Qodo Test Automation (QTA) containing real test execution data from a mobile device CI/CD pipeline.

## Statistics

| Statistic | Train | Test | Total |
|-----------|-------|------|-------|
| Total Executions | ~41,680 | ~10,420 | 52,102 |
| Unique Builds | ~1,070 | ~269 | 1,339 |
| Builds with Failures | ~222 | ~55 | 277 |
| Pass:Fail Ratio | 37:1 | 37:1 | 37:1 |

## Files

| File | Description | Size |
|------|-------------|------|
| `train.csv` | Training data (80% of builds) | ~1.7 GB |
| `test.csv` | Test data (20% of builds) | ~580 MB |

## Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `Build_ID` | string | Unique build identifier (e.g., "QPW30.18") |
| `Build_ID_entry` | string | Build entry identifier |
| `TP_Key` | string | Test plan key |
| `TC_Key` | string | Test case key (unique identifier) |
| `TE_Summary` | string | Test execution summary/description |
| `TE_Key` | string | Test execution key |
| `Build_Test_Start_Date` | datetime | When build started |
| `TE_Date` | date | Test execution date |
| `TE_Created_Date` | datetime | When test execution was created |
| `TE_Test_Result` | string | "Pass" or "Fail" |
| `TC_Steps` | string | Detailed test case steps |
| `TE_Updated_Date` | datetime | Last update timestamp |
| `commit` | string | Associated commit message |
| `CR` | string | Change request number |
| `CR_Resolution` | string | CR resolution status |
| `CR_Resolved_Date` | date | When CR was resolved |
| `CR_Component_Name` | string | Component affected |
| `CR_Type` | string | Type of change request |

## Usage

```bash
# Train with this dataset
python main.py --config configs/experiment_industry.yaml
```

## Notes

- Data is from a real industrial CI/CD environment
- Contains semantic information (test descriptions, commit messages)
- Highly imbalanced (37:1 Pass:Fail ratio)
- Used as primary evaluation dataset in IEEE TSE submission

## License

Proprietary - Qodo Test Automation
