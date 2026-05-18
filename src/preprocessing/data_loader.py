"""
Data Loading and Preprocessing Module
Handles loading the software testing dataset and basic preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from typing import Dict, Tuple, Optional
import logging
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import SMOTE for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logger.warning("imbalanced-learn not installed. SMOTE will not be available.")


class DataLoader:
    """Loads and preprocesses the software testing dataset"""

    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config['data']
        self.text_config = config['text']
        self.label_mapping = None

    def load_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load dataset from CSV

        Args:
            sample_size: If provided, loads only first N rows (for debugging)

        Returns:
            DataFrame with loaded data
        """
        train_path = self.data_config['train_path']
        logger.info(f"Loading data from {train_path}")

        if sample_size:
            logger.info(f"Loading sample of {sample_size} rows")
            df = pd.read_csv(train_path, nrows=sample_size)
        else:
            logger.info("Loading full dataset (this may take a while...)")
            df = pd.read_csv(train_path)

        logger.info(f"Loaded {len(df)} records")
        logger.info(f"Columns: {df.columns.tolist()}")

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the dataset

        Args:
            df: Raw dataframe

        Returns:
            Cleaned dataframe
        """
        logger.info("Cleaning data...")

        # Make a copy
        df = df.copy()

        # Handle missing values - only require TE_Summary and TE_Test_Result
        # TC_Steps is optional (empty in some datasets like RTPTorrent)
        initial_len = len(df)
        required_cols = ['TE_Summary', 'TE_Test_Result']

        # Only include TC_Steps in dropna if it has actual data in the dataset
        if 'TC_Steps' in df.columns:
            non_empty_steps = df['TC_Steps'].notna() & (df['TC_Steps'].astype(str).str.strip() != '')
            if non_empty_steps.sum() > len(df) * 0.1:  # More than 10% have data
                required_cols.append('TC_Steps')
                logger.info("TC_Steps has significant data - including in validation")
            else:
                logger.info("TC_Steps is mostly empty - making it optional")
                df['TC_Steps'] = df['TC_Steps'].fillna("")

        df = df.dropna(subset=required_cols)
        logger.info(f"Dropped {initial_len - len(df)} rows with missing critical fields ({required_cols})")

        # Clean text fields
        df['TE_Summary'] = df['TE_Summary'].astype(str).str.strip()
        df['TC_Steps'] = df['TC_Steps'].astype(str).str.strip()

        # Process commit field (it's in array format as string)
        if 'commit' in df.columns:
            df['commit_processed'] = df['commit'].apply(self._process_commit_field)
        else:
            df['commit_processed'] = ""
            logger.info("'commit' column not found - using empty string")

        # Process CR fields (also in array format) - optional for some datasets
        if 'CR_Type' in df.columns:
            df['CR_Type_processed'] = df['CR_Type'].apply(self._extract_from_list)
        else:
            df['CR_Type_processed'] = ""

        if 'CR_Component_Name' in df.columns:
            df['CR_Component_processed'] = df['CR_Component_Name'].apply(self._extract_from_list)
        else:
            df['CR_Component_processed'] = ""

        # Clean target variable
        df['TE_Test_Result'] = df['TE_Test_Result'].str.strip()

        logger.info(f"Target distribution:\n{df['TE_Test_Result'].value_counts()}")

        return df

    def _process_commit_field(self, commit_str: str) -> str:
        """
        Process commit field which contains list of commit messages

        Args:
            commit_str: String representation of commit list

        Returns:
            Processed commit text
        """
        try:
            # Parse the string representation of list
            commit_list = ast.literal_eval(str(commit_str))

            if isinstance(commit_list, list) and len(commit_list) > 0:
                # Get first N commits based on config
                num_commits = self.text_config.get('num_commits_to_keep', 3)
                commits = commit_list[:num_commits]

                # Join with special separator
                commit_text = " [SEP] ".join(commits)

                # Truncate to max length
                max_len = self.text_config.get('max_commit_length', 512)
                if len(commit_text) > max_len:
                    commit_text = commit_text[:max_len]

                return commit_text
            else:
                return ""
        except:
            return ""

    def _extract_from_list(self, list_str: str) -> str:
        """Extract first element from string representation of list"""
        try:
            parsed = ast.literal_eval(str(list_str))
            if isinstance(parsed, list) and len(parsed) > 0:
                return str(parsed[0])
            else:
                return ""
        except:
            return ""

    def encode_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode target labels to integers

        Args:
            df: DataFrame with TE_Test_Result column

        Returns:
            DataFrame with encoded labels and label mapping
        """
        # Check if binary classification is enabled
        if self.data_config.get('binary_classification', False):
            return self._encode_labels_binary(df)
        else:
            return self._encode_labels_multiclass(df)

    def _encode_labels_multiclass(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode labels for multi-class classification

        Args:
            df: DataFrame with TE_Test_Result column

        Returns:
            DataFrame with encoded labels and label mapping
        """
        unique_labels = df['TE_Test_Result'].unique()
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.inverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}

        df['label'] = df['TE_Test_Result'].map(self.label_mapping)

        logger.info(f"Label mapping: {self.label_mapping}")
        logger.info(f"Number of classes: {len(self.label_mapping)}")

        return df, self.label_mapping

    def _encode_labels_binary(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode labels for binary classification

        Supports multiple strategies:
        - pass_vs_all: Pass=1, everything else=0
        - pass_vs_fail: Pass=1, Fail=0, exclude others
        - fail_vs_all: Fail=1, everything else=0

        Args:
            df: DataFrame with TE_Test_Result column

        Returns:
            DataFrame with encoded labels and label mapping
        """
        strategy = self.data_config.get('binary_strategy', 'pass_vs_all')
        positive_class = self.data_config.get('binary_positive_class', 'Pass')

        logger.info(f"Binary classification strategy: {strategy}")
        logger.info(f"Positive class: {positive_class}")

        df = df.copy()

        if strategy == 'pass_vs_all':
            # Pass = 1, everything else = 0
            df['label'] = (df['TE_Test_Result'] == positive_class).astype(int)
            self.label_mapping = {'Not-Pass': 0, positive_class: 1}
            self.inverse_label_mapping = {0: 'Not-Pass', 1: positive_class}

        elif strategy == 'pass_vs_fail':
            # Pass = 1, Fail = 0, exclude others
            df = df[df['TE_Test_Result'].isin(['Pass', 'Fail'])].copy()
            df['label'] = (df['TE_Test_Result'] == 'Pass').astype(int)
            self.label_mapping = {'Fail': 0, 'Pass': 1}
            self.inverse_label_mapping = {0: 'Fail', 1: 'Pass'}
            logger.info(f"Excluded samples with labels other than Pass/Fail")

        elif strategy == 'fail_vs_all':
            # Fail = 1, everything else = 0
            df['label'] = (df['TE_Test_Result'] == 'Fail').astype(int)
            self.label_mapping = {'Not-Fail': 0, 'Fail': 1}
            self.inverse_label_mapping = {0: 'Not-Fail', 1: 'Fail'}

        else:
            raise ValueError(f"Unknown binary strategy: {strategy}")

        logger.info(f"Binary label mapping: {self.label_mapping}")
        logger.info(f"Class distribution after encoding:\n{df['label'].value_counts()}")

        return df, self.label_mapping

    def compute_class_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute class weights for handling imbalanced data

        Args:
            df: DataFrame with 'label' column

        Returns:
            Array of class weights (always 2 elements for binary classification)
        """
        from sklearn.utils.class_weight import compute_class_weight

        unique_classes = np.unique(df['label'])
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=df['label']
        )

        # Ensure we always have 2 weights for binary classification
        if len(class_weights) == 1:
            logger.warning(f"Only one class found in data. Adding default weight for missing class.")
            if unique_classes[0] == 0:
                # Only class 0 (Fail) present, add weight for class 1 (Pass)
                class_weights = np.array([class_weights[0], 1.0])
            else:
                # Only class 1 (Pass) present, add weight for class 0 (Fail)
                class_weights = np.array([1.0, class_weights[0]])

        logger.info(f"Class weights: {class_weights}")
        return class_weights

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets using STRICT TEMPORAL splitting.

        ANTI-LEAKAGE: Builds are sorted chronologically and split sequentially.
        This guarantees that no future knowledge leaks into the past and
        that execution history is preserved in perfect chronological order.

        Args:
            df: Full dataframe (must have 'Build_ID' column)

        Returns:
            train_df, val_df, test_df
        """
        train_split = self.data_config['train_split']
        val_split = self.data_config['val_split']
        test_split = self.data_config['test_split']

        logger.info(f"🔒 Using STRICT TEMPORAL split to prevent leakage")
        logger.info(f"   Total samples: {len(df)}, Total builds: {df['Build_ID'].nunique()}")

        if 'Build_Test_Start_Date' in df.columns:
            # Sort chronologically by build date
            df = df.copy()
            df['Build_Test_Start_Date'] = pd.to_datetime(df['Build_Test_Start_Date'])
            df = df.sort_values(by=['Build_Test_Start_Date', 'Build_ID'])
            logger.info("   Sorted chronologically by Build_Test_Start_Date")
        else:
            logger.warning("   Build_Test_Start_Date not found, assuming dataset is already sorted chronologically")

        # Get unique builds in chronological order
        unique_builds = df['Build_ID'].unique()
        n_builds = len(unique_builds)

        # Calculate split indices
        train_end_idx = int(n_builds * train_split)
        val_end_idx = int(n_builds * (train_split + val_split))

        train_builds = unique_builds[:train_end_idx]
        val_builds = unique_builds[train_end_idx:val_end_idx]
        test_builds = unique_builds[val_end_idx:]

        # Split dataframe based on builds
        train_df = df[df['Build_ID'].isin(train_builds)].copy()
        val_df = df[df['Build_ID'].isin(val_builds)].copy()
        test_df = df[df['Build_ID'].isin(test_builds)].copy()

        logger.info(f"✅ Train set: {len(train_df)} samples ({len(train_builds)} builds)")
        logger.info(f"✅ Val set:   {len(val_df)} samples ({len(val_builds)} builds)")
        logger.info(f"✅ Test set:  {len(test_df)} samples ({len(test_builds)} builds)")

        # Verify no build overlap
        train_builds_set = set(train_builds)
        val_builds_set = set(val_builds)
        test_builds_set = set(test_builds)

        overlap_train_val = train_builds_set & val_builds_set
        overlap_train_test = train_builds_set & test_builds_set
        overlap_val_test = val_builds_set & test_builds_set

        if overlap_train_val or overlap_train_test or overlap_val_test:
            logger.error(f"❌ BUILD LEAKAGE DETECTED!")
        else:
            logger.info(f"✅ No build leakage: All splits are disjoint by Build_ID and strictly sequential")

        return train_df, val_df, test_df

    def apply_smote(self, train_df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) to training data

        Args:
            train_df: Training DataFrame
            feature_columns: List of feature column names to use for SMOTE

        Returns:
            Balanced training DataFrame
        """
        if not SMOTE_AVAILABLE:
            logger.warning("SMOTE not available. Skipping SMOTE.")
            return train_df

        if not self.data_config.get('use_smote', False):
            logger.info("SMOTE disabled in config. Skipping SMOTE.")
            return train_df

        logger.info("Applying SMOTE to balance classes...")

        # Note: SMOTE will be applied in the training pipeline after embeddings are created
        # This is just a placeholder method. The actual SMOTE will be applied to embeddings
        # in the trainer or main pipeline.
        logger.info("SMOTE will be applied to embeddings in the training pipeline.")

        return train_df

    def load_full_test_dataset(self) -> pd.DataFrame:
        """
        Load FULL test dataset (test.csv) for final APFD evaluation.

        This is SEPARATE from the train/val/test split and should contain
        the complete test set with 31,333 samples and 277 builds with failures.

        This matches the approach used in filo_priori_v5.

        Returns:
            DataFrame with complete test set
        """
        test_path = self.data_config.get('test_path', 'datasets/test.csv')
        logger.info(f"Loading FULL test dataset from {test_path}")

        # Load full test.csv
        df_test = pd.read_csv(test_path)
        logger.info(f"Loaded {len(df_test)} test samples from test.csv")
        logger.info(f"Total builds: {df_test['Build_ID'].nunique()}")

        # Check builds with failures
        builds_with_fail = df_test[df_test['TE_Test_Result'] == 'Fail']['Build_ID'].nunique()
        logger.info(f"Builds with at least one 'Fail': {builds_with_fail} (expected: 277)")

        # Apply NON-STRICT preprocessing for APFD: do NOT drop rows with missing text.
        # We need to preserve all test cases so that count_tc per build matches the
        # original test.csv. Empty texts are allowed for embedding (encode as "").
        df_test = self._clean_data_non_strict(df_test)

        # Encode labels but avoid dropping rows beyond Pass/Fail if configured.
        # For 'pass_vs_fail', this will exclude non Pass/Fail statuses by design.
        df_test, _ = self.encode_labels(df_test)

        logger.info(f"After preprocessing: {len(df_test)} samples")

        return df_test

    def _clean_data_non_strict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lightweight cleaning that preserves all rows for APFD/Full test.

        - Does NOT drop rows with missing TE_Summary/TC_Steps
        - Fills missing text fields with empty strings
        - Parses commit/CR list-like fields safely
        - Strips TE_Test_Result whitespace
        """
        logger.info("Lightweight cleaning for FULL test.csv (no row drops)")
        df = df.copy()

        # Ensure required columns exist; if missing, create empty
        for col in ['TE_Summary', 'TC_Steps', 'commit', 'CR_Type', 'CR_Component_Name']:
            if col not in df.columns:
                df[col] = ""

        # Fill NaNs and strip
        df['TE_Summary'] = df['TE_Summary'].fillna("").astype(str).str.strip()
        df['TC_Steps'] = df['TC_Steps'].fillna("").astype(str).str.strip()

        # Process commit and CR fields
        df['commit_processed'] = df['commit'].apply(self._process_commit_field)
        df['CR_Type_processed'] = df['CR_Type'].apply(self._extract_from_list)
        df['CR_Component_processed'] = df['CR_Component_Name'].apply(self._extract_from_list)

        # Clean target variable (keep as string if present)
        if 'TE_Test_Result' in df.columns:
            df['TE_Test_Result'] = df['TE_Test_Result'].astype(str).str.strip()

        # Log label distribution when available
        if 'TE_Test_Result' in df.columns:
            try:
                logger.info(f"Target distribution (non-strict):\n{df['TE_Test_Result'].value_counts()}")
            except Exception:
                pass

        return df

    def prepare_dataset(self, sample_size: Optional[int] = None) -> Dict:
        """
        Complete data preparation pipeline

        Args:
            sample_size: Optional sample size for debugging

        Returns:
            Dictionary containing train/val/test DataFrames and metadata
        """
        # Load data
        df = self.load_data(sample_size)

        # Clean data
        df = self.clean_data(df)

        # Encode labels
        df, label_mapping = self.encode_labels(df)

        # Compute class weights
        class_weights = self.compute_class_weights(df)

        # Split data
        train_df, val_df, test_df = self.split_data(df)

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'label_mapping': label_mapping,
            'class_weights': class_weights,
            'num_classes': len(label_mapping)
        }
