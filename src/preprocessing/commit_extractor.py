"""
Commit Extractor Module
Extracts and preprocesses commit information for encoding
"""

import re
import json
from typing import List, Dict, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class CommitExtractor:
    """Extracts and preprocesses commit information"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CommitExtractor

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.max_commits_per_tc = self.config.get('max_commits_per_tc', 10)
        self.include_metadata = self.config.get('include_commit_metadata', True)

    def clean_commit_message(self, message: str) -> str:
        """
        Clean commit message

        Args:
            message: Raw commit message

        Returns:
            Cleaned commit message
        """
        if not message or pd.isna(message):
            return ""

        # Convert to string
        message = str(message)

        # Remove excessive whitespace
        message = re.sub(r'\s+', ' ', message)

        # Remove control characters
        message = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', message)

        # Remove common commit prefixes (merge, revert, etc.)
        message = re.sub(r'^(Merge|Revert|Fix|Add|Update|Remove|Refactor):\s*', '', message, flags=re.IGNORECASE)

        # Remove issue/ticket references (e.g., #123, JIRA-123)
        message = re.sub(r'#\d+|\b[A-Z]+-\d+\b', '', message)

        # Strip leading/trailing whitespace
        message = message.strip()

        return message

    def parse_commit_field(self, commit_field: str) -> List[Dict]:
        """
        Parse commit field which may contain JSON array or simple string

        Args:
            commit_field: Commit field from dataset

        Returns:
            List of commit dictionaries with 'message' key
        """
        if not commit_field or pd.isna(commit_field):
            return []

        commit_field = str(commit_field).strip()

        # Try to parse as JSON array
        if commit_field.startswith('[') and commit_field.endswith(']'):
            try:
                commits = json.loads(commit_field)
                if isinstance(commits, list):
                    # Ensure each commit is a dict with at least 'message' key
                    parsed_commits = []
                    for c in commits:
                        if isinstance(c, dict):
                            parsed_commits.append(c)
                        elif isinstance(c, str):
                            parsed_commits.append({'message': c})
                    return parsed_commits[:self.max_commits_per_tc]
            except json.JSONDecodeError:
                pass

        # Treat as single commit message
        return [{'message': commit_field}]

    def extract_commit_text(self, commit: Dict) -> str:
        """
        Extract text from commit dictionary

        Args:
            commit: Commit dictionary

        Returns:
            Formatted commit text
        """
        parts = []

        # Commit message (primary)
        message = commit.get('message', '')
        if message:
            clean_message = self.clean_commit_message(message)
            if clean_message:
                parts.append(clean_message)

        # Optional: Include commit metadata
        if self.include_metadata:
            # Author
            author = commit.get('author', '')
            if author and not pd.isna(author):
                parts.append(f"Author: {str(author).strip()}")

            # Files changed (if available)
            files = commit.get('files_changed', [])
            if files and isinstance(files, list) and len(files) > 0:
                # Only include filename, not full path
                file_names = [f.split('/')[-1] for f in files[:5]]  # Limit to 5 files
                parts.append(f"Files: {', '.join(file_names)}")

        return ' '.join(parts)

    def process_commits(self, commit_field: str) -> str:
        """
        Process commit field into a single text representation

        Args:
            commit_field: Raw commit field from dataset

        Returns:
            Processed commit text
        """
        # Parse commits
        commits = self.parse_commit_field(commit_field)

        if not commits:
            return ""

        # Extract text from each commit
        commit_texts = []
        for commit in commits:
            commit_text = self.extract_commit_text(commit)
            if commit_text:
                commit_texts.append(commit_text)

        # Join all commits with separator
        if not commit_texts:
            return ""

        # Format: "Commit 1: <text> | Commit 2: <text> | ..."
        formatted_commits = []
        for i, text in enumerate(commit_texts, 1):
            formatted_commits.append(f"Commit {i}: {text}")

        return " | ".join(formatted_commits)

    def extract_batch(self, commit_fields: List[str]) -> List[str]:
        """
        Process a batch of commit fields

        Args:
            commit_fields: List of raw commit fields

        Returns:
            List of processed commit texts
        """
        processed_commits = []

        for commit_field in commit_fields:
            processed = self.process_commits(commit_field)
            processed_commits.append(processed)

        return processed_commits

    def extract_from_dataframe(self, df: pd.DataFrame, commit_column: str = 'commit') -> List[str]:
        """
        Extract and process commits from a dataframe

        Args:
            df: DataFrame with commit column
            commit_column: Name of the commit column

        Returns:
            List of processed commit texts
        """
        if commit_column not in df.columns:
            logger.warning(f"Column '{commit_column}' not found in dataframe")
            return [""] * len(df)

        commit_fields = df[commit_column].fillna("").tolist()
        processed_commits = self.extract_batch(commit_fields)

        logger.info(f"Extracted commits for {len(processed_commits)} samples")

        # Statistics
        non_empty = sum(1 for c in processed_commits if c)
        logger.info(f"  Non-empty commits: {non_empty}/{len(processed_commits)} ({100*non_empty/len(processed_commits):.1f}%)")

        return processed_commits


def extract_commit_texts(
    df: pd.DataFrame,
    commit_column: str = 'commit',
    config: Optional[Dict] = None
) -> List[str]:
    """
    Convenience function to extract commit texts from a dataframe

    Args:
        df: DataFrame with commit column
        commit_column: Name of the commit column
        config: Optional configuration

    Returns:
        List of processed commit texts
    """
    extractor = CommitExtractor(config)
    return extractor.extract_from_dataframe(df, commit_column)
