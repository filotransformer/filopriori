"""
Text Processing Module
Handles text cleaning and formatting for transformer input
"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Processes text fields for transformer input"""

    def __init__(self, config: Dict = None):
        self.config = config
        self.text_config = config['text'] if config else {}

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing excessive whitespace and special characters

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def combine_text_fields(
        self,
        summary: str,
        steps: str,
        commit: str,
        metadata: str = ""
    ) -> str:
        """
        Combine multiple text fields into a single input for the transformer

        Args:
            summary: Test execution summary
            steps: Test case steps
            commit: Commit messages
            metadata: Optional metadata (CR type, component, etc.)

        Returns:
            Combined text with special tokens
        """
        # Clean all fields
        summary = self.clean_text(summary)
        steps = self.clean_text(steps)
        commit = self.clean_text(commit)
        metadata = self.clean_text(metadata) if metadata else ""

        # Combine with special separators
        # Format: [CLS] summary [SEP] steps [SEP] commits [SEP] metadata [SEP]
        parts = [summary, steps]

        if commit:
            parts.append(commit)

        if metadata:
            parts.append(metadata)

        combined = " [SEP] ".join(parts)

        return combined

    def prepare_batch_texts(
        self,
        summaries: List[str],
        steps: List[str],
        commits: List[str],
        metadata: List[str] = None
    ) -> List[str]:
        """
        Prepare a batch of text inputs

        Args:
            summaries: List of test summaries
            steps: List of test steps
            commits: List of commits
            metadata: Optional list of metadata

        Returns:
            List of combined text strings
        """
        if metadata is None:
            metadata = [""] * len(summaries)

        combined_texts = []
        for i in range(len(summaries)):
            combined = self.combine_text_fields(
                summaries[i],
                steps[i],
                commits[i],
                metadata[i]
            )
            combined_texts.append(combined)

        return combined_texts

    def truncate_text(self, text: str, max_length: int) -> str:
        """
        Truncate text to maximum character length

        Args:
            text: Input text
            max_length: Maximum character length

        Returns:
            Truncated text
        """
        if len(text) > max_length:
            return text[:max_length]
        return text

    def prepare_multi_field_texts(
        self,
        summaries: List[str],
        steps: List[str],
        commits: List[str],
        cr_types: List[str] = None,
        cr_components: List[str] = None
    ) -> Dict[str, List[str]]:
        """
        Prepare texts separated by field (for multi-field embeddings)

        Args:
            summaries: List of test summaries
            steps: List of test steps
            commits: List of commits
            cr_types: List of CR types
            cr_components: List of CR components

        Returns:
            Dict mapping field_name → List[cleaned_texts]
        """
        field_texts = {}

        # Summary field
        field_texts['summary'] = [self.clean_text(s) for s in summaries]

        # Steps field
        field_texts['steps'] = [self.clean_text(s) for s in steps]

        # Commits field
        field_texts['commits'] = [self.clean_text(c) for c in commits]

        # CR field (combine type + component)
        if cr_types is not None and cr_components is not None:
            cr_texts = []
            for cr_type, cr_comp in zip(cr_types, cr_components):
                # Combine CR info
                cr_type_clean = self.clean_text(cr_type) if cr_type else ""
                cr_comp_clean = self.clean_text(cr_comp) if cr_comp else ""

                # Format: "Type: {type} Component: {component}"
                parts = []
                if cr_type_clean:
                    parts.append(f"Type: {cr_type_clean}")
                if cr_comp_clean:
                    parts.append(f"Component: {cr_comp_clean}")

                cr_text = " ".join(parts) if parts else ""
                cr_texts.append(cr_text)

            field_texts['CR'] = cr_texts
        else:
            # Empty CR field if not provided
            field_texts['CR'] = [""] * len(summaries)

        return field_texts
