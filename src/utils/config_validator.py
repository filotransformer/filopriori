"""
Configuration Validator Module
Validates experiment configuration files to catch errors before execution starts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    pass


class ConfigValidator:
    """Validates experiment configuration against required schema"""

    # Required configuration structure
    REQUIRED_SCHEMA = {
        'experiment': {
            'required': True,
            'fields': {
                'name': {'type': str, 'required': True},
                'version': {'type': str, 'required': False},
                'seed': {'type': int, 'required': False, 'default': 42}
            }
        },
        'data': {
            'required': True,
            'fields': {
                'train_path': {'type': str, 'required': True},
                'test_path': {'type': str, 'required': True},
                'train_split': {'type': float, 'required': True, 'min': 0.0, 'max': 1.0},
                'val_split': {'type': float, 'required': True, 'min': 0.0, 'max': 1.0},
                'test_split': {'type': float, 'required': True, 'min': 0.0, 'max': 1.0},
                'binary_classification': {'type': bool, 'required': False, 'default': True}
            }
        },
        'semantic': {
            'required': True,
            'fields': {
                'model_name': {'type': str, 'required': True},
                'embedding_dim': {'type': int, 'required': True, 'min': 128},
                'combined_embedding_dim': {'type': int, 'required': True, 'min': 256},
                'max_length': {'type': int, 'required': True, 'min': 32, 'max': 32768},
                'batch_size': {'type': int, 'required': True, 'min': 1, 'max': 512},
                'cache_path': {'type': str, 'required': False}
            }
        },
        'structural': {
            'required': True,
            'fields': {
                'extractor': {'type': dict, 'required': True},
                'input_dim': {'type': int, 'required': True, 'min': 1}
            }
        },
        'model': {
            'required': True,
            'fields': {
                'type': {'type': str, 'required': True},
                'semantic': {'type': dict, 'required': True},
                'structural': {'type': dict, 'required': True},
                'classifier': {'type': dict, 'required': True},
                'num_classes': {'type': int, 'required': True, 'min': 2}
            }
        },
        'training': {
            'required': True,
            'fields': {
                'num_epochs': {'type': int, 'required': True, 'min': 1, 'max': 1000},
                'batch_size': {'type': int, 'required': True, 'min': 1, 'max': 512},
                'learning_rate': {'type': float, 'required': True, 'min': 1e-7, 'max': 1.0},
                'weight_decay': {'type': float, 'required': False, 'min': 0.0, 'max': 1.0}
            }
        },
        'hardware': {
            'required': True,
            'fields': {
                'device': {'type': str, 'required': True, 'choices': ['cuda', 'cpu']},
                'num_workers': {'type': int, 'required': False, 'min': 0, 'max': 32, 'default': 4},
                'pin_memory': {'type': bool, 'required': False, 'default': True}
            }
        }
    }

    def __init__(self, strict: bool = True):
        """
        Initialize validator

        Args:
            strict: If True, raises exception on validation errors. If False, logs warnings.
        """
        self.strict = strict
        self.errors = []
        self.warnings = []

    def validate(self, config: Dict) -> bool:
        """
        Validate configuration

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if validation passes, False otherwise

        Raises:
            ConfigValidationError: If strict=True and validation fails
        """
        self.errors = []
        self.warnings = []

        logger.info("Validating configuration...")

        # Check required sections
        for section_name, section_schema in self.REQUIRED_SCHEMA.items():
            if section_schema.get('required', True):
                if section_name not in config:
                    self.errors.append(f"Missing required section: '{section_name}'")
                    continue

                # Validate fields within section
                self._validate_section(section_name, config[section_name], section_schema.get('fields', {}))

        # Additional custom validations
        self._validate_splits(config)
        self._validate_dimensions(config)
        self._validate_paths(config)

        # Report results
        if self.errors:
            error_msg = f"Configuration validation failed with {len(self.errors)} error(s):\n"
            error_msg += "\n".join(f"  - {err}" for err in self.errors)
            logger.error(error_msg)

            if self.strict:
                raise ConfigValidationError(error_msg)
            return False

        if self.warnings:
            logger.warning(f"Configuration has {len(self.warnings)} warning(s):")
            for warn in self.warnings:
                logger.warning(f"  - {warn}")

        logger.info("Configuration validation passed!")
        return True

    def _validate_section(self, section_name: str, section_data: Dict, fields_schema: Dict):
        """Validate a configuration section"""
        for field_name, field_schema in fields_schema.items():
            field_path = f"{section_name}.{field_name}"

            # Check required fields
            if field_schema.get('required', False):
                if field_name not in section_data:
                    self.errors.append(f"Missing required field: '{field_path}'")
                    continue

            # Skip if field not present and not required
            if field_name not in section_data:
                continue

            value = section_data[field_name]

            # Type validation
            expected_type = field_schema.get('type')
            if expected_type and not isinstance(value, expected_type):
                # Special case: Allow int where float is expected (e.g., weight_decay: 0)
                if expected_type == float and isinstance(value, int):
                    # Convert int to float
                    section_data[field_name] = float(value)
                else:
                    self.errors.append(
                        f"Invalid type for '{field_path}': expected {expected_type.__name__}, "
                        f"got {type(value).__name__} (value: {value})"
                    )
                    continue

            # Range validation for numeric types
            if isinstance(value, (int, float)):
                if 'min' in field_schema and value < field_schema['min']:
                    self.errors.append(
                        f"Value for '{field_path}' ({value}) is below minimum ({field_schema['min']})"
                    )
                if 'max' in field_schema and value > field_schema['max']:
                    self.errors.append(
                        f"Value for '{field_path}' ({value}) exceeds maximum ({field_schema['max']})"
                    )

            # Choice validation for strings
            if 'choices' in field_schema:
                if value not in field_schema['choices']:
                    self.errors.append(
                        f"Invalid value for '{field_path}': '{value}' not in {field_schema['choices']}"
                    )

    def _validate_splits(self, config: Dict):
        """Validate that train/val/test splits sum to 1.0"""
        if 'data' not in config:
            return

        data_config = config['data']
        splits = ['train_split', 'val_split', 'test_split']

        if all(s in data_config for s in splits):
            total = sum(data_config[s] for s in splits)
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                self.errors.append(
                    f"Data splits must sum to 1.0, got {total} "
                    f"(train={data_config['train_split']}, "
                    f"val={data_config['val_split']}, "
                    f"test={data_config['test_split']})"
                )

    def _validate_dimensions(self, config: Dict):
        """Validate that embedding dimensions are consistent"""
        if 'semantic' not in config or 'model' not in config:
            return

        semantic_config = config['semantic']
        model_config = config['model']

        # Check combined embedding dimension
        if 'combined_embedding_dim' in semantic_config and 'embedding_dim' in semantic_config:
            expected_combined = 2 * semantic_config['embedding_dim']
            actual_combined = semantic_config['combined_embedding_dim']

            if expected_combined != actual_combined:
                self.warnings.append(
                    f"Combined embedding dim mismatch: expected {expected_combined} "
                    f"(2 * {semantic_config['embedding_dim']}), got {actual_combined}"
                )

        # Check model semantic input dim matches config
        if 'semantic' in model_config:
            model_input_dim = model_config['semantic'].get('input_dim')
            config_combined_dim = semantic_config.get('combined_embedding_dim')

            if model_input_dim and config_combined_dim:
                if model_input_dim != config_combined_dim:
                    self.errors.append(
                        f"Model semantic input_dim ({model_input_dim}) doesn't match "
                        f"semantic combined_embedding_dim ({config_combined_dim})"
                    )

    def _validate_paths(self, config: Dict):
        """Validate that required file paths exist"""
        if 'data' not in config:
            return

        data_config = config['data']

        # Check dataset paths
        for path_key in ['train_path', 'test_path']:
            if path_key in data_config:
                path = Path(data_config[path_key])
                if not path.exists():
                    self.errors.append(f"Dataset file not found: {data_config[path_key]}")
                elif not path.is_file():
                    self.errors.append(f"Dataset path is not a file: {data_config[path_key]}")
                else:
                    # Check file size
                    size_mb = path.stat().st_size / (1024 * 1024)
                    if size_mb < 1:
                        self.warnings.append(
                            f"Dataset file seems too small ({size_mb:.2f} MB): {data_config[path_key]}"
                        )


def validate_config(config: Dict, strict: bool = True) -> bool:
    """
    Validate configuration dictionary

    Args:
        config: Configuration to validate
        strict: If True, raises exception on errors

    Returns:
        True if valid, False otherwise
    """
    validator = ConfigValidator(strict=strict)
    return validator.validate(config)
