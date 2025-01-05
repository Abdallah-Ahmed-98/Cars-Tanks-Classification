from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    dataset_name: str



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int




@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list



@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    metric_file_name: Path
    model_path: Path
    dataset_dir: Path
    confusion_matrix_file_path: Path
    confusion_matrix_data_file_path: Path
    roc_curve_file_path: Path
    roc_data_file_path: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
    params_registered_model_name: str