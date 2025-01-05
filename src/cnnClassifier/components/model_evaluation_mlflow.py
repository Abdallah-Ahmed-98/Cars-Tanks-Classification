import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bicubic"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.dataset_dir,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        # Load the model
        self.model = self.load_model(self.config.model_path)

        # Prepare the validation generator
        self._valid_generator()

        # Evaluate on the validation set
        self.score = self.model.evaluate(self.valid_generator, verbose=1)

        # Generate predictions
        y_pred = self.model.predict(self.valid_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = self.valid_generator.classes

        # Calculate precision and recall
        self.report = classification_report(y_true, y_pred_classes, output_dict=True)
        precision = self.report["macro avg"]["precision"]
        recall = self.report["macro avg"]["recall"]

        # Save scores
        self.save_score()

        # Save and log confusion matrix
        self.save_confusion_matrix(y_true, y_pred_classes)

        # Save and log ROC curve
        self.save_roc_curve(y_true, y_pred)

        return precision, recall



    def save_score(self):
        scores = {
            "loss": self.score[0],
            "accuracy": self.score[1],
            "precision": self.report["macro avg"]["precision"],
            "recall": self.report["macro avg"]["recall"]
        }
        save_json(path=Path(self.config.metric_file_name), data=scores)

    def save_confusion_matrix(self, y_true, y_pred_classes):
        """
        Save the normalized confusion matrix as a JSON file and as a heatmap image.
    
        Args:
            y_true (np.ndarray): True labels.
            y_pred_classes (np.ndarray): Predicted class labels.
        """
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)

        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)  # Handle cases where rows sum to 0

        # Save normalized confusion matrix as JSON
        cm_data_file = Path(self.config.confusion_matrix_data_file_path)
        cm_data = {
            "confusion_matrix": cm.tolist(),
            "normalized_confusion_matrix": cm_normalized.tolist(),
        }
        cm_data_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        save_json(cm_data_file, cm_data)

        # Save the normalized confusion matrix as a heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Normalized Confusion Matrix")
        cm_image_file = Path(self.config.confusion_matrix_file_path)
        plt.savefig(cm_image_file, bbox_inches='tight', dpi=300)
        plt.close()

        return cm_data_file, cm_image_file



    def save_roc_curve(self, y_true, y_pred):
        # Binarize the labels for ROC curve calculation
        y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_true_binarized.shape[1]

        # Calculate ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Save ROC data for all classes
        roc_data_file = Path(self.config.roc_data_file_path)
        roc_data = {
            f"class_{i}": {"fpr": fpr[i].tolist(), "tpr": tpr[i].tolist(), "auc": roc_auc[i]}
            for i in range(n_classes)
        }
        save_json(roc_data_file, roc_data)

        # Plot ROC curve for each class
        plt.figure()
        for i in range(n_classes):
            plt.plot(
                fpr[i], tpr[i], lw=2, label=f"Class {i} (AUC = {roc_auc[i]:.2f})"
            )
        plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        roc_image_file=Path(self.config.roc_curve_file_path)
        plt.savefig(roc_image_file)
        plt.close()

        return roc_data_file, roc_image_file


    def log_into_mlflow(self, precision, recall):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log parameters and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {
                    "loss": self.score[0],
                    "accuracy": self.score[1],
                    "precision": precision,
                    "recall": recall,
                }
            )

            # Log confusion matrix
            cm_file, cm_image_file = self.save_confusion_matrix(
                self.valid_generator.classes, np.argmax(self.model.predict(self.valid_generator), axis=1)
            )
            mlflow.log_artifact(str(cm_file))
            mlflow.log_artifact(cm_image_file)

            # Log ROC curve
            roc_file, roc_image_file = self.save_roc_curve(
                self.valid_generator.classes, self.model.predict(self.valid_generator)
            )
            mlflow.log_artifact(str(roc_file))
            mlflow.log_artifact(roc_image_file)

            # Register the model
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name=self.config.params_registered_model_name)
            else:
                mlflow.keras.log_model(self.model, "model")
