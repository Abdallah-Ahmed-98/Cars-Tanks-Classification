import numpy as np
from tensorflow.keras.models import load_model
#import mlflow.tensorflow
from tensorflow.keras.preprocessing import image
import os
from timeit import default_timer as timer
from typing import Tuple, Dict


# Setup class names
class_names = ['Car','Tank']

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self, input_image) -> Tuple[Dict, float]:
        ## Load model
        model = load_model(os.path.join("artifacts", "training", "model.keras"))
        #logged_model = 'registered model url'
        #model = mlflow.tensorflow.load_model(logged_model)

        # Start a timer
        start_time = timer()

        # Preprocess the input image
        test_image = image.img_to_array(input_image.resize((224, 224)))  # Resize and convert to array
        test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

        # Predict probabilities for all classes
        probabilities = model.predict(test_image)[0]  # Get the probabilities for the first (and only) image

        # Map class names to their probabilities
        pred_labels_and_probs = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

        # Calculate prediction time
        end_time = timer()
        pred_time = round(end_time - start_time, 4)

        # Return prediction dictionary and prediction time
        return pred_labels_and_probs, pred_time
