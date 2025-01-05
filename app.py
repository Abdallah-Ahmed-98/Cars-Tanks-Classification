import gradio as gr
import os
from cnnClassifier.pipeline.prediction import PredictionPipeline

class ClientApp:
    def __init__(self):
        self.classifier = PredictionPipeline()

# Prediction function for Gradio
def predict(input_image):
    
    # Get prediction results
    pred_labels_and_probs, pred_time = clApp.classifier.predict(input_image)
    
    return pred_labels_and_probs, pred_time


# Title and description for the Gradio interface
title = "Cars Tanks Classifier"
description = "A deep learning-based binary classifier that categorizes images into (two) classes 'Cars' or 'Tanks'."
article = "Created by Abdallah Ahmed"

# Create example list
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
            gr.Number(label="Prediction time (s)")],
    examples=example_list,
    title=title,
    description=description,
    article=article
)


if __name__ == "__main__":
    # Initialize the client app
    clApp = ClientApp()
    
# Launch Gradio app
demo.launch()
