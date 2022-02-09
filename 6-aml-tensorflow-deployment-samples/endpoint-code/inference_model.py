import numpy as np
import tensorflow as tf

from PIL import Image
class InferenceModel():
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def _preprocess_image(self, image_bytes):
        image = Image.open(image_bytes)
        
        # Reshape the image and convert it to monochrome
        image = image.resize((28,28)).convert('L')

        # Normalize data (we need to invert image, as training dataset images were inverted)
        image_np = (255 - np.array(image.getdata())) / 255.0

        # Return image data reshaped to a shape expected by model
        return image_np.reshape(-1,28,28,1)

    def predict(self, image_bytes):
        image_data = self._preprocess_image(image_bytes)
        prediction = self.model.predict(image_data)

        return np.argmax(prediction, axis=1)
