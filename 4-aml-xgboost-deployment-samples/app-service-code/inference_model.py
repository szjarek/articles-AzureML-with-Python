import numpy
import xgboost as xgb

from PIL import Image

class InferenceModel():
    def __init__(self, model_path):
        self.model = xgb.XGBRFClassifier()
        self.model.load_model(model_path)

    def _preprocess_image(self, image_bytes):
        image = Image.open(image_bytes)
        
        # Reshape the image and convert it to monochrome
        image = image.resize((28,28)).convert('L')
        
        # Normalize data
        image_np = (255 - numpy.array(image.getdata())) / 255.0

        # Return image data reshaped to a vector
        return image_np.reshape(1, -1)

    def predict(self, image_bytes):
        image_data = self._preprocess_image(image_bytes)
        prediction = self.model.predict(image_data)
        return prediction
