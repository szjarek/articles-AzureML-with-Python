import os
import json
from inference_model import InferenceModel

from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It contains the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "mnist.xgb_model" # model-path/model-file-name
    )

    model = InferenceModel(model_path)

@rawhttp
def run(request):
    if request.method != 'POST':
        return AMLResponse(f"Unsupported verb: {request.method}", 400)

    image_data = request.files['image']
    preds = model.predict(image_data)
    
    return AMLResponse(json.dumps({"preds": preds.tolist()}), 200)
    