import json
from flask import Flask, request
from inference_model import InferenceModel

from azureml.core.authentication import MsiAuthentication
from azureml.core import Workspace
from azureml.core.model import Model

def get_inference_model():
    global model
    if model == None:
        auth = MsiAuthentication()
        ws = Workspace(subscription_id="<your-subscription-id>",
                        resource_group="azureml-rg",
                        workspace_name="demo-ws",
                        auth=auth)
                        
        aml_model = Model(ws, 'mnist-xgb-model', version=1)
        model_path = aml_model.download(target_dir='.', exist_ok=True)
        model = InferenceModel(model_path)
    
    return model

app = Flask(__name__)

@app.route("/score", methods=['POST'])
def score():
    image_data = request.files.get('image')
    model = get_inference_model()
    preds = model.predict(image_data)
    
    return json.dumps({"preds": preds.tolist()})

model = None
