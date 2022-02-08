from fastapi import FastAPI, File

from io import BytesIO
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
        aml_model = Model(ws, 'mnist-pt-model', version=1)

        model_path = aml_model.download(target_dir='.', exist_ok=True)

        model = InferenceModel(model_path)
    
    return model

app = FastAPI(title="PyTorch MNIST Service API", version="1.0")

@app.post("/score")
async def score(image: bytes = File(...)):
    if not image:
        return {"message": "No image_file"}

    model = get_inference_model()
    preds = model.predict(BytesIO(image))
    
    return {"preds": str(preds)}

model = None
