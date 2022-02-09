import requests
import azure.functions as func

def main(req: func.HttpRequest) -> func.HttpResponse:
    image_file = req.files['image']
    files = { "image": image_file }

    url = '<your-endpoint-uri>'
    api_key = '<your-api-key>' 
    headers = {'Authorization':('Bearer '+ api_key)}

    response = requests.post(url, files=files, headers=headers)
    return func.HttpResponse(response.content, status_code=response.status_code)
