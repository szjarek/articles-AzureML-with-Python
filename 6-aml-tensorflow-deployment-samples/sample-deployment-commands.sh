#!/bin/bash

# NOTE:
# This script illustrates steps described in the related article.
# For the best learning experience, it is intented to be executed manually, command-by-command, not as a whole.

# IMPORTANT: Please set all properties starting with "<your-*" to proper values.

export AZURE_SUBSCRIPTION="<your-subscription-id>"
export RESOURCE_GROUP="azureml-rg"
export AML_WORKSPACE="demo-ws"
export LOCATION="westeurope"

# 0. PREREQUISITES
# $ az login

# 1. Model Upload (optional)
az ml model create --name "mnist-tf-model" --local-path "./mnist-tf-model.h5" --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE

# ========================
# = AML Online Endpoint  =
# ========================

export ENDPOINT_NAME="<your-endpoint-name>" # NOTE: Globally unique name (it will be part of the URI)

# 2. Create AML Endpoint and Deployment
az ml online-endpoint create -n $ENDPOINT_NAME -f aml-endpoint.yml --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE 
az ml online-deployment create -n blue --endpoint $ENDPOINT_NAME -f aml-endpoint-deployment.yml --all-traffic --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE 

# 3. Obtain scoring URI and ENDPOINT_KEY
SCORING_URI=$(az ml online-endpoint show -n $ENDPOINT_NAME -o tsv --query scoring_uri --resource-group $RESOURCE_GROUP --workspace $AML_WORKSPACE)
echo $SCORING_URI

ENDPOINT_KEY=$(az ml online-endpoint get-credentials --name $ENDPOINT_NAME --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE -o tsv --query primaryKey)
echo $ENDPOINT_KEY

# 4. Test the endpoint
curl -X POST -F 'image=@./test-images/d5.png' -H "Authorization: Bearer $ENDPOINT_KEY" $SCORING_URI

# 5. Get endpoint logs
az ml online-deployment get-logs -n blue --endpoint $ENDPOINT_NAME --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE 

# ===========================
# = Azure Function Requests =
# ===========================

# Manual TODO list:
# * Replace <your-endpoint-uri> in the function's __init__.py file with the $SCORING_URI value
# * Replace <your-api-key> in the function's __init__.py file with the $SCORING_KEY value
# * Publish the function app (from the aml-function-proxy folder)

# 6. Send request to Function
FUNCTION_APP_NAME="<your-function-app>"
curl -X POST -F 'image=@./test-images/d7.png' https://$FUNCTION_APP_NAME.azurewebsites.net/api/AmlMnistHttpTrigger
