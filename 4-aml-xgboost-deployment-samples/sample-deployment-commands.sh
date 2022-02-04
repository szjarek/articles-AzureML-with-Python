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

# =================================
# = App Service Flask Application =
# =================================
export APP_SERVICE_PLAN="<your-app-service-plan-name>"
export APP_NAME="<your-app-name>" # NOTE: Globally unique name (it will be part of the URI)

# 1. Model Upload (optional)
az ml model create --name "mnist-xgb-model" --local-path "./mnist.xgb_model" --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE

# 2. Publish Flask App
# Create app service plan and publish application from local folder (app-service-code):
cd app-service-code
az webapp up --name $APP_NAME --plan $APP_SERVICE_PLAN --sku F1 --os-type Linux --runtime "python|3.8" --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP
cd ..

# Assign system identity to the created application
PRINCIPAL_ID=$(az webapp identity assign --name $APP_NAME --resource-group $RESOURCE_GROUP --output tsv --query "principalId") 
echo $PRINCIPAL_ID

# Assign Azure Machine Learning Workspace contributor role to the created application (you may need to wait a few seconds before it works)
az role assignment create --role contributor --assignee $PRINCIPAL_ID \
--scope /subscriptions/$AZURE_SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$AML_WORKSPACE

# 3. Test published application (it may take a while on the first request)
curl -X POST -F 'image=@./test-images/d4.png' https://$APP_NAME.azurewebsites.net/score

# 4. (OPTIONAL) Start tracing logs.
#    NOTE: It will block the terminal until interrupted (Ctrl+C)
# $ az webapp log tail --name $APP_NAME --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP

# ========================
# = AML Online Endpoint  =
# ========================
export ENDPOINT_NAME="<your-endpoint-name>" # NOTE: Globally unique name (it will be part of the URI)

# 5. Create AML Endpoint and Deployment
az ml online-endpoint create -n $ENDPOINT_NAME -f aml-endpoint.yml --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE 
az ml online-deployment create -n blue --endpoint $ENDPOINT_NAME -f aml-endpoint-deployment.yml --all-traffic --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE 

# 6. Obtain scoring URI and ENDPOINT_KEY
SCORING_URI=$(az ml online-endpoint show -n $ENDPOINT_NAME -o tsv --query scoring_uri --resource-group $RESOURCE_GROUP --workspace $AML_WORKSPACE)
echo $SCORING_URI

ENDPOINT_KEY=$(az ml online-endpoint get-credentials --name $ENDPOINT_NAME --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE -o tsv --query primaryKey)
echo $ENDPOINT_KEY

# 7. Test the endpoint
curl -X POST -F 'image=@./test-images/d5.png' -H "Authorization: Bearer $ENDPOINT_KEY" $SCORING_URI

# 8. Get endpoint logs
az ml online-deployment get-logs -n blue --endpoint $ENDPOINT_NAME --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE 

