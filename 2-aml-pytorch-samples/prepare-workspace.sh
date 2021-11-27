#!/bin/bash
export AZURE_SUBSCRIPTION="<your-subscription-id>"
export RESOURCE_GROUP="azureml-rg"
export AML_WORKSPACE="demo-ws"
export AML_SP="amluser"

az ad sp create-for-rbac --name $AML_SP \
                         --role contributor \
                         --scopes /subscriptions/$AZURE_SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP \
                         --sdk-auth

az ml workspace create --name $AML_WORKSPACE --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP 
az ml dataset create --file aml-mnist-dataset.yml --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE 

# The following steps should be executed by GitHub Actions:
# az ml compute create --file aml-compute-cpu.yml --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE 
# az ml job create --file aml-job-train.yml --subscription $AZURE_SUBSCRIPTION --resource-group $RESOURCE_GROUP --workspace-name $AML_WORKSPACE 