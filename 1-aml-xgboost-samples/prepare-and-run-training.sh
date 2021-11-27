#!/bin/bash
export SUBSCRIPTION="<your-subscription-id>"
export GROUP="azureml-rg"
export WORKSPACE="demo-ws"

az ml dataset create --file aml-mnist-dataset.yml --subscription $SUBSCRIPTION --resource-group $GROUP --workspace-name $WORKSPACE 
az ml compute create --file aml-compute-cpu.yml --subscription $SUBSCRIPTION --resource-group $GROUP --workspace-name $WORKSPACE 
az ml job create --file aml-job-train.yml --subscription $SUBSCRIPTION --resource-group $GROUP --workspace-name $WORKSPACE 