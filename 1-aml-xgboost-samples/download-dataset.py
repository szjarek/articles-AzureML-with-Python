import os
import urllib.request

DATA_FOLDER = 'datasets/mnist-data'
DATASET_BASE_URL = 'https://azureopendatastorage.blob.core.windows.net/mnist/'

os.makedirs(DATA_FOLDER, exist_ok=True)

urllib.request.urlretrieve(
    os.path.join(DATASET_BASE_URL, 'train-images-idx3-ubyte.gz'),
    filename=os.path.join(DATA_FOLDER, 'train-images.gz'))
urllib.request.urlretrieve(
    os.path.join(DATASET_BASE_URL, 'train-labels-idx1-ubyte.gz'),
    filename=os.path.join(DATA_FOLDER, 'train-labels.gz'))
urllib.request.urlretrieve(
    os.path.join(DATASET_BASE_URL, 't10k-images-idx3-ubyte.gz'),
    filename=os.path.join(DATA_FOLDER, 'test-images.gz'))
urllib.request.urlretrieve(
    os.path.join(DATASET_BASE_URL, 't10k-labels-idx1-ubyte.gz'),
    filename=os.path.join(DATA_FOLDER, 'test-labels.gz'))