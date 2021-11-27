import os
import gzip
import struct
import numpy as np

import argparse
import mlflow

import torch
import torch.optim as optim

from torch.nn import functional as F
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from azureml.core import Run
from azureml.core.model import Model

def load_dataset(dataset_path):
    def unpack_mnist_data(filename: str, label=False):
        with gzip.open(filename) as gz:
            struct.unpack('I', gz.read(4))
            n_items = struct.unpack('>I', gz.read(4))
            if not label:
                n_rows = struct.unpack('>I', gz.read(4))[0]
                n_cols = struct.unpack('>I', gz.read(4))[0]
                res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
                res = res.reshape(n_items[0], n_rows * n_cols) / 255.0
            else:
                res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
                res = res.reshape(-1)
        return res
    
    X_train = unpack_mnist_data(os.path.join(dataset_path, 'train-images.gz'), False)
    y_train = unpack_mnist_data(os.path.join(dataset_path, 'train-labels.gz'), True)
    X_test = unpack_mnist_data(os.path.join(dataset_path, 'test-images.gz'), False)
    y_test = unpack_mnist_data(os.path.join(dataset_path, 'test-labels.gz'), True)

    return X_train.reshape(-1,28,28,1), y_train, X_test.reshape(-1,28,28,1), y_test

class NetMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) 
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
 
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.dropout(F.relu(self.conv2(x)), p=0.2), (2,2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DatasetMnist(torch.utils.data.Dataset):
    def __init__(self, X, y=None):
        self.X, self.y = X,y

        self.transform = transforms.Compose([
            transforms.ToTensor()])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        item = self.transform(self.X[index])
        if self.y is None:
            return item.float()
        
        label = self.y[index]
        return item.float(), np.long(label)

def get_aml_workspace():
    run = Run.get_context()
    ws = run.experiment.workspace
    return ws

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_known_args()[0]

    return args

def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()

    epoch_loss = 0.0
    epoch_acc = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        epoch_acc += (preds == target).sum().item()
        
        if batch_idx % 200 == 0 and batch_idx != 0:
            print(f"[{epoch:2d}:{batch_idx:5d}] \tBatch loss: {loss.item():.5f}, Epoch loss: {epoch_loss:.5f}")
            
    epoch_acc /= len(train_loader.dataset)
    
    print(f"[{epoch:2d} EPOCH] \tLoss: {epoch_loss:.6f} \tAcc: {epoch_acc:.6f}")
    mlflow.log_metrics({
        'loss': epoch_loss,
        'accuracy': epoch_acc})

def train_model(X, y, model_filename, epochs=5, batch_size=64):   
    RANDOM_SEED = 101

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    else:
        cuda_kwargs = {}
    
    train_dataset = DatasetMnist(X, y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, **cuda_kwargs)

    model = NetMNIST().to(device)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, epochs+1):
        train_epoch(model, device, train_loader, optimizer, epoch)

    torch.save(model.state_dict(), model_filename)

def evaluate_model(X, y, model_filename, batch_size=64):
    test_dataset = DatasetMnist(X)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = NetMNIST()
    model.load_state_dict(torch.load(model_filename))
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch_preds = model(batch).numpy()
            preds.extend(np.argmax(batch_preds, axis=1))

        accscore = (preds == y).sum().item()        
    accscore /= len(test_dataset)

    mlflow.log_metric('test_accuracy', accscore)

def register_model(ws, model_filename):
    model = Model.register(
        workspace=ws,
        model_name=model_filename,
        model_path=model_filename,
        model_framework=Model.Framework.PYTORCH,
        model_framework_version=torch.__version__
    )

def main():
    args = parse_arguments()

    ws = get_aml_workspace()
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    mlflow.start_run()

    X_train, y_train, X_test, y_test = load_dataset(args.data)
    model_filename = "mnist.pt_model"

    train_model(X_train, y_train, model_filename)
    evaluate_model(X_test, y_test, model_filename)
    register_model(ws, model_filename)
    
if __name__ == "__main__":
    main()