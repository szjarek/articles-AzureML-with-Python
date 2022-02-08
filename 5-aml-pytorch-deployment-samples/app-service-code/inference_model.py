import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F
from torch import nn

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
        
class InferenceModel():
    def __init__(self, model_path):
        is_cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if is_cuda_available else "cpu")
        self.model = NetMNIST().to(self.device)
        self.model.load_state_dict(torch.load(model_path))

    def _preprocess_image(self, image_bytes):
        image = Image.open(image_bytes)
        
        # Reshape the image and convert it to monochrome
        image = image.resize((28,28)).convert('L')

        # Normalize data (we need to invert image, as training dataset images were inverted)
        image_np = (255 - np.array(image.getdata())) / 255.0

        # Return image data reshaped to a shape expected by model
        return torch.tensor(image_np).float().to(self.device)

    def predict(self, image_bytes):
        image_data = self._preprocess_image(image_bytes)

        with torch.no_grad():
            prediction = self.model(image_data.reshape(-1,1,28,28)).cpu().numpy()

        return np.argmax(prediction, axis=1)
