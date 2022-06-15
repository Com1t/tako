import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dset
import torch.optim as optim
import time

import torchvision
from torchvision import datasets, transforms

torch.manual_seed(1)

image_x = 28
image_y = 28
image_channel = 1
output_channel = 10

# around 2040MB

class MNIST_NN(nn.Module):
    def __init__(self, image_x, image_y, image_channel, output_channel):
        super(MNIST_NN, self).__init__()
        
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=(5,5))
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(5,5))
        
        self.linear_1 = nn.Linear(1024, 256)
        self.linear_2 = nn.Linear(256, output_channel)
        
    def forward(self, image):
        out = F.max_pool2d(F.relu(self.conv_1(image)), kernel_size=(2,2))
        out = F.max_pool2d(F.relu(self.conv_2(out)), kernel_size=(2,2))
        
        out = self.linear_1(torch.flatten(out, 1))
        out = self.linear_2(out)
        
        return out
    
# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Device State:', device)

from torch_npz.FLDataset import FLDataset
train_data = FLDataset('/ML/FL_algo/nonIIDdataset/client_1.pickle', 'train')
test_data = FLDataset('/ML/FL_algo/nonIIDdataset/client_1.pickle', 'test')

trainLoader = dset.DataLoader(train_data, batch_size=256, shuffle=True)
testLoader = dset.DataLoader(test_data, batch_size=1024, shuffle=False)


model = MNIST_NN(image_x, image_y, image_channel, output_channel).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

begin = time.time()
num_epochs = 3
for epochs in range(3):
    grad_norm = 0.0
    start = time.time()
    for i, (data) in enumerate(trainLoader):
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        image = data[0].to(device)
        labels = data[1].to(device)
        
        # Step 3. Run our forward pass.
        res = model(image)
        
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(res, labels)
        
        if (i + 1) % 2 == 0:
            with torch.no_grad():
                correct_count = 0
                for _, (data) in enumerate(testLoader):
                    image = data[0].to(device)
                    labels = data[1].to(device)
                    output = model(image)
                    predict_label = torch.argmax(nn.Softmax(dim=1)(output), dim=1, keepdim=False)
                    correct_count += (predict_label == labels).float().sum()
                acc = correct_count / len(test_data)
            print('Epoch [{}/{}], Step [{}], Loss: {:.4f}, Acc : {:.4f}'.format(epochs + 1, num_epochs, i + 1, loss.item(), acc.item()))
            
        loss.backward()
        
        batch_grad_norm = 0.0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            batch_grad_norm += param_norm.item() ** 2
        grad_norm += batch_grad_norm ** 0.5
        
        optimizer.step()
    
    print(grad_norm)
    print((time.time() - start) / len(trainLoader))
print((time.time() - begin) / 1)