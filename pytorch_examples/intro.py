import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

import torchvision.datasets as datasets

import numpy as np
import pickle

import random

torch.manual_seed(1)
torch.cuda.current_device()


class LinearNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(LinearNet, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


class CNN(torch.nn.Module):
    def __init__(self, N_features):
        super(CNN, self).__init__()
        #          N, channels, Height, Width(features)
        # shape = (?, 1, 120, 46)
        #      N, Channels, Length
        # (batch, N_steps, N_features)

        """
        nn.conv1d -> (N, C, L)
        C = channels | features | filters
        L = sequence length

        kernel size = number of time steps to view
        """
        """
        nn.conv2d -> (N, C, H, W)
        C = channels -> [3] for RGB, [1] for Grey
        H = img height
        W = img width
        """

        keep_prob = 0.6
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(N_features, 32, kernel_size=5, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob)
            )
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - keep_prob))
          
        self.fc1 = torch.nn.Linear(128 * 15, 8, bias=True)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))

        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(8, 1, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters
        
        

    def forward(self, x):
        #print("--------")
        out = self.layer1(x)
        #print(out.size())
        out = self.layer2(out)
        #print(out.size())
        out = self.layer3(out)

        #print(out.size())
        out = out.view(out.size(0), -1)
        #print(out.size())
        out = self.fc1(out)
        #print(out.size())
        out = self.fc2(out)
        
        #print(out.size())
        return out

    
train_data = pickle.load(open('../data/train.hadoop', 'rb'))

#x = np.expand_dims(train_data['x'], axis=1)
x = train_data['x']
x = np.swapaxes(x, 1, 2)
y = train_data['y']
y = np.expand_dims(np.array(y, np.float32), axis=1)

print(x.shape)
print(y.shape)

def make_loader(x, y, batch_size):
    zipped = list(zip(x, y))
    random.shuffle(zipped)
    for i in range(0, len(zipped)-batch_size, batch_size):
        yield (
            np.array(x[i:i+batch_size], np.float32),
            np.array(y[i:i+batch_size], np.float32)
        )

n_feature = 8
n_hidden = 16
n_output = 1

NN = CNN(N_features=x.shape[1]).cuda()
optimizer = torch.optim.Adam(NN.parameters(), lr=0.001)
loss_func = torch.nn.L1Loss()  # this is for regression mean squared loss

EPOCHS = 100
BATCH_SIZE = 32

for epoch in range(EPOCHS):
    print(f"---------------------{epoch}----------------------")
    loader = make_loader(x, y, BATCH_SIZE)
    for step, (x_batch, y_batch) in enumerate(loader):
        x_Tensor = torch.Tensor(x_batch).cuda()
        y_Tensor = torch.Tensor(y_batch).cuda()

        x_Variable = Variable(x_Tensor).cuda()
        y_Variable = Variable(y_Tensor).cuda()

        #print(x_Variable.size())

        pred = NN(x_Variable)
        loss = loss_func(pred, y_Variable)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        if step % 500 == 0:
            for i in range(4):
                print(y_batch[i], pred[i])
            print(loss)
