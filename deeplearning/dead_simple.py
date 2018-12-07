import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Custom_Loss(torch.nn.Module):
    def forward(self, solution_matrix, cost_matrix):
        _, path = solution_matrix.max(0)
        # print(path)
        # print(solution_matrix)
        # Stop
        n = cost_matrix.shape[0]
        cost = torch.autograd.Variable(torch.zeros(1), requires_grad=True)
        for i, city_idx in enumerate(path):

            #cost = torch.sum(cost + cost_matrix[src, dest])
            src = path[i]
            dest = path[(i + 1) % n]
            cost = cost.add(cost_matrix[src, dest])
        return cost

class Naive_Loss(torch.nn.Module):
    def forward(self, solution_matrix, cost_matrix):
        return torch.sum(solution_matrix.add(-cost_matrix)**2)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

cost_matrix = torch.ones([5,5], requires_grad=False, dtype=torch.float)
cost_matrix[-1:] = -1
#cost_matrix = torch.rand(5,5)
print(cost_matrix)

input_size = 5
hidden_size = 4
num_classes = 5
num_epochs = 1000
batch_size = 1
learning_rate = 0.01

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = Custom_Loss() # nn.CrossEntropyLoss()
#criterion = Naive_Loss() #nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
losses = []
for epoch in range(num_epochs):
        input = torch.rand(5,5)

        # Forward pass
        outputs = model(input)
        loss = criterion(outputs, cost_matrix)
        losses.append(loss.data[0])

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

plt.plot(losses)
plt.show()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0


    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
