# Python 3.8.6
import time

# torch 1.7.0
# torchvision 0.8.1
# matplotlib 3.3.3
# numpy 1.19.4
# opencv-python 4.4.0
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

now = time.strftime("%H:%M:%S", time.localtime())
print("[TIMER] Process Time:", now)

TRAIN_EPOCHS = 40
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_TEST = 4

print("[INFO] Done importing packages.")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Input: 32 x 32 x 3 = 3072

        # Frame: 3 x 3, Stride: 1, Depth: 6, No padding
        # So Output size = 30 x 30 x 6 = 5400
        self.conv1 = nn.Conv2d(3, 6, 3)

        # Frame: 2 x 2, Stride: 2
        # So Output size = 15 x 15 x 6 = 1350
        self.pool2 = nn.MaxPool2d(2, 2)

        # Kernel: 2 x 2, Stride: 1, Depth: 16, No padding
        # So output size = 14 x 14 x 16 = 3136
        self.conv2 = nn.Conv2d(6, 16, 2)

        # Then repeat MaxPool2d
        # Then Output size = 7 x 7 x 16 = 784

        # Flatten 3-D 7 x 7 x 16 down to 1-dimensional 784

        # Fully Connected/Dense Layers
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

        # Activation function to use
        self.activation = F.relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)
        # Flatten
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

print("[INFO] Loading Traning and Test Datasets.")

transform = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
    download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset,
    batch_size = BATCH_SIZE_TRAIN, shuffle = True)
testset = torchvision.datasets.CIFAR10(root = './data', train = False,
    download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset,
    batch_size = BATCH_SIZE_TEST, shuffle = True)

print("[INFO] Done loading data.")

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

trainiter = iter(trainloader)
for i in range(4):
    images, labels = trainiter.next()
    print('  '.join(f"{classes[labels[j]]}" for j in range(4)))
    imshow(torchvision.utils.make_grid(images))

net = Net()
print("Network:", net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

for epoch in range(TRAIN_EPOCHS):
    now = time.strftime("%H:%M:%S", time.localtime())
    print("[TIMER] Process Time:", now)
    print(f"Beginning Epoch {epoch + 1}...")
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 500 == 499:
            print(f"Epoch: {epoch + 1}, Mini-Batches Processed: {i + 1:5}, Loss: {running_loss/2000:3.5}")
            running_loss = 0.0

    now = time.strftime("%H:%M:%S", time.localtime())
    print("[TIMER] Process Time:", now)
    print("Starting validation...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            outputs = net(images)
            # For overall accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"[TRAINING] {correct} out of {total}")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            # For overall accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"[VALIDATION] {correct} out of {total}")


print("[INFO] Finished training.")

testiter = iter(testloader)
images, labels = testiter.next()

print('Ground Truth:',' '.join(f"{classes[labels[j]]:5}" for j in range(4)))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted:',' '.join(f"{classes[predicted[j]]:5}" for j in range(4)))
imshow(torchvision.utils.make_grid(images))

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        # For overall accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # For class-by-class accuracy
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(BATCH_SIZE_TEST):
            label = labels[i]
            try:
                class_correct[label] += c[i].item()
            except:
                class_correct[label] += c.item()
            class_total[label] += 1

print(f"Accuracy of the network on the 10000 test items: {100 * correct / total:.4}%")

for i in range(10):
    print(f"Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.3}%")

now = time.strftime("%H:%M:%S", time.localtime())
print("[TIMER] Process Time:", now)

trainiter = iter(trainloader)
for i in range(4):
    images, labels = trainiter.next()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    print('  '.join(f"{classes[labels[j]]}" for j in range(4)))
    print('  '.join(f"{classes[predicted[j]]}" for j in range(4)))
    imshow(torchvision.utils.make_grid(images))
