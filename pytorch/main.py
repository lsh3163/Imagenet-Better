import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from model import *
from torch.utils.model_zoo import load_url as load_state_dict_from_url

N_ITERS = 20000
MODEL = "mobilenet_v2"
TASK = 2
N_CLASSES = 10
BATCH_SIZE = 64

transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=32)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=8)

NUM_EPOCHS = N_ITERS // (len(trainset)//256)
print(NUM_EPOCHS)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Task number 2, 3

print("Task {} ".format(TASK))
if TASK==2:
    if MODEL=="mobilenet_v1":
        model = mobilenet_v1(n_classes=N_CLASSES)
    elif MODEL=="mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Linear(1280, N_CLASSES)
else:
    if MODEL=="mobilenet_v1":
        model = mobilenet_v1(n_classes=N_CLASSES)
    elif MODEL=="mobilenet_v2":
        model = mobilenet_v2(n_classes=N_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, total_steps=None, epochs=5, steps_per_epoch=5021,
                                                    pct_start=0.0,
                                                    anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85,
                                                    max_momentum=0.95, div_factor=100.0)
for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    print("Epoch {}".format(epoch))
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
    scheduler.step()
print('Finished Training')

PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)

if TASK==2:
    model = models.mobilenet_v2(pretrained=True)
else:
    if MODEL=="mobilenet_v1":
        model = mobilenet_v1(n_classes=N_CLASSES)
    elif MODEL=="mobilenet_v2":
        model = mobilenet_v2(n_classes=N_CLASSES)
        model.classifier = nn.Linear(1280, N_CLASSES)
model.load_state_dict(torch.load(PATH))

class_correct = list(0. for i in range(N_CLASSES))
class_total = list(0. for i in range(N_CLASSES))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(BATCH_SIZE):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

max_acc = 0.0
for i in range(N_CLASSES):
    max_acc = max(max_acc, class_correct[i] / class_total[i])
print("Top 1 acc : {}".format(max_acc))
