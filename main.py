# imports

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets, transforms
from torch.optim import Adam


#initiating instances
transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ]
)
train_data = datasets.ImageFolder(
    root='archive/train',
    transform=transform
)
test_data = datasets.ImageFolder(
    root='archive/test',
    transform=transform
)
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)
test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)
class PClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(1),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128,2)
        )
    def forward(self,x):
        return self.net(x)
model = PClassifierModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(),lr = 0.0001)
epochs = 5
for epoch in range(epochs):
    for image,labels in train_loader:
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output,labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    print(f'Epoch : {epoch} | Loss : {loss}')

model.eval()
total_correct = 0
total_images = 0
for image,labels in test_loader:
    output = model(image)
    pred = torch.argmax(output,dim=1)
    total_images += labels.size(0)
    total_correct += (pred == labels).sum().item()
accuracy = (total_correct / total_images) * 100
torch.save(model.state_dict(),'PCLNetwork.pth')

print(accuracy)
