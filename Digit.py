import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 🌀 Transform: normalize grayscale images
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 📥 Load MNIST
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, download=True, transform=transform),
    batch_size=1000)

# 🧠 CNN Architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 🔁 Training loop
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

# ✅ Evaluation
model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        pred = model(images).argmax(dim=1)
        correct += (pred == labels).sum().item()

print("Test Accuracy:", 100. * correct / len(test_loader.dataset))

# 🔍 Prediction Visualization
samples = next(iter(test_loader))[0][:5]
outputs = model(samples)
predictions = outputs.argmax(dim=1)

for i in range(5):
    plt.imshow(samples[i][0], cmap="gray")
    plt.title(f"Predicted: {predictions[i].item()}")
    plt.axis("off")
    plt.show()
