import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Define a custom dataset class with data augmentation
class CornerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'img')
        self.lab_dir = os.path.join(root_dir, 'lab')
        self.img_files = os.listdir(self.img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        lab_path = os.path.join(self.lab_dir, self.img_files[idx].replace('.png', '.txt'))

        image = Image.open(img_path).convert('RGB')
        label = self.load_label(lab_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def load_label(self, lab_path):
        with open(lab_path, 'r') as file:
            lines = file.readlines()
            label = [float(coord) for coord in lines[0].strip().replace(',', ' ').split()]
        return torch.tensor(label)

# Define a more complex CNN model
class ComplexCornerDetectionModel(nn.Module):
    def __init__(self):
        super(ComplexCornerDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)  # Assuming 2 for x, y coordinates

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transformations and create datasets and dataloaders
transform = transforms.Compose([
    transforms.RandomResizedCrop((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_dataset = CornerDataset('/home/eyerov/workspace/corner_detector/screenshots/train', transform=transform)
validate_dataset = CornerDataset('/home/eyerov/workspace/corner_detector/screenshots/validate', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=False)

# Initialize the more complex model, loss function, and optimizer
model = ComplexCornerDetectionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with data augmentation
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, labels in validate_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(validate_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_train_loss}, Validation Loss: {average_val_loss}')

# Testing the model on an unknown image
# Load an image and preprocess it
unknown_image_path = '/home/eyerov/workspace/corner_detector/screenshots/test/img/121.png'
unknown_image = Image.open(unknown_image_path).convert('RGB')
unknown_image = transforms.ToTensor()(unknown_image).unsqueeze(0).to(device)

# Make a prediction
model.eval()
with torch.no_grad():
    predicted_coords = model(unknown_image)

# Display or save the image with the predicted coordinates marked
# (You may need to adjust this part based on your visualization preferences)
print('Predicted Coordinates:', predicted_coords.squeeze().cpu().numpy())

# Save the trained model
torch.save(model.state_dict(), 'corner_detector_model.pth')
