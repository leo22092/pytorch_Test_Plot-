import torch
from torchvision import datasets, transforms

# Define the transformations to apply to the images
transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # Resize the images to a fixed size
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the images
])

# Define the path to the root directory containing the class sub-directories
# data_dir = 'C:\\Users\\shitosu\\Desktop\\Mini_dataset'
data_dir="C:\\Users\\shitosu\\Documents\\PhD Data\\Gest224"
# Load the data using ImageFolder
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Split the dataset into training and validation sets
train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])  # 80% for training, 20% for validation

# Create data loaders for the training and validation sets
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False)
