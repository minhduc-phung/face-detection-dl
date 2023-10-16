import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim


from net import Net

train_dir = './train_images'    # folder containing training images
test_dir = './test_images'    # folder containing test images

transform = transforms.Compose(
    [transforms.Grayscale(),   # transforms to gray-scale (1 input channel)
     transforms.ToTensor(),    # transforms to Torch tensor (needed for PyTorch)
     transforms.Normalize(mean=(0.5,),std=(0.5,))]) # subtracts mean (0.5) and devides by standard deviation (0.5) -> resulting values in (-1, +1)

# Define two pytorch datasets (train/test)
train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

valid_size = 0.2   # proportion of validation set (80% train, 20% validation)
batch_size = 128   # number of examples in a training batch

# Define randomly the indices of examples to use for training and for validation
num_train = len(train_data)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

# Define two "samplers" that will randomly pick examples from the training and validation set
train_sampler = SubsetRandomSampler(train_new_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Dataloaders (take care of loading the data from disk, batch by batch, during training)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=10)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=10)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=10)

classes = ('noface','face')  # indicates that "1" means "face" and "0" non-face (only used for display)

criterion = torch.nn.CrossEntropyLoss()  # loss function
# Training

# loop over epochs: one epoch = one pass through the whole training dataset
# for epoch in range(1, n_epochs+1):
#   loop over iterations: one iteration = 1 batch of examples
#   for data, target in train_loader:
net = Net()
net = net.to(torch.device('cuda'))
n_epochs = 20
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.0001)
print("Training with {0} epochs with {1} batches each.".format(n_epochs, len(train_loader)))
for epoch in range(n_epochs):
    running_loss = 0.0
    print("Epoch", epoch+1, "/", n_epochs)
    for inputs, labels in train_loader:
        inputs = inputs.to(torch.device('cuda'))
        labels = labels.to(torch.device('cuda'))
        optimizer.zero_grad()  # zero the gradient buffers
        outputs = net(inputs)   # pass the data through the model
        loss = criterion(outputs, labels)  # compute the loss
        loss.backward()  # compute the gradients
        optimizer.step()  # update the weights
        running_loss += loss.item()
    print("Loss: ", running_loss)
print("Training finished.")


# Test
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(torch.device('cuda'))
        labels = labels.to(torch.device('cuda'))
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: {0:.4f}%'.format(
    100 * correct / total))

PATH = './face_recognition_net.pth'
torch.save(net.state_dict(), PATH)
