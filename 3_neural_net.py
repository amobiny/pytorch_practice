import torch
import torch.nn as nn
import os

# Device configuration
from utils.eval_utils import evaluate
from utils.loader_utils import get_train_valid_loader, get_test_loader


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    input_size = 784
    hidden_size = 500
    num_classes = 10
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001

    # MNIST dataset
    root = r'C:\Users\z0042n0w\Desktop\pytorch_practice\data'
    if not os.path.exists(root):
        os.mkdir(root)

    train_loader, valid_loader = get_train_valid_loader(root, batch_size=batch_size, random_seed=0)
    test_loader = get_test_loader(root, batch_size=batch_size)

    # Fully connected neural network with one hidden layer
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Evaluate the model on validation data
        evaluate(model, criterion, valid_loader, device, flatten_input=True)

    # Test the model
    evaluate(model, criterion, test_loader, device, flatten_input=True)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model_nn.ckpt')


if __name__ == '__main__':
    run()
