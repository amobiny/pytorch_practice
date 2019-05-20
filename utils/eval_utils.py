import torch


def evaluate(model, criterion, data_loader, device_, flatten_input=False):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0
        for iter, (images, labels) in enumerate(data_loader):
            if flatten_input:
                images = images.reshape(-1, 28 * 28)
            images = images.to(device_)
            labels = labels.to(device_)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels)

        print('loss: {0:.4f} , Accuracy: {1:.02%}'.format(loss/(iter+1), correct / total))
