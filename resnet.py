import torch
from torchvision import datasets, models, transforms


TRAIN_DIR = "./frames/train"
TEST_DIR = "./frames/test"


def get_dataloader(img_dir, batch_size=64, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    img_dataset = datasets.ImageFolder(root=img_dir, transform=transform)
    img_loader = torch.utils.data.DataLoader(dataset=img_dataset, batch_size=batch_size, shuffle=shuffle)
    return img_loader


def get_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    return model


def train(model, train_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    model.eval()
    torch.save(model.state_dict(), "resnet.ckpt")


def test(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load("resnet.ckpt"))
    model.eval()

    total = 0
    correct = 0
    confusion_matrix = torch.zeros(1, 1)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("Accuracy of the model on the test images: {}%".format(100 * correct / total))


def main():
    batch_size = 64
    num_epochs = 2
    learning_rate = 0.001
    
    train_loader = get_dataloader(TRAIN_DIR, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(TEST_DIR, batch_size=batch_size, shuffle=False)
    model = get_model()
    # train(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate)
    model.load_state_dict(torch.load("resnet.ckpt"))
    model.eval()
    test(model, test_loader)


if __name__ == "__main__":
    main()
