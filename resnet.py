import sys

import torch
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score, confusion_matrix


TRAIN_DIR = "./frames/train"
TEST_DIR = "./frames/test"
RESNET_LAYERS = 18
 

def get_dataloader(img_dir, batch_size=64, is_train=True):
    train_transform = transforms.Compose([
        # perform image augmentations
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(),
        # resize and normalize according to ResNet18 requirements
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    img_dataset = datasets.ImageFolder(root=img_dir, transform=train_transform if is_train else test_transform)
    img_loader = torch.utils.data.DataLoader(dataset=img_dataset, batch_size=batch_size, shuffle=is_train)
    return img_loader


def get_model():
    if RESNET_LAYERS == 18:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif RESNET_LAYERS == 50:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 1)

    return model


def train(model, train_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"resnet{RESNET_LAYERS}.ckpt{epoch + 1}")


def test(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total = 0
    correct = 0
    preds = torch.tensor([])
    targets = torch.tensor([])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images)
            outputs = outputs.squeeze()
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            preds = torch.cat((preds, predicted.cpu()))
            targets = torch.cat((targets, labels.cpu()))
        print(f"Accuracy of the model on the test images: {100 * correct / total:.2f}%")
    
    bcm = confusion_matrix(targets, preds)
    f1 = f1_score(targets, preds)
    print(f"F1 score: {f1:.2f}")
    print(f"Confusion matrix: \n{bcm}")


def main():
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.005
    
    train_loader = get_dataloader(TRAIN_DIR, batch_size=batch_size)
    test_loader = get_dataloader(TEST_DIR, batch_size=batch_size, is_train=False)
    model = get_model()

    if len(sys.argv) < 2:
        print("Usage: python resnet.py [train|test] [checkpoint file]")
        sys.exit(1)
    elif len(sys.argv) == 3:
        model.load_state_dict(torch.load(sys.argv[2]))

    if sys.argv[1] == "train":
        train(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate)
        test(model, test_loader)
    elif sys.argv[1] == "test":
        test(model, test_loader)


if __name__ == "__main__":
    main()
