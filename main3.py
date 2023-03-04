import copy
import os.path
import sys

from customdata3 import customData

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import models
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# augmentation, transform ==============================================================================================
train_transform = A.Compose([
    # A.SmallestMaxSize(max_size=160),
    # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.09, rotate_limit=25, p=0.6),
    A.Resize(height=224, width=224),
    A.CLAHE(p=1),
    # A.RandomShadow(p=0.5),
    A.RandomFog(p=0.4),
    # A.RandomSnow(p=0.4),
    A.RandomBrightnessContrast(p=0.5),
    # A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.6),
    # A.ShiftScaleRotate(shift_limit=5, scale_limit=0.09, rotate_limit=25, p=1),
    # A.GaussNoise(p=0.5),
    # A.Equalize(p=0.5),
    A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
valid_transform = A.Compose([
    # A.SmallestMaxSize(max_size=160),
    A.Resize(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# data ==============================================================================================================
train_dataset = customData("./data/train", transform=train_transform)
valid_dataset = customData("./data/val", transform=valid_transform)

# data loader ==========================================================================================================
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# model loader =========================================================================================================
# name = "swin_tiny"
# HUB_URL = "SharanSMenon/swin-transformer-hub:main"
# MODEL_NAME = "swin_tiny_patch4_window7_224"
# model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True)

# name = "swin_tiny"
# model = models.swin_t(weights="IMAGENET1K_V1")
# model.head = nn.Linear(in_features=768, out_features=10)

# name = "resnet18"
# model = models.resnet18(pretrained=False)
# model.fc = nn.Linear(in_features=512, out_features=10)

name = "resnet34"
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=10)

# name = "resnet_50"
# model = models.resnet50(pretrained=True)
# model.fc = nn.Linear(in_features=2048, out_features=10)

# name = "efficientnet_b3"
# model = models.efficientnet_b3(pretrained=True)
# model.classifier[1] = nn.Linear(1536, 10)

# name = "vgg19"
# model = models.vgg19(pretrained=True)
# model.classifier[6] = nn.Linear(in_features=4096, out_features=10)

# name = "DeIT"
# model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
# model.head = nn.Linear(in_features=192, out_features=10)

# check features
# print(model)
# exit()

model.to(device)

# hyper parameter ======================================================================================================
loss_function = LabelSmoothingCrossEntropy()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam,  AdamW, RMSprop
epochs = 25
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

best_valid_acc = 0.0

train_steps = len(train_loader)
valid_steps = len(valid_loader)

# save =================================================================================================================


# train loop ===========================================================================================================
def train(model, loss_function, train_loader, valid_loader, best_valid_acc, optimizer,
          scheduler, epochs, train_steps, valid_steps, name, device=device):

    save_path = f"./{name}_best.pt"
    dfForAccuracy = pd.DataFrame(index=list(range(epochs)),
                                 columns=["Epoch", "Train_Accuracy", "Train_Loss", "Valid_Accuracy", "Valid_Loss"])
    if os.path.exists(save_path):
        best_valid_acc = max(pd.read_csv(f"./{name}_modelAccuracy.csv")["Accuracy"].tolist())

    for epoch in range(epochs):
        running_loss = 0
        valid_acc = 0
        train_acc = 0

        model.train()
        train_bar = tqdm(train_loader, file=sys.stdout, colour="magenta")
        for step, data in enumerate(train_bar):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            train_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}], loss >> {loss.data:.3f}"

# validation loop ======================================================================================================
        model.eval()  # set validation mode and no more training
        with torch.no_grad():
            valid_loss = 0
            valid_bar = tqdm(valid_loader, file=sys.stdout, colour="magenta")
            for data in valid_bar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item()
                valid_acc += (torch.argmax(outputs, dim=1) == labels).sum().item()

        valid_accuracy = valid_acc / len(valid_dataset)
        train_accuracy = train_acc / len(train_dataset)

        dfForAccuracy.loc[epoch, "Epoch"] = epoch + 1
        dfForAccuracy.loc[epoch, "Train_Accuracy"] = round(train_accuracy, 3) # round는 반올림해준다.
        dfForAccuracy.loc[epoch, "Train_Loss"] = round(running_loss / train_steps, 3)
        dfForAccuracy.loc[epoch, "Val_Accuracy"] = round(valid_accuracy, 3)
        dfForAccuracy.loc[epoch, "Val_Loss"] = round(valid_loss / valid_steps, 3)
        print(f"epoch [{epoch + 1}/{epochs}] train_loss {(running_loss / train_steps):.3f}"
              f"train acc : {train_accuracy:.3f} valid_acc : {valid_accuracy:.3f}"
              )
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            torch.save(model.state_dict(), save_path)

        if epoch == epochs - 1:
            dfForAccuracy.to_csv(f"./{name}_modelAccuracy.csv", index=False)

    torch.save(model.state_dict(), "./last.pt")


# visualize augmentation ===============================================================================================
def visualize_augmentations(dataset, idx=0, samples=20, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform
                                   if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
            image, _ = dataset[idx]
            ax.ravel()[i].imshow(image)
            ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


# visualize graph ======================================================================================================
def graph(name):
    csv = pd.read_csv(f"./{name}_modelAccuracy.csv")
    plt.plot(csv['Train_Accuracy'])
    plt.plot(csv['Valid_Accuracy'])
    plt.plot(csv['Train_Loss'])
    plt.plot(csv['Valid_Loss'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(["Train_Accuracy", "Valid_Accuracy", "Train_Loss", "Valid_Loss"], loc='upper left')
    plt.show()


# test =================================================================================================================
def test(model, valid_loader, name, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (image, labels) in enumerate(valid_loader):
            image, label = image.to(device), labels.to(device)
            output = model(image)
            _, argmax = torch.max(output, 1)

            total += image.size(0)
            print(image.size(0), total)  # 128
            correct += (label == argmax).sum().item()

        acc = correct / total * 100
        print("acc for {} image : {:.2f}%".format(
            total, acc
        ))


if __name__ == "__main__":

    # train, validation ================================================================================================
    # train(model, loss_function, train_loader, valid_loader,  best_valid_acc, optimizer,
    #       scheduler=exp_lr_scheduler, epochs=epochs, train_steps=train_steps, valid_steps=valid_steps,
    #       name=name, device=device)

    # test =============================================================================================================
    # test(model, valid_loader, name=name, device=device)
    # model.load_state_dict(torch.load(f"./{name}_best.pt", map_location=device))

    # visualize_augmentations ==========================================================================================
    # visualize_augmentations(customData("./data/train", transform=train_transforms))

    # graph ============================================================================================================
    # graph(name=name)

