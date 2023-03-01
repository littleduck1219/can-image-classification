# drink_image_classification
![Screenshot 2023-03-01 at 9 52 06 PM](https://user-images.githubusercontent.com/107936957/222144925-380c6050-b2f0-4817-91f0-dde6a1143e1f.png)
![Screenshot 2023-03-01 at 9 51 50 PM](https://user-images.githubusercontent.com/107936957/222144987-b9867cad-d040-4870-815f-41d574444113.png)

기능설명:
- 학습한 음료 10가지 웹에서 웹캠을 이용하여 인식 및 분류(이름 표시 및 유사도)
- 웹캠 on/off 기능
- 웹캠 출력화면 캡쳐 기능

사용언어 및 개발환경:
- 사용언어 : python, html, css, javascript
- 개발환경\
         Azure Virtual Rab :\
                  - Window Server 2019 Datacenter\
                  - Intel Xeon CPU E5-2690 2.60GHz, 2.59 GHz\
                  - 112GB RAM\
                  - NDVIA Tesla V100-16GB\
         Pycharm\
         Flask

Dataset:
- 10개의 라벨 각 300장씩 수집
- 찍은 이미지와, 합성한 이미지의 사이즈가 달라 resize를 하려고 하였으나, 
  합성사진을 resize시에 형태가 깨지는 현상이 발생하여, padding을 먼저 실시한 후에 resize하였습니다.

Augmentation:
```
data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
)
```
Data link = https://drive.google.com/file/d/1dkXELnY5UpUrX0o4NoqVa6Bc-e5C6maH/view?usp=share_link

Model:
- Resnet_50

Model lin = https://drive.google.com/file/d/1M0CGAgtcFYVYdDd3523KzhG9CpgYvsUL/view?usp=sharing

```
# model = torch.load("Resnet50_Left_Pretraine%d_ver1.1.pth") #Load model to CPU
model = models.resnet50()
model.fc = nn.Linear(in_features=2048, out_features=10)
# model.load_state_dict(torch.load("resnet18_AdamW_Lr001_E40.pt", map_location=torch.device('cpu')))
model.load_state_dict(torch.load("resnet_50_best.pt", map_location=torch.device('cpu')))
model = model.to(device)  # set where to run the model and matrix calculation
model.eval()  # set the device to eval() mode for testing
```

![Screenshot 2023-01-18 at 12 41 25 PM](https://user-images.githubusercontent.com/107936957/214236417-c520fcbb-4626-42c3-ac73-c192ab64d929.png)

Accurancy : 100


![capture 2023-01-17 01_36_42](https://user-images.githubusercontent.com/107936957/214236698-0a074051-a669-4800-8333-bf081d9cc18b.png)
![capture 2023-01-18 21_22_33](https://user-images.githubusercontent.com/107936957/214236705-3f287f40-533e-424c-a5a4-b0c8d8df0a38.png)
![capture 2023-01-18 21_22_54](https://user-images.githubusercontent.com/107936957/214236707-8fb663f5-74de-4fa4-9237-22b055b8d373.png)
![capture 2023-01-18 21_22_56](https://user-images.githubusercontent.com/107936957/214236709-1a317ed9-8e5f-4e0f-ac22-64d1ce8f3df1.png)
![capture 2023-01-18 21_23_54](https://user-images.githubusercontent.com/107936957/214236713-51c53fcd-4d64-44f2-853a-e0407b1b39fa.png)
![capture 2023-01-18 21_24_23](https://user-images.githubusercontent.com/107936957/214236716-110cce9c-c9ff-48ad-8cf2-e4c8292800c4.png)
![capture 2023-01-18 21_24_41](https://user-images.githubusercontent.com/107936957/214236721-938e71d9-9959-4423-8532-e5d6b98a3a15.png)
![capture 2023-01-18 21_25_02](https://user-images.githubusercontent.com/107936957/214236724-268330c7-1d9d-4097-b2bb-2446701e7004.png)

음료별로 진행한 웹캠테스트에서 준수한 결과를 보여주었습니다.

이 프로젝트는 교육중에 진행되었고 할 수 있는 부분까지만 일단 진행을 해 보았습니다.
