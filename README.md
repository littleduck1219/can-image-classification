# drink_image_classification
캔 이미지를 분류 하는 모델을 만들어, 웹에서 웹캠으로 실행할 수 있도록 하였습니다.
![Screenshot 2023-01-18 at 9 29 45 PM](https://user-images.githubusercontent.com/107936957/214236266-a4783dde-9d74-4aa9-96ef-519973952619.png)

라벨은 총 10개로 진행하였으며, 인당 2개의 라벨을 맡아서 정보수집을 하였습니다.

정보수집은 일정 수량 직접 찍어 나머지는 합성을 하는 방법으로 진행하였습니다.

찍은 이미지와, 합성한 이미지의 사이즈가 달라 resize를 하려고 하였으나, 합성사진을 resize시에 형태가 깨지는 현상이 발생하여,
padding을 먼저 실시한 후에 resize하였습니다.

albumentation으로 최대한 라벨의 형태와 색을 건드리지 않는 것 들로만 선택하였습니다.


분류 모델은 ResNET_50을 사용하였습니다.

![Screenshot 2023-01-18 at 12 41 25 PM](https://user-images.githubusercontent.com/107936957/214236417-c520fcbb-4626-42c3-ac73-c192ab64d929.png)

epoch에 따른 accurancy결과 이며 test는 100%결과가 나왔습니다.


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
