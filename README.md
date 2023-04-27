# YOLOv7 NAS for TANGO Framework

## Members
* [Wonseon Lim](https://github.com/laymond1)
* [Heon-Sung Park](https://github.com/hopo55)
* [Sang-Hyeong Jin](https://github.com/feVeRin)   

## Goal

[YOLOv7](https://github.com/WongKinYiu/yolov7)을 Base Architecture로 Neural Architecture Search(NAS), specifically [Once-for-All Paper](https://github.com/mit-han-lab/once-for-all), 방식으로 Scalable YOLOv7 버전을 만드는 것이 목표.

## 2 ways to implement our idea

1. Control the number of E-ELAN blocks in each a stage 

    : 각 Stage에서 E-ELAN 블록의 수(depth)를 조절하는 방식

2. Control the number of layers(branch) in each an E-ELAN block

    : 각 E-ELAN block 내에서 layer의 분기 수를 조절하는 방식