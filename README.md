## Introduction
The model architecture is referenced from Microsoft CVPR2020 paper “Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition”.

[There Github is Here](https://github.com/microsoft/SGN)

Referencing this paper to try to use gcn in any general cases.


## Data
Data are the assets of IGS which is private.

They are 2D data which columns represent each features and rows represent behavior at specific time.


## Model Overview and architecture


The whole model architectur can be splited to Feature-Level-Module and Time-Level-Module

- Feature-Level-Module is to get feature map from single row.
- Time-Level-Module it to combine and get infromation from all feature maps.

![](https://i.imgur.com/yqSl2pR.png)

## Feature Level Module architecture
![](https://i.imgur.com/bAz65O1.png)

## Time Level Module architecture
![](https://i.imgur.com/YpXY4h8.png)


## Conclusion
- Final accuracy of IGS test-data: 92.1%
- Just an experiment to see the ability of GCN.




