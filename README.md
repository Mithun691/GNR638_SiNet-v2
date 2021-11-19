# GNR638_SiNet-v2
Implementation of SiNet-v2 network for Concealed Object Detection (COD), which aims to identify objects that are visually embedded in their background.

# Training pipeline
- Download and place the COD-10K dataset in the same directory as the code
- python train.py for training the SiNet-v2 Network on the dataset
- Auxiliary args for running train.py --epoch, --lr, --batchsize, --trainsize, --decay_rate, --decay_epoch, --train_root, --val_root
- For example : python train.py --epoch 10 --lr 1e-5 --batchsize 20     ( for changing num epochs, learning rate & minibatchsize from default setting )

To understand the network architecture have a look at our presentation,
https://docs.google.com/presentation/d/1oB76b41hPfaMUQnz3UUtJUsZeg2MEp5Rti97d4DmPFM/edit#slide=id.g10293833661_2_149

# Results

## Visualizing the detected object mask
<img src="https://github.com/Mithun691/GNR638_SiNet-v2/blob/main/img1.jpg" width=50% height=50%><img src="https://github.com/Mithun691/GNR638_SiNet-v2/blob/main/maks1.png" width=50% height=50%>
<img src="https://github.com/Mithun691/GNR638_SiNet-v2/blob/main/img2.jpg" width=50% height=50%><img src="https://github.com/Mithun691/GNR638_SiNet-v2/blob/main/mask2.png" width=50% height=50%>
<img src="https://github.com/Mithun691/GNR638_SiNet-v2/blob/main/mask3.jpg" width=50% height=50%><img src="https://github.com/Mithun691/GNR638_SiNet-v2/blob/main/img3.png" width=50% height=50%>

## Training MAE v/s Epoch
<img src="https://github.com/Mithun691/GNR638_SiNet-v2/blob/main/learning_curve.png" width=70% height=70%>

## Visualizing the COD-10K results
<img src="https://github.com/Mithun691/GNR638_SiNet-v2/blob/main/COD10K-1.jpg" width=70% height=70%>
<img src="https://github.com/Mithun691/GNR638_SiNet-v2/blob/main/COD10K-2.jpg" width=70% height=70%>
