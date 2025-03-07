# Long-term Prediction

This task aims to predict cargo capacity, namely cargo weight or volume, for civil aircraft of different types and flights.

# Required Python Packages

To install all required packages, run the following pip commands:
```bash
pip install os
pip install pandas
pip install torch
pip install numpy
pip install glob2
```

## Example 1: LSTM model
Long Short-Term Memory (LSTM) is a specialized variant of recurrent neural networks designed to capture long-term temporal dependencies in sequential data. It introduces gating mechanisms—namely the input, forget, and output gates—to regulate the flow of information, thereby mitigating the vanishing and exploding gradient problems that commonly arise in standard RNNs.
Run the following command in Python:

```bash
python LSTM.py --lr 0.0001 --num_layers 3 --batch_size 16 --epochs 600
```
![image](https://github.com/user-attachments/assets/e1a22112-2ce6-4095-b2c1-680b07090804)

## Example 2: CNN model
Convolutional Neural Network (CNN) constitutes a pivotal deep learning architecture extensively employed in computer vision tasks, including image classification, object detection, and semantic segmentation. By leveraging learnable convolutional filters and pooling mechanisms, CNNs hierarchically extract and refine spatial features from raw input data, thereby capturing both local patterns and global context. Among the notable CNN variants, Residual Networks (ResNets) incorporate skip connections to address vanishing gradients, enabling the training of deeper models and further advancing the representational capacity of CNN architectures.
Run the following command in Python:

```bash
python CNN.py --lr 0.0005 --num_layers 3 --batch_size 16 --epochs 800
```
![image](https://github.com/user-attachments/assets/6b2a8b90-eec4-48c3-a550-971fd6b68daf)

## Example 3: Regression model
Multiple linear regression (MLR) is a fundamental statistical modeling technique used to examine the relationship between a continuous dependent variable and multiple explanatory (independent) variables.
Run the following command in Python:
```bash
python regression.py --lr 0.0005 --num_layers 3 --batch_size 16 --epochs 300
```
![image](https://github.com/user-attachments/assets/982feb30-6b24-43a1-bb8b-b466f5479fd2)



