# Image Caption: Project Overview 
The goal of this project is to combine computer vision and natural language processing to create a system that can understand and describe the content of images. This can have various applications, such as aiding visually impaired individuals in understanding images or improving image search functionality.

## Code and Resources Used 
**Python Version:** 3.10  
**Packages:** pandas, numpy, sklearn, matplotlib, tensorflow, keras  
**Data source:** https://www.kaggle.com/datasets/aladdinpersson/flickr8kimagescaptions/download
**Article:** https://arxiv.org/pdf/1706.03762

## Data Structure
Dataset consist of 8000 images with 5 associated captions.

![alt text](https://github.com/HalyshAnton/Image-Caption-with-Transformer/blob/main/data_visual.png)

## Model Building
According to article model consist of three parts:
* **Image Embedding:** I used pretrained EfficientNetB0 with additional Reshape layer that return output with shape (num_features, last_kernel_size)
* **Encoder:** take output from Image Embedding and apply MultiHeadAttention with small forward feed network(for more detail read article)
* **Decoder:** take embedded caption(with positional encoding) apply Masked MultiHeadAttention(to avoid information from furture) and then apply MultiHeadAttention with output from Encoder. After another small forward feed network return propability distribution of the next word in caption.

  ![alt text](https://github.com/HalyshAnton/Image-Caption-with-Transformer/blob/main/model_achitecture.png)

## Model Performance
I have used Early Stop method and Adam optimizer. Also I have used image augmentation: horizontal flip, rotation and contrast. After 7 epochs I got 38% accurancy with validation data. Using trained model I generated captions for 5 random images:
![alt text](https://github.com/HalyshAnton/Image-Caption-with-Transformer/blob/main/predicted_captions.png)
