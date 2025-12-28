# Classify-leaves
Machine Learning project 2025.12  

https://www.kaggle.com/competitions/classify-leaves  

The task is predicting categories of leaf images. This dataset contains 176 categories, 18353 training images, 8800 test images. Each category has at least 50 images for training.   

File descriptions:

train.csv - the training set  
test.csv - the test set  
sample_submission.csv - a sample submission file in the correct format (the labels are randomly generated so the initial accuracy is roughly 1/176)  
images/ - the folder contains all images  
Data fields  

image - the image path, such as `images/0.jpg`  
label - the category name  

I used a pretrained ResNet50 as the base CNN model and apply the techniques including:  
Data Augmentation, SGDmomentum, CosineAnnealingWarmRestarts 
to achieve more than 90% accuracy within 50 epoches.  
The training process runs on my personal RTX Geforce 4050.  