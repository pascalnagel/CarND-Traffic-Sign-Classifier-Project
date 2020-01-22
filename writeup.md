# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figs/class_dist.png "Class distribution"
[image2]: ./figs/YUV.png "Images in Y-channel grayscale"
[image3]: ./figs/learning_curve.png "Learning curve"
[image4]: ./test_downloads/1.jpg "Traffic Sign 1"
[image5]: ./test_downloads/2.jpg "Traffic Sign 2"
[image6]: ./test_downloads/3.jpg "Traffic Sign 3"
[image7]: ./test_downloads/4.jpg "Traffic Sign 4"
[image8]: ./test_downloads/5.jpg "Traffic Sign 5"
[image9]: ./figs/softmax1.png "Softmax 1"
[image10]: ./figs/softmax2.png "Softmax 2"
[image11]: ./figs/softmax3.png "Softmax 3"
[image12]: ./figs/softmax4.png "Softmax 4"
[image13]: ./figs/softmax5.png "Softmax 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I started off with a small EDA of the provided dataset:

 * Number of training and test samples
 * Resolution of the images
 * Number of classes
 * Figures of the class distribution in each of the dataset splits
 * Visualization of randomly chosen 25 images in the dataset

#### 2. Include an exploratory visualization of the dataset.

The figure below shows the class distribution for taining (left), validation (middle) and test (right) set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Following the assumption that most information is carried by shapes and textures, rather than colors, I converted the color images to grayscale. To be more specific, I converted the images to YUV color space of which I used the Y channel, as was suggested by the paper by Sermanet and LeCun.

Below is a sample of Y-channel traffic sign images:

![alt text][image2]

As a second step, I normalized the 8-bit grayscale values to the range [-1,1] to make training more stable and reshaped each image back to the shape (32,32,1), which the network expects.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

As a first step I tried out the LeNet architecture with only slight adaptations and achieved up to 94% validation error. To improve on this I increased the depth of the network and the number of filters to obtain the following architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Normalized Y-channel image   			| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x32 		    		|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 12x12x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x64 		    		|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 3x3x128 	|
| RELU					|												|
| Convolution 2x2     	| 1x1 stride, valid padding, outputs 2x2x128 	|
| RELU					|												|
| Flatten       	    | outputs size 512  							|
| Fully connected		| outputs size 80   							|
| RELU					|												|
| Fully connected		| outputs size n_classes=43 					|
| Softmax				|             									|
 
 The number of layers was increased to capture more levels of abstractions in the traffic sign images. It was not increased beyond this point in the anticipation of unstable training due to the vanishing gradient problem, since we are not using any measures to avert it, as modern very deep architectures do. We have no residuals, no batch normalization, nor do we initialize the weights ideally (constant variance instead of e.g. He initialization).

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

As in the tutorial LeNet training, I've used the Adam optimizer, which combines SGD with the momentum method and an adapatively chosen step size. The learning rate was chosen to be 0.001.

The batch size was kept at a constant 128 throughout training for simplicity. Following the behavior found in this [Google Brain paper](https://openreview.net/pdf?id=B1Yy1BxCZ), starting off with a small batch size and increasing it each epoch should have yielded faster convergence, but training time was not the primary concern for this small model and dataset.

The model was trained for 50 epochs, after which no significant improvement on the validation error could be observed.

The figure below illustrates the learning curves of training (blue) and validation (orange) accuracy throughout training:

![alt text][image3]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 97.7%
* test set accuracy of 96.4%

The iterative approach and reasoning behind chosing architecture and training methods has been described in the previous sections.
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

5 German traffic signs with an appropriate license which I found were:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The 4th image might pose a challenge since it is rather damaged. However most challenging will be the last sign, since the 30 km/h sign is not featured in the common round shape.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                        |     Prediction	        				| 
|:-------------------------------------:|:-----------------------------------------:| 
| Stop Sign      		                | Stop sign   								| 
| Yield     		                	| Yield 									|
| Priority Road		                	| Priority Road								|
| Right-of-way at the next intersection	| Right-of-way at the next intersection		|
| Speed limit 30km/h                	| Roundabout mandatory      				|

The model correctly predicted 4 of 5 traffic signs, yielding an accuracy of 80%. This is worse than the 96% of the test set, however a test set size of 5 is not significant enough to prove a deviation with any confidence. In addition the uncommon shape of the 30 km/h sign is likely outside the training/validation/test set distribution and the model is just unable to generalize.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top five softmax probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999977     			| Stop sign   									| 
| 1.000000 				| Yield 										|
| .9999964				| Priority Road								    |
| 1.000000     			| Right-of-way at the next intersection			|
| .568  			    | Roundabout mandatory    					    |

We can see, that the network is unsure how to classify the 30km/h sign. In addition we can clearly see the tendency of neural networks to drastically overstate their confidence.

The following figures illustrate the top 5 softmax probabilities for each prediction:

![alt text][image9] ![alt text][image10] ![alt text][image11] 
![alt text][image12] ![alt text][image13]