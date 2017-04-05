#**Traffic Sign Recognition** 

##Writeup Template

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/explore_viz_1.png "Visualization"
[image2]: ./examples/explore_viz_2.png "Count"
[image3]: ./examples/normalization.png "Normalization"
[image4]: ./realtest/1.jpg "Traffic Sign 1"
[image5]: ./realtest/2.jpg "Traffic Sign 2"
[image6]: ./realtest/3.jpg "Traffic Sign 3"
[image7]: ./realtest/4.jpg "Traffic Sign 4"
[image8]: ./realtest/5.jpg "Traffic Sign 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

Here is a link to my [project code](https://github.com/ken107/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and 4th code cell of the IPython notebook.  

I picked one image from each category and display them in a grid:

![signs][image1]

Then I printed out the number of training samples for each category:

![count of training samples][image2]

---

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 5th and 6th code cells of the IPython notebook.

First I convert the images to grayscale.  Then I normalize the pixel values of each image, the biggest value becomes 1, the smallest 0.  As suggested during the lessons, without this normalization, we would have had to adjust the learning rate to accommodate.

Then I display 3 random images from the training, validation, and test datasets:

![normalized images][image3]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The original dataset were already split into training, validation, and testing.  I did not augment the datasets.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:----------------------|:----------------------------------------------| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24					|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x36	|
| RELU					|												|
| Convolution 5x5		| 1x1 stride, same padding, outputs 6x6x48		|
| RELU					|												|
| Fully connected		| outputs 120, dropout 50%						|
| Fully connected		| outputs 84, dropout 50%						|
| Fully connected		| outputs softmax probs							|
|						|												|
|						|												|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

I use SoftMax cross entropy with the Adam Optimizer.  I define a `train` function that breaks the training dataset into batches and run them one by one in the session.  And define a `evaluate` function that computes the accuracy of the predictions, which will be called on the validation and test datasets.

I tried various batch_sizes, learning rates, and # of epochs, but generally I set them to 128, .001, and 10 respectively.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:

* Training set accuracy of .993
* Validation set accuracy of .966
* Test set accuracy of .938


##### What was the first architecture that was tried and why was it chosen?
I chose LeNet, as suggested

##### What were some problems with the initial architecture?
It did not achieve the required accuracy

##### How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I did a lot of trials to try to understand how each hyperparameter affects the accuracy performance of the network.  I recorded the results of my trials here:  https://drive.google.com/open?id=0B3HII9xilWfCLVl5aUpqcHQ3M28

I actually didn't realize, until I added code to calculate the "Training accuracy" much later, that the model has already reached its limit, and regardless what I did I would not get a significant improvement.  Moreover by trying hard to improve the validation accuracy when the training accuracy had already maxed out, I had been overfitting the model.

My observation is:
- Larger batch size reduced accuracy of the output, for some reason
- Increasing epochs isn't necessary, 10 seems to be more than enough
- removing one of the fully connected layer does not significantly affect accuracy
- adding an additional convolution layer seems to improve validation accuracy
- smaller filter dimensions like 3x3 appears to reduce accuracy
- increasing depth of the convolution layers definitely improve accuracy, and is the key to achieving .93 project requirement
- adding dropouts to the fully connected layers also improve accuracy
- I also tried feeding the outputs of both convolution layers to the fully connected layers, as suggested in one of the lessons, but no noticeable improvement were detected

##### Which parameters were tuned? How were they adjusted and why?
In the end, I added an additional convolution layer, and simply increased the depths of the 3 convolution layers to 24, 36,and 48.  I also added dropouts to the fully connected layers.  All other hyperparameters remained the same, since I did not notice much improvement when tweaking them.

##### What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
A convolution layer works well for this application because the filters picks up features in the image.  Higher convolution layers pick up higher abstractions of these features that were picked up in the lower layers.  Then finally the fully connected layers does the classifying.  (this is what I understood from the lectures :)

The dropouts prevents overfitting by forcing the neural nets to learn alternative ways to identify the input.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![120 km/h][image4] ![30 km/h][image5] ![children crossing][image6] 
![children crossing][image7] ![turn right ahead][image8]

The 1st image has a person standing next to it.  The 4th image is probably not a real German "Children Crossing sign"; it has different shape.  The 2nd and 5th image has graffiti/stickers on their faces.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 120 km/h      		| End of speed limit (80 km/h)   				| 
| 30 km/h     			| 30 km/h 										|
| Children crossing		| Children crossing								|
| Chidlren crossing		| No entry						 				|
| Right turn ahead		| Wild animals crossing							|


The model was able to correctly guess only 2 of the 5 traffic signs, which gives an accuracy of 40%.  It is quite strange that it wasn't able to recognize the first image (120km/h) because the sign is quite clear, and there are relatively many training samples for this sign.  The 4th image is understandable, since the sign differs in shape to the training signs.  The 5th image is another mystery.  Despite graffiti on the sign's face, a human could otherwise easily recognize it.

I frankly would love to have more time to investigate this, but I'm on a time crunch at the moment.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model completely misses this sign (120km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:----------------------|:----------------------------------------------| 
| 0.23					| End of speed limit (80km/h)					|
| 0.17					| End of no passing by vehicles over 3.5 metric tons
| 0.15					| Speed limit (30km/h)
| 0.10					| Roundabout mandatory
| 0.09					| Right-of-way at the next intersection

For the 2nd image (30km/h), it guesses correctly with 100% certainty.

| Probability         	|     Prediction	        					| 
|:----------------------|:----------------------------------------------| 
| 1.00					| Speed limit (30km/h)
| 0.00					| Speed limit (20km/h)
| 0.00					| Speed limit (80km/h)
| 0.00					| End of speed limit (80km/h)
| 0.00					| Speed limit (70km/h)

For the 3rd image (Children crossing), it guesses correctly with 99% certainty.

| Probability         	|     Prediction	        					| 
|:----------------------|:----------------------------------------------| 
| 0.99					| Children crossing
| 0.00					| Turn left ahead
| 0.00					| End of no passing
| 0.00					| End of all speed and passing limits
| 0.00					| Ahead only

For the 4th image (bad Children crossing sign), it guesses incorrect but with 90% certainty!

| Probability         	|     Prediction	        					| 
|:----------------------|:----------------------------------------------| 
| 0.90					| No entry
| 0.10					| Stop
| 0.00					| Speed limit (20km/h)
| 0.00					| Traffic signals
| 0.00					| Speed limit (30km/h)

For the 5th image (Right turn ahead), it does not recognize at all.

| Probability         	|     Prediction	        					| 
|:----------------------|:----------------------------------------------| 
| 0.22					| Wild animals crossing
| 0.14					| Speed limit (60km/h)
| 0.13					| Road work
| 0.09					| Speed limit (30km/h)
| 0.07					| Double curve| .60         			

