
# Dataset Selection:
* This Model is used to predict images categorized as "Happy" and "Sad." ,the dataset is sufficiently 
not that large however it is diverse to provide a challenging problem for classification.

# Data Preprocessing:
* The data is then Cleaned and preprocessed to be fed into the model. by removing the dodgy files from the dataset,

# Loading the dataset:
* The dataset is then loaded into the model and the model is trained on the dataset, using keras preprocessing pipeline.
* I have converted the images into a numpy array and then reshaped the array to fit the model.
* I wanted to train the model in batches , so I have created a batch of images and labels to be able to process multiple images and labels simultaneously, to improve training efficiency and optimize memory usage, here I noticed that the model is trained faster in batches. 
* I wanted to quickly interpret the labels of the images in the batch and gain insights into the distribution and diversity of the classes present.
by creating a mapping of the labels to the images in the batch.
* I have then created a function to plot the images in the batch, along with the corresponding labels.

# Model Selection:
* preprocessing the data by scaling the pixel values to the range of 0-1, and then splitting the data into training and testing sets.
* I have split the data into 3 sets: training, validation, and testing. 
1. Training data is 70%
2. validation data is 20%
3. Test data is 10%

# Model Building:
* I have built a deep learning model using Keras Sequential API.
* I have used a Convolutional Neural Network (CNN) to build the model. 
  1. The first layer is a Conv2D layer with 16 filters, a 3x3 kernel size, and a stride of 1. It uses the ReLU activation function and takes input images of size 256x256 with 3 channels (RGB).
  2. Following the Conv2D layer, a MaxPooling2D layer with a pool size of 2x2 is added. This layer performs max pooling to downsample the feature maps.
  3. The next layer is a Conv2D layer with 32 filters, a 3x3 kernel size, and a stride of 1. It uses the ReLU activation function.
  4. Another Conv2D layer, a MaxPooling2D layer with a pool size of 2x2 is added.
  5. followed by Conv2D layer with 16 filters, a 3x3 kernel size, and a stride of 1 is added, followed by another MaxPooling2D layer.
* The feature maps are then flattened using the Flatten layer to convert them into a 1D vector.
  6. The Dense layer with 256 units and the ReLU activation function is added. This layer serves as a fully connected layer to learn higher-level representations from the flattened features.
  7. Finally, a Dense layer with 1 unit and the sigmoid activation function is added. This layer produces the output of the model, which is a binary classification prediction (0 or 1) using the sigmoid activation function.

# Data Augmentation:
* I have used data augmentation to increase the diversity of the training set by applying random transformations to the images in the training set.
* I have used the ImageDataGenerator class from Keras to apply data augmentation to the training set.
* I have used the rescale parameter to scale the pixel values to the range of 0-1.
* I have used the rotation_range parameter to randomly rotate the images in the training set by 30 degrees.
* I have used the width_shift_range and height_shift_range parameters to randomly shift the images horizontally and vertically by 0.2 times the width and height of the images.
* I have used the horizontal_flip parameter to randomly flip the images horizontally.
* I have used the fill_mode parameter to fill in any missing pixels that may appear after the rotation or shifting of the images.
* and much more data augmentation techniques.

# Model Training:
* i have used the fitting method to train the model on the training set. which i took several parameters:
  1. The first parameter is the training set.
  2. The second parameter is the number of epochs, which is the number of times the model will be trained on the entire training set.
  3. and a validation_data which is used to evaluate the model's performance after each epoch.
* By using the model.fit() method, the model is trained on the training set for 20 epochs, and the model's performance is evaluated on the validation set after each epoch.
* Here i see that the model's accuracy on the training set is 0.99, and the model's accuracy on the validation set is 0.98. However if i train active learning model on the same dataset, the model's accuracy on the training set is 0.99, and the model's accuracy on the validation set is 0.98. This shows that the model is overfitting on the training set. which i had problems with in the beginning where i have trained my model several times, so i have decieded to train my model once and save it, then run test on it.

# Model Evaluation:
  * Created and Utilized the appropriate evaluation metrics to evaluate the model's performance on the test set.
  * Then Processed a batch of images to predict the accuracy of the model which gave me have achieved a perfect score of 1/1 (100%) on the test set.

# Model Testing:
* i have loaded the model and tested it on untrained data to see how it performs on unseen data. and as it looks it performs well on unseen data.
* I can at any specific point in time run the saved model and load it to test it on unseen data, either from the folder shown (data_unseen) or from the web by using the url of the image.
* and the model works perfectly fine.










Note: If you want to create a virtual environment with pre-installed packages, 
you can use a requirements.txt file. First, create a virtual environment as described above. 
Then, activate the environment and run the following command:

pip install -r requirements.txt

Create a new conda environment: You can create a new conda environment by running the   
Activate the new conda environment: You can activate the new conda environment by running the 
following command in your terminal:

 conda activate cv


