# Machine Learning for Computer Vision Example
This is an example on usage of model training in tensorflow.
The repository includes:
* A dummy dataset consisting from some images from the dogs vs cats data set (source: https://www.microsoft.com/en-us/download/details.aspx?id=5476)
* Functions to read images and their metadata with following forming of tensorflow datasets by means of python generators (more information tf.Data.Dataset)
* Script to train the model including several implemented callbacks (saving checkpoints, early callback and tensorboard logs)

Please bear in mind that image preprocessing steps among with some others are hard coded and might need adjusting for
particular applications.