# Laboration 1 - Using convolutional network for classification of the Fashion MNIST data set.

Since I'm not an experienced programmer and Python is new to me, I took the opportunity to practice and made a GUI for developing convolutional networks. It is implemented in Tkinter. Since developing a flexible GUI; capable of every possible variation of data processing, network architectures and training schemes; is an overwelming task, naturally this GUI will have its limitations.

At the moment, the GUI can work with both the MNIST and the Fashion MNIST data sets (and any image data set that has been preprocessed correct and have the right format; 28x28 pixels, 1 channel and 10 categories). In the GUI there are different tabs for each step in processing, building, training and evaluating networks:

  1. Loading data and creating augmented data (for the MNIST data set there is also the possiblity of generating synthetic data).
  2. Visualizing the data
  3. Building a network
  4. Training the network, with the option of performing transfer learning.
  5. Evaluation of the network by confusion matrix and testing prediction capability of manually drawn figures.
  6. Visualization of the activations.

Below is an illustration of the GUI, using an example of training a network.

### Preparing data before starting using the GUI

The MNIST and the Fashion MNIST data sets have to be preprocessed and saved in the right way before starting using the GUI. This can be done with the class "createData" in "mnist_data.py". For the Fashion MNIST data set:

  ```
  fashion_mnist = createData()
  fashion_mnist.loadFashionMNIST()
  fashion_mnist.prepareData()
  fashion_mnist.saveFashionMNIST()
  ```
 
and for the MNIST data set:

  ```
  mnist = createData()
  mnist.loadMNIST()
  mnist.prepareData()
  mnist.saveMNIST()
  ```
  
It is also possible to augment the data:

  ```
  fashion_mnist.augmentDataSet(number_of_examples_for_training,number_of_examples_for_testing)
  fashion_mnist.saveAugData()
  ```
  
and

  ```
  mnist.augmentDataSet(number_of_examples_for_training,number_of_examples_for_testing)
  mnist.saveAugData()
  ```
  
### Loading data and creating augmented/synthetic data

The MNIST data set will load automatically when starting the application. Other saved data sets can be loaded manually by clicking on the "Load from file" button and choosing data file from the file dialog.

It is also possible to generate augmented data from either the MNIST or the Fashion MNIST data set and choosing the number of examples of the training and testing batch. For the MNIST, there is the alternative of generating synthetic data, which is created from a couple of different fonts that are given a further variation by translations, rotations and changing font size.

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Data_management.png)


### Visualizing the data

Any of the loaded data sets can be visualized, with a couple of choices for number of displayed images and the images choosen randomly.

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Data_visualization.png)


### Building the network

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Network_model.png)


### Training the network

When the network has been built, training can be started by clicking on the "Train" button. If the button is clicked again after finnishing training, the training will continue from were it left off. So to train from start, i.e., with reset weights, the "Reset network" button must be pressed.

Training can be performed directly, using the choosen data set to train an untrained network. Or the network can be pretrained on one data set and then retrained on another data set. When retraining, the training can be applied to the whole network or from a certain layer. It is possible to reuse the layers or to replace them with new untrained layers.

As mentioned above, a network can be continued to train after training for the specified number of epochs, which is accomplished by simply pressing the "Train" button again. So it is possible to train for a number of epochs at a time. It is actually also possible to continue the training with another data set.

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Training.png)

In this example, overfitting is seen, slightly in the accuracy plot but more obvious in the loss plot. An early stop at epoch two may be suitable.

### Network evaluation

When training is done, this tab will automatically display the confusion matrix, which will show a more detailed view of the performance compared to the overall accuracy.

There is also the possibility of drawing a figure and see if the network can predict what it is. This feature was initialy developed for the MNIST data set (I misunderstood the assignment and thought we could choose data set and that the "fashion MNIST" just was some fancy looking numbers...). However, it is possible to use it with the fashion MNIST, but your drawing skills may be put to test.

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Results.png)

In this example, when testing the network with drawings, some of the categories that were predicted were dominating, for example "ankle boot". Of course, the artistic talant of the person drawing will affect the outcome. But when drawing a figure, each pixel that has been "drawn" gets the maximum value (1). Looking att the visualization of the Fashion MNIST images, it is seen the pixels generally have lower values (greener). Introducing a function for setting the level of the drawing may give better predictions.

### Visualizing the activations

In the GUI, when displaying the activations of a layer, the image is choosen at random. So, for this illustration this will be a bit unpedagogical (it would have been more pedagogical to have the same image, but the reason for randomizing in the GUI is to use different images since how the features appear may depend on the image).

In this example there are four convolutional layers. For the first layer, it is possible to somewhat understand what it does and it seem to mainly detect edges of different orientations. As one progress toward higher layers the features become more abstract and harder to decipher.

#### First convolutional layer

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Activation_0.png)

#### Second convolutional layer

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Activation_1.png)

#### Third convolutional layer

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Activation_2.png)

#### Fourth convolutional layer

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Activation_3.png)
