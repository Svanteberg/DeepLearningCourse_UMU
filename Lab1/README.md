# Laboration 1 - Using convolutional network for classification of the Fashion MNIST data set.

Since I'm not an experienced programmer and Python is new to me, I took the oppurtunity to practice and made a GUI for developing convolutional networks. It is implemented in Tkinter. Since developing a flexible GUI; capable of every possible variation of data processing, network architectures and training schemes; is an overwelming task, naturally this GUI will have it's limitations.

In the GUI there are different tabs for each step in processing, building, training and evaluating networks:

  1. Loading data and creating augmented data (for the MNIST data set there is also the possiblity of generating synthetic data).
  2. Visualizing the data
  3. Building a network
  4. Training the network, with the option of performing transfer learning.
  5. Evaluation of the network by confusion matrix and testing prediction capability of manually drawn figures.
  6. Visualization of the activations.

### Loading data and creating augmented/synthetic data

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Data_management.png)


### Visualizing the data

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Data_visualization.png)


### Building the network

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Network_model.png)


### Training the network

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Training.png)


### Network evaluation

When training is done, this tab will automatically display the confusion matrix which will show a more detailed view of the performance compared to the overall accuracy.

There is also the possibility to draw a figure and see if the network can predict what it is. This feature was initialy developed for the MNIST data set (I misunderstood the assignment and thought we could choose data set and that the "fashion MNIST" just was some fancy looking letters...). However, it is possible to use it with the fashion MNIST, but your drawing skills may be put to test.

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Results.png)


### Visualizing the activations

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Activation_0.png)

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Activation_1.png)

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Activation_3.png)
