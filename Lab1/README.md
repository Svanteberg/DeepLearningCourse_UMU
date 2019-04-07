# Laboration 1 - Using convolutional networks for the classification of the Fashion MNIST data set.

## Introduction

Since I'm not an experienced programmer and Python is new to me, I took the opportunity to practice and made a GUI for developing convolutional networks. It is implemented in Tkinter. Since developing a flexible GUI; capable of every possible variation of data processing, network architectures and training schemes; is an overwelming task, naturally this GUI will have its limitations.

The GUI is run by typing

```
python mnist_project.py
```

in the terminal. An auxillary script

```
mnist_data.py
```

is necessary for data processing.

At the moment, the GUI can work with both the MNIST and the Fashion MNIST data sets (and any image data set that has been preprocessed correct and have the right format; 28x28 pixels, 1 channel and 10 categories). In the GUI there are different tabs for each step in processing, building, training and evaluating networks:

  1. Loading data and creating augmented data (for the MNIST data set there is also the possiblity of generating synthetic data).
  2. Visualizing the data
  3. Building a network
  4. Training the network, with the option of performing transfer learning.
  5. Evaluation of the network by confusion matrix and testing prediction capability of manually drawn figures.
  6. Visualization of the activations.

Below, is a walk through of the GUI with an example. After that, a couple of variations of network architecture, parameter choices and training strategies are shown.

### Preparing data before starting using the GUI

In this project, the data set is used as provided with a split in a training and a test set. The test set could be split into a validation set, used during training, and a test set, used to test the final network. But, these two sets probably would not be disjoint, i.e., there could be correlations between them since both would have examples written by the same persons. In a way, since theses data sets have been used so extensively, some form of feedback of what parameters work best to get good result on the test set may exist on a collective level.

The MNIST and the Fashion MNIST data sets have to be preprocessed and saved in the right way before starting using the GUI. This can be done with the class "createData" in "mnist_data.py". For the Fashion MNIST data set:

  ```
  fashion_mnist = createData()
  fashion_mnist.loadFashionMNIST()
  fashion_mnist.prepareData()
  fashion_mnist.saveFashionMNIST()
  ```
 
which saves the data in the file "fashion_mnist_data.npz" in the current folder. And for the MNIST data set:

  ```
  mnist = createData()
  mnist.loadMNIST()
  mnist.prepareData()
  mnist.saveMNIST()
  ```
  
which saves the data in the file "mnist_data.npz" in the current folder. It is also possible to augment the data:

  ```
  fashion_mnist.augmentDataSet(number_of_examples_for_training,number_of_examples_for_testing)
  fashion_mnist.saveAugData()
  ```
  
and

  ```
  mnist.augmentDataSet(number_of_examples_for_training,number_of_examples_for_testing)
  mnist.saveAugData()
  ```
where the data is saved as an npz-file with a name specified through a file dialog window. 


## The GUI

### Tab 0 - Loading data and creating augmented/synthetic data

The Fashion MNIST data set will load automatically when starting the application. Other saved data sets can be loaded manually by clicking on the "Load from file" button and choosing data file from the file dialog.

It is also possible to generate augmented data from either the MNIST or the Fashion MNIST data set and choosing the number of examples of the training and testing batch. For the MNIST, there is the alternative of generating synthetic data, which is created from a couple of different fonts that are given a further variation by translations, rotations and changing font size. (However, there are only a few different fonts and these will be in both training and test data. It is easy to achieve a high accuracy but since both training and test data will be quite similar, it may be hard to know if there is overfitting.)

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Data_management.png)


### Tab 1 - Visualizing the data

Any of the loaded data sets can be visualized, with a couple of choices for the number of displayed images and the images being choosen randomly.

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Data_visualization.png)


### Tab 2 - Building the network

The network is always initialized as:

```
self.model = Sequential()
```

The network can then be built by adding layers using the "Add layer" button. The layer type of the active tab will be added. Some of the layers will have choices for different parameters. The last layer can be deleted by pressing the "Delete layer" button. Pressing the "New network" button will delete the whole network and re-initialize the model.

#### Convolutional layer (choose number of filters, kernel size, stride length, padding):

```
self.model.add(Conv2D(filters=self.convFilterSize.var.get(),
                   kernel_size=self.convKernelSize.var.get(),
                   strides=self.convStride.var.get(),
                   padding=self.convPadding.var.get(),
                   input_shape=self.data_shape))
```

#### Batch normalization layer (no parameter choices):

```
self.model.add(BatchNormalization())
```

#### Activation layer (ReLU; no parameter choices):

```
self.model.add(Activation('relu'))
```

Max pooling layer (choose stride length):

```
self.model.add(MaxPooling2D(pool_size=(self.poolStride.var.get(),self.poolStride.var.get())))
```

#### Drop out layer (choose dropout rate):

```
self.model.add(Dropout(self.dropDropoutRate.var.get()))
```

#### Flattening layer (no parameter choices):

```
self.model.add(Flatten())
```

#### Fully connected layer (dense; choise number of nodes, activation preset to ReLU):

```
self.model.add(Dense(self.fullyConnectedNum.var.get(),activation='relu'))
```

#### Classification layer (no parameter choices):

```
self.model.add(Dense(self.num_classes,activation='softmax'))
```

Choosing a layer combination that is incompatible will not result in any error message in the GUI, but may be seen in the terminal. If trying to proceed and train the network anyway, it will not work. If an incompatible layer has been choosen, it may be that the whole model needs to be re-initialized (by pressing the "New network" button).

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Network_model.png)

The model above has four convolutional layers. Each convolutional layer is followed by a batch normalization layer, to prevent values from diverging to much, which in turn is followed by an activation layer, to introduce nonlinearities. All this is followed by a flattening layer to reshape the output to match a fully connected layer. The final part of the model consists of a fully connected layer with a softmax activation which will give a probability for each category as the final output from the network.

### Tab 3 - Training the network

When the network has been built, training can be started by clicking on the "Train" button. This will also result in the compilation of the model before starting training: 

```
self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
```

If the button is clicked again after finnishing training, the training will continue from were it left off. So to train from start, i.e., with reset weights, the "Reset network" button must be pressed.

An addition, not shown in the figure below, is a checkbox for using the `ImageDataGenerator` for data augmentation.

Training can be performed directly, using the choosen data set to train an untrained network. Or, the network can be pretrained on one data set and then retrained on another data set. When retraining, the training can be applied to the whole network or from a certain layer. It is possible to reuse the layers or to replace them with new untrained layers.

As mentioned above, a network can be continued to train after training for the specified number of epochs, which is accomplished by simply pressing the "Train" button again. So it is possible to train for a restricted number of epochs at a time. It is actually also possible to continue the training with another data set.

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Training.png)

In this example, overfitting is seen, slightly in the accuracy plot but more obvious in the loss plot. An early stop at epoch two (epoch zero being the first) may be suitable.

### Tab 4 - Network evaluation

When training is done, this tab will automatically display the confusion matrix, which will show a more detailed view of the performance compared to the overall accuracy.

There is also the possibility of drawing a figure and see if the network can predict what it is. This feature was initialy developed for the MNIST data set (it was at first assumed that the "fashion MNIST" just was some fancy looking numbers...). However, it is possible to use it with the fashion MNIST, but your drawing skills may be put to test.

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Results.png)

In this example, when testing the network with drawings, some of the categories that were predicted were dominating, for example "ankle boot". Of course, the artistic talant of the person drawing will affect the outcome. But when drawing a figure, each pixel that has been "drawn" gets the maximum value (1). Looking att the visualization of the Fashion MNIST images, it is seen the pixels generally have lower values (greener). Introducing a function for setting the level of the drawing may give better predictions.

### Tab 5 - Visualizing the activations

In the GUI, when displaying the activations of a layer, the image is choosen at random. So, for this illustration this will be a bit unpedagogical. (It would of course have been more pedagogical to have the same image. The reason for randomizing in the GUI is to use different images since how the features appear may depend on the image and it is an easy solution for avoiding using the same image all the time).

In this example there are four convolutional layers. For the first layer, it is possible to somewhat understand what it does and it seem to mainly detect edges of different orientations. As one progress toward higher layers the features become more abstract and harder to decipher.

#### First convolutional layer

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Activation_0.png)

#### Second convolutional layer

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Activation_1.png)

#### Third convolutional layer

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Activation_2.png)

#### Fourth convolutional layer

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Activation_3.png)

## Examples

### Number of convolutional layers

### Batchnormalization

### Fully connected layers and dropout

Network with two fully connected layers (512 and 128 nodes). To the left with no dropout layer and to the right with a dropout layer set to 50 percent in between.

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/FC_2L.png)

Network with four fully connected layers (all have 128 nodes). To the left with no dropout layer and to the right with dropout layers set to 50 percent in between.

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/FC_4L.png)

The effect of the dropout layers on overfitting tendensies is obvious, but it doesn't always give a better validation accuracy.

### Using augmented data

Here the `ImageDataGenerator` is used for data augmentation during training. The network starts with three blocks each consisting of a convolutional, an activation and a max pooling layer. After a flattening layer, this is followed by three blocks each consisting of a dropout layer set to 0.7 and a fully connected layer with 1024 nodes. The network is finnished with a fully connected classification layer.

```
self.model.add(Conv2D(filters=128,kernel_size=3,strides=1,padding='same',input_shape=(28,28,1))
self.model.add(Activation('relu'))
self.model.add(MaxPooling2D(pool_size=(2,2)))

self.model.add(Conv2D(filters=256,kernel_size=3,strides=1,padding='same')
self.model.add(Activation('relu'))
self.model.add(MaxPooling2D(pool_size=(2,2)))

self.model.add(Conv2D(filters=512,kernel_size=3,strides=1,padding='same')
self.model.add(Activation('relu'))
self.model.add(MaxPooling2D(pool_size=(2,2)))

self.model.add(Flatten())
self.model.add(Dropout(0.7))
self.model.add(Dense(1024,activation='relu'))

self.model.add(Dropout(0.7))
self.model.add(Dense(1024,activation='relu'))

self.model.add(Dropout(0.7))
self.model.add(Dense(1024,activation='relu'))

self.model.add(Dense(self.num_classes,activation='softmax'))
```

The `ImageDataGenerator` is set as:

```
rotation_range=12,
shear_range = 7,
width_shift_range=0.075,
height_shift_range=0.075,
horizontal_flip=True)
```

![](https://github.com/Svanteberg/DeepLearningCourse_UMU/blob/master/Lab1/Images/Aug_CNN_FCx3.png)
