from __future__ import print_function

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D, Input, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.image as mpimg
import imageio as im
import io
from IPython.display import clear_output
import itertools
import random
import sys

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.cm as cmaps

from tkinter import *
from tkinter import ttk
from tkinter import _setit
from tkinter import ttk, messagebox, filedialog

from mnist_data import createData, syntheticData

class plotTrainingProgress(tf.keras.callbacks.Callback):

    # Adapted from https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e (2019-03-12)

    def on_epoch_end(self,epoch,logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
                
        clear_output(wait=True)
        
        self.accFrame.ax.cla()
        self.accFrame.ax.plot(self.x,self.acc,linestyle='-',marker='o',label="Accuracy")
        self.accFrame.ax.plot(self.x,self.val_acc,linestyle='-',marker='o',label="Validation accuracy")
        self.accFrame.ax.text(0.5*self.x[-1],self.val_acc[-1]-0.2,'Validation accuracy: '+str(self.val_acc[-1]))
        self.accFrame.ax.set_title('Highest accuracy '+str(np.max(self.val_acc))+' at epoch '+str(np.argmax(self.val_acc)))
        self.accFrame.ax.legend()
        self.accFrame.ax.set_ylabel('Accuracy')
        self.accFrame.ax.set_xlabel('Epochs')
        self.accFrame.ax.set_ylim(0,1.1)
        self.accFrame.canvas.draw()

        self.lossFrame.ax.cla()
        self.lossFrame.ax.plot(self.x,self.losses,linestyle='-',marker='o',label="Loss")
        self.lossFrame.ax.plot(self.x,self.val_losses,linestyle='-',marker='o',label="Validation loss")
        min_epoch = np.argmin(self.val_losses)
        self.lossFrame.ax.set_title('Lowest loss '+str(np.round(np.min(self.val_losses),5))+' at epoch '+str(min_epoch)+' with accuracy '+str(self.val_acc[min_epoch]))
        self.lossFrame.ax.legend()
        self.lossFrame.ax.set_ylabel('Loss')
        self.lossFrame.canvas.draw()

class mnistLab(object):

    def __init__(self):

        self.batch_size = 128
        self.num_classes = 10
        self.num_epochs = 10
        self.data_shape = (28,28,1)
        self.model = Sequential()
        self.model_copy = Sequential()
        # Initialize log for training progress
        self.training_progress_plot = plotTrainingProgress()
        self.training_progress_plot.i = 0
        self.training_progress_plot.x = []
        self.training_progress_plot.losses = []
        self.training_progress_plot.val_losses = []
        self.training_progress_plot.acc = []
        self.training_progress_plot.val_acc = []
        self.training_progress_plot.logs = []

        self.cm_plot = None
        self.drawnNumber = np.zeros((28,28))
        self.data_set_num = 0
        self.layer_names = []
        self.fashion_key = ['T-shirt/Top','Trousers','Pullover','Dress','Coat','Sandal','Sneaker','Bag','Ankle boot']

        # Load data
        data = np.load('fashion_mnist_data.npz')
        (self.train_x,self.train_y,self.test_x,self.test_y) = [data[idx] for idx in data]
        self.train_x = [self.train_x]
        self.train_y = [self.train_y]
        self.test_x = [self.test_x]
        self.test_y = [self.test_y]
        self.file_names = ['mnist_data']

    def updateOptionMenus(self):

        option_list = []
        self.dataChoicePlot.oM['menu'].delete(0,END)
        self.dataSet_0.oM['menu'].delete(0,END)
        self.dataSet_1.oM['menu'].delete(0,END)
        self.dataSet_2.oM['menu'].delete(0,END)
        for option in self.file_names:
            option_list.append(option)
            self.dataChoicePlot.oM['menu'].add_command(label=option,command=_setit(self.dataChoicePlot.var,option))
            self.dataChoicePlot.var.set(option)
            self.dataSet_0.oM['menu'].add_command(label=option,command=_setit(self.dataSet_0.var,option))
            self.dataSet_0.var.set(option)
            self.dataSet_1.oM['menu'].add_command(label=option,command=_setit(self.dataSet_1.var,option))
            self.dataSet_1.var.set(option)
            self.dataSet_2.oM['menu'].add_command(label=option,command=_setit(self.dataSet_2.var,option))
            self.dataSet_2.var.set(option)
        self.dataChoicePlot.option_nums = option_list
        self.dataSet_0.option_nums = option_list
        self.dataSet_1.option_nums = option_list
        self.dataSet_2.option_nums = option_list

    def loadData(self):

        # Get path/filename
        file_name = filedialog.askopenfilename(initialdir = "/media/proboscis/Seagate 4 TB/Course_UMU",title="Select file", filetypes = (("npz-files","*.npz"),("all files","*.*")))
        # Extract filename from path
        f_name = file_name[::-1]
        f_name = f_name[4:f_name.find('/')]
        self.file_names.append(f_name[::-1])
        # Load data
        data = np.load(file_name)
        # Distribute data
        (train_x,train_y,test_x,test_y) = [data[idx] for idx in data]
        self.train_x.append(train_x)
        self.train_y.append(train_y)
        self.test_x.append(test_x)
        self.test_y.append(test_y)
        # Update all option menus
        self.updateOptionMenus()

    def generateData(self):

        if self.createDataWhichSet.var.get() == 'Augmented MNIST':
            data = createData()
            data.loadPrepMNIST()
            data.augmentDataSet(self.createDataTrainingSize.var.get(),self.createDataTestSize.var.get())            # 1st number is the factor of augmentation of training set and the 2nd for testing set
            data.save()
        if self.createDataWhichSet.var.get() == 'Augmented fashion MNIST':
            data = createData()
            data.loadPrepF_MNIST()
            data.augmentDataSet(self.createDataTrainingSize.var.get(),self.createDataTestSize.var.get())            # 1st number is the factor of augmentation of training set and the 2nd for testing set
            data.save()
        if self.createDataWhichSet.var.get() == 'Synthetic MNIST':
            data = syntheticData()
            data.generateBatch(self.createDataTrainingSize.var.get(),self.createDataTestSize.var.get())     # 1st number is the number of examples for the training set and the 2nd for testing set
            data.save()

    # Plot MNIST -------------------------------------------------------------------------------

    def plotNumbers(self,image_data):

        num_images = int(np.sqrt(self.dataChoiceNumIm.var.get()))
        image_collage = np.zeros((28*num_images,28*num_images))
        for i in np.arange(num_images):
            for j in np.arange(num_images):
                    image_collage[(27*i+i):(27*i+i+28),(27*j+j):(27*j+j+28)] = np.reshape(image_data[num_images*i+j,:,:,:],(28,28))
        self.numbersPlot.ax.imshow(image_collage)
        self.numbersPlot.canvas.draw()

    def chooseDataSetAndPlot(self):

        data_set_num = self.file_names.index(self.dataChoicePlot.var.get())
        idx = np.arange(np.shape(self.train_x[data_set_num])[0])
        np.random.shuffle(idx)
        idx = idx[0:self.dataChoiceNumIm.var.get()]
        image_data = self.train_x[data_set_num]
        self.plotNumbers(image_data)

    # Build network -----------------------------------------------------------------------------

    def getModelSummary(self):

        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        sum_str = stream.getvalue()
        stream.close()
        return sum_str

    def addLayer(self):

        tabs = ['Conv2D','Normalization','Activation','MaxPooling','Dropout','Flatten','Fully connected','Classification']
        tab = self.layerNotebook.notebookFrame.tab(self.layerNotebook.notebookFrame.select(), "text")
        if tabs[0] == tab:
            self.model.add(Conv2D(filters=self.convFilterSize.var.get(),
                            kernel_size=self.convKernelSize.var.get(),
                            strides=self.convStride.var.get(),
                            padding=self.convPadding.var.get(),
                            input_shape=self.data_shape))
        if tabs[1] == tab:
            self.model.add(BatchNormalization())
        if tabs[2] == tab:
            self.model.add(Activation('relu'))
        if tabs[3] == tab:
            self.model.add(MaxPooling2D(pool_size=(self.poolStride.var.get(),self.poolStride.var.get())))
        if tabs[4] == tab:
            self.model.add(Dropout(self.dropDropoutRate.var.get()))
        if tabs[5] == tab:
            self.model.add(Flatten())
        if tabs[6] == tab:
            self.model.add(Dense(self.fullyConnectedNum.var.get(),activation='relu'))
        if tabs[7] ==tab:
            self.model.add(Dense(self.num_classes,activation='softmax'))
        self.textFrame.delete('1.0', END)
        sum = self.getModelSummary()
        self.textFrame.insert(END,sum)
        self.getLayerNames()
        self.model_copy.add(self.model.layers[-1])

    def delLayer(self):

        self.model.pop()
        self.model_copy.pop()
        self.textFrame.delete('1.0', END)
        sum = self.getModelSummary()
        self.textFrame.insert(END,sum)
        self.getLayerNames()

    def newNetwork(self):

        self.model = Sequential()
        self.model_copy = Sequential()
        self.textFrame.delete('1.0', END)
        self.getLayerNames()
        # Reset training logs
        self.training_progress_plot.i = 0
        self.training_progress_plot.x = []
        self.training_progress_plot.losses = []
        self.training_progress_plot.val_losses = []
        self.training_progress_plot.acc = []
        self.training_progress_plot.val_acc = []
        self.training_progress_plot.logs = []

    def getLayerNames(self):

        self.layerChoice.option_nums = []
        self.layerBreak_2.option_nums = []
        self.layerChoice.oM['menu'].delete(0,END)
        self.layerBreak_2.oM['menu'].delete(0,END)
        for layer in self.model.layers[:]:
            self.layerBreak_2.oM['menu'].add_command(label=layer.name,command=_setit(self.layerBreak_2.var,layer.name))
            self.layerBreak_2.var.set(layer.name)
            self.layerBreak_2.option_nums.append(layer.name)
            if layer.name[0:3] in ['con','bat','act']:
                self.layerChoice.oM['menu'].add_command(label=layer.name,command=_setit(self.layerChoice.var,layer.name))
                self.layerChoice.var.set(layer.name)
                self.layerChoice.option_nums.append(layer.name)

    # Training network ---------------------------------------------------------------------------
    
    def trainNetwork(self):

        tabs_mode = ['Direct learning','Transfer learning']
        tab_mode = self.trainNotebook.notebookFrame.tab(self.trainNotebook.notebookFrame.select(), "text")
        if tab_mode == tabs_mode[0]: # Direct learning
            self.data_set_num = self.file_names.index(self.dataSet_0.var.get())
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
            self.model.fit(self.train_x[self.data_set_num],self.train_y[self.data_set_num],
                    batch_size=self.buildBatchSize_0.var.get(),
                    epochs=self.buildEpochsNum_0.var.get(),
                    callbacks=[self.training_progress_plot],
                    verbose=0,
                    validation_data=(self.test_x[self.data_set_num],self.test_y[self.data_set_num]))
        if tab_mode == tabs_mode[1]: # Transfer learning
            # Pretrain
            self.data_set_num = self.file_names.index(self.dataSet_1.var.get())
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
            self.model.fit(self.train_x[self.data_set_num],self.train_y[self.data_set_num],
                    batch_size=self.buildBatchSize_1.var.get(),
                    epochs=self.buildEpochsNum_1.var.get(),
                    callbacks=[self.training_progress_plot],
                    verbose=0,
                    validation_data=(self.test_x[self.data_set_num],self.test_y[self.data_set_num]))
            # Retrain
            index = self.layerBreak_2.option_nums.index(self.layerBreak_2.var.get())
            for layer in self.model.layers[:index]:
                layer.trainable = False
            if self.trainingMode_2.var.get() == 'New layers':
                for idx in np.arange(index,len(self.layer_names)+1):
                    self.model.layers[idx] = self.model_copy.layers[idx]
            self.data_set_num = self.file_names.index(self.dataSet_2.var.get())
            self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
            self.model.fit(self.train_x[self.data_set_num],self.train_y[self.data_set_num],
                    batch_size=self.buildBatchSize_2.var.get(),
                    epochs=self.buildEpochsNum_2.var.get(),
                    callbacks=[self.training_progress_plot],
                    verbose=0,
                    validation_data=(self.test_x[self.data_set_num],self.test_y[self.data_set_num]))

        self.plotConfusionMatrix(self.test_x[self.data_set_num],self.test_y[self.data_set_num])


    # Results -----------------------------------------------------------------------------------

    def plotConfusionMatrix(self,test_x,test_y):

        # Adapted from https://www.kaggle.com/grfiv4/plot-a-confusion-matrix (2019-03-12)

        def reformatCategories(y_data):

            data_length = np.shape(y_data)[0]
            new_y_data= np.zeros(data_length)
            for i in np.arange(data_length):
                new_y_data[i] = np.argmax(y_data[i,:])
            
            return new_y_data

        test_y = reformatCategories(test_y)
        self.pred_y = self.model.predict(test_x)
        self.pred_y = self.pred_y.argmax(axis=-1)
        self.confusionMatrixPlot.ax.cla()
        self.conf_mat = confusion_matrix(test_y,self.pred_y)
        #self.confusionMatrixPlot.cm_plot.set_data(self.conf_mat)
        self.confusionMatrixPlot.cm_plot = self.confusionMatrixPlot.ax.imshow(self.conf_mat,interpolation='nearest',cmap=cmaps.Blues)
        self.confusionMatrixPlot.cm_plot.set_clim(vmin=0,vmax=np.max(self.conf_mat))
        thresh = self.conf_mat.max()/2
        for i, j in itertools.product(range(self.conf_mat.shape[0]), range(self.conf_mat.shape[1])):
            self.confusionMatrixPlot.ax.text(j, i, "{:,}".format(self.conf_mat[i, j]),
                     horizontalalignment="center",
                     color="white" if self.conf_mat[i, j] > thresh else "black")
        self.confusionMatrixPlot.ax.set_xlabel('Predicted categories')
        self.confusionMatrixPlot.ax.set_ylabel('True categories')
        self.confusionMatrixPlot.ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
        self.confusionMatrixPlot.ax.set_yticks([0,1,2,3,4,5,6,7,8,9])
        self.confusionMatrixPlot.canvas.draw()

    # Drawing functions

    def redrawNumber(self):

        self.writeANumberImage.set_data(self.drawnNumber)
        self.writeANumberImage.set_clim(vmin=0,vmax=1)
        self.writeANumberFrame.canvas.draw()

    def drawANumber(self,event):

        inv = self.writeANumberFrame.ax.transData.inverted()
        x,y = inv.transform([event.x,event.y])
        if np.max(self.drawnNumber) == 0:
            self.writeANumberImage = self.writeANumberFrame.ax.imshow(self.drawnNumber,interpolation='nearest')
        self.drawnNumber[int(np.round((27-y))),int(np.round(x))] = 1
        self.redrawNumber()
        
    def clearDrawnNumber(self):

        self.drawnNumber = np.zeros((28,28))
        self.predTestVar.set('---')
        self.redrawNumber()

    def predictDrawnNumber(self):

        if self.data_set_num == 1:
            self.predTestVar.set(self.fashion_key[np.argmax(self.model.predict(self.drawnNumber[np.newaxis,:,:,np.newaxis]))])
        else:
            self.predTestVar.set(str(np.argmax(self.model.predict(self.drawnNumber[np.newaxis,:,:,np.newaxis]))))

    # Activations -------------------------------------------------------------------------------

    def plotActivations(self):

        # Adapted from https://github.com/gabrielpierobon/cnnshapes (2019-03-12)

        test_image = self.test_x[self.data_set_num][random.choice(np.arange(np.shape(self.test_x[self.data_set_num])[0]))]
        test_image = test_image[np.newaxis,:,:,:]

        layer_outputs = [layer.output for layer in self.model.layers[:]]                # Extracts the outputs of the top 12 layers
        activation_model = Model(inputs=self.model.input, outputs=layer_outputs)        # Creates a model that will return these outputs, given the model input
        activations = activation_model.predict(test_image)                              # Returns a list of five Numpy arrays: one array per layer activation
 
        index = self.layerChoice.option_nums.index(self.layerChoice.var.get())
        layer_activation = activations[index]                                           # Displays the feature maps
        if len(layer_activation.shape) < 4:
            if index == layer_activation.shape[2]:
                images_per_row = 10
            else:
                images_per_row = int(np.sqrt(layer_activation.shape[2]))
        else:
            images_per_row = int(np.sqrt(layer_activation.shape[3]))
        n_features = layer_activation.shape[-1]                                         # Number of features in the feature map
        size = layer_activation.shape[1]                                                #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row                                           # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):                                                       # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,:,:,col*images_per_row+row]
                channel_image -= channel_image.mean()                                   # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image,0,255).astype('uint8')
                display_grid[col*size:(col+1)*size,row*size:(row+1)*size] = channel_image
        self.activations.ax.imshow(display_grid, aspect='auto', cmap='viridis')
        self.activations.canvas.draw()

    def testFun(self):

        print('Test')

# GUI objects

class figureFrame(object):

    def __init__(self,frame,figure_size,side,hide_axes,color):

        self.fig = Figure(figsize=figure_size, dpi=100,tight_layout=True,facecolor=color)
        self.ax = self.fig.add_subplot(111)
        if hide_axes:
            self.ax.tick_params(right=False,left=False,top=False,bottom=False,labelleft=False,labelbottom=False)
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['bottom'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['left'].set_visible(False)
        self.canvas = FigureCanvasTkAgg(self.fig,frame)
        self.canvas.draw()
        self.canvas._tkcanvas.pack(side=side,fill=BOTH,padx=5,pady=5)

class labelWithOption(object):

    def __init__(self,frame,text,nums,init_var,row,column,width,color):

        Label(frame,text=text,background=color).grid(row=row,column=column,padx=5,pady=5)
        self.option_nums = nums
        if isinstance(nums[0],int):
            self.var = IntVar()
        elif isinstance(nums[0],str):
            self.var = StringVar()
        else:
            self.var = DoubleVar()
        self.var.set(init_var)
        self.oM = OptionMenu(frame,self.var,*self.option_nums)
        self.oM.config(width=width,bg=color)
        self.oM.grid(row=row,column=column+1,padx=5,pady=5)

class noteBook(object):
    
    def __init__(self,homeFrame,tab_text,side_of_notebook,sides_of_tabs):
        
        self.notebookFrame = ttk.Notebook(homeFrame)
        self.notebookFrame.pack(side=side_of_notebook,fill=BOTH)
        self.tab = []
        for tab_idx in np.arange(len(tab_text)):
            self.tab.append(ttk.Frame(self.notebookFrame))
            self.tab[tab_idx].pack(side=sides_of_tabs)
            self.notebookFrame.add(self.tab[tab_idx],text=tab_text[tab_idx])
			
# ==========================GUI=========================================

# --------------------------Init GUI------------------------------------

color = '#dadada'   # Use '#eaeaea' for Mac
W1 = 15
W2 = 5
root = Tk()
root.configure(background=color)
root.title('MNIST Lab')
mainFrame = Frame(root,background=color)
mainFrame.pack(fill=BOTH)
workFrame = Frame(mainFrame,background=color)
workFrame.pack(fill=BOTH,padx=5,pady=5)
bottomFrame = Frame(mainFrame,background=color)
bottomFrame.pack(side=BOTTOM,fill=BOTH,padx=5,pady=5)

MNIST = mnistLab()

# --------------------------Init notebook-------------------------------

MNIST.mainNotebook = noteBook(workFrame,['Data management','Data visualization','Network model','Training','Results','Activations'],'left','left')

# --------------------------TAB 0: Data management---------------------------------
# Load or delete data from list of data sets
dataManagementFrame = Frame(MNIST.mainNotebook.tab[0],background=color)
dataManagementFrame.pack(side=TOP,fill=X,padx=5,pady=5)
loadDataFrame = LabelFrame(dataManagementFrame,text=' Load data ',background=color)
loadDataFrame.pack(side=TOP,padx=5,pady=5)
Button(loadDataFrame,text='Load from file',width=15,background=color,highlightbackground=color,command=MNIST.loadData).grid(row=0,column=2,padx=5,pady=5)
# Generate new data
createDataFrame = LabelFrame(dataManagementFrame,text=' Create data ',background=color)
createDataFrame.pack(side=TOP,padx=5,pady=5)
MNIST.createDataWhichSet = labelWithOption(createDataFrame,'Type',['Augmented MNIST','Augmentetd fashion MNIST','Synthetic MNIST'],'Augmented MNIST',0,0,W1,color)
MNIST.createDataTrainingSize = labelWithOption(createDataFrame,'Size of training batch',[100,500,1000,5000,10000,50000,100000,200000,500000],100000,1,0,W2,color)
MNIST.createDataTestSize = labelWithOption(createDataFrame,'Size of test batch',[100,500,1000,5000,10000,50000,100000],10000,2,0,W2,color)
Button(createDataFrame,text='Generate data',width=15,background=color,highlightbackground=color,command=MNIST.generateData).grid(row=3,column=1,padx=5,pady=5)

# --------------------------TAB 1: Data visualization---------------------------------
numbersFrame = LabelFrame(MNIST.mainNotebook.tab[1],text='Numbers',background=color)
numbersFrame.pack(side=TOP,padx=5,pady=5)
MNIST.numbersPlot = figureFrame(numbersFrame,(6.5,6.5),'left',True,color)
dataChoiceFrame = LabelFrame(MNIST.mainNotebook.tab[1],text='Display data set',background=color)
dataChoiceFrame.pack(padx=5,pady=5)
MNIST.dataChoicePlot = labelWithOption(dataChoiceFrame,'Data set',['mnist_data'],'mnist_data',0,0,W1,color)
MNIST.dataChoiceNumIm = labelWithOption(dataChoiceFrame,'Number of images',[1,4,9,16,25,36,49,64,81,100,144,196,256,400],100,1,0,W1,color)
Button(dataChoiceFrame,text='Plot data set',width=15,background=color,highlightbackground=color,command=MNIST.chooseDataSetAndPlot).grid(row=2,column=0,columnspan=2,padx=5,pady=5)

# --------------------------TAB 2: Network model------------------------

layerFrame = Frame(MNIST.mainNotebook.tab[2],background=color)
layerFrame.pack(side=TOP,fill=BOTH,padx=5,pady=5)
MNIST.textFrame = Text(layerFrame,height=40,width=65)
MNIST.textFrame.pack(padx=5,pady=5)
MNIST.layerNotebook = noteBook(layerFrame,['Conv2D','Normalization','Activation','Pooling','Flatten','Dropout','Fully connected','Classification'],'top','left')
# TAB 2.0
MNIST.convFilterSize =  labelWithOption(MNIST.layerNotebook.tab[0],'Filter size',[2,4,8,16,32,64,128],16,0,0,W2,color)
MNIST.convKernelSize =  labelWithOption(MNIST.layerNotebook.tab[0],'Kernel size',[2,3,4,5,6],2,0,2,W2,color)
MNIST.convStride =  labelWithOption(MNIST.layerNotebook.tab[0],'Stride',[1,2,3],1,0,4,W2,color)
MNIST.convPadding =  labelWithOption(MNIST.layerNotebook.tab[0],'Padding',['Valid','Same','Causal'],'Same',0,6,W2,color)
# TAB 2.1
Label(MNIST.layerNotebook.tab[1],text='Batch normalization').grid(padx=5,pady=5)
# TAB 2.2
Label(MNIST.layerNotebook.tab[2],text='Rectifying linear unit').grid(padx=5,pady=5)
# TAB 2.3
MNIST.poolStride = labelWithOption(MNIST.layerNotebook.tab[3],'Stride',[1,2],2,0,0,W2,color)
# TAB 2.4
#Label(MNIST.layerNotebook.tab[4],text='Flatten').grid(padx=5,pady=5)
# TAB 2.5
MNIST.dropDropoutRate = labelWithOption(MNIST.layerNotebook.tab[5],'Dropout rate',[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],0.2,0,0,W2,color)
# TAB 2.6
MNIST.fullyConnectedNum = labelWithOption(MNIST.layerNotebook.tab[6],'Size',[16,32,64,128,256,512],128,0,0,W2,color)
# Build network
layerButtonsFrame = Frame(MNIST.mainNotebook.tab[2],background=color)
layerButtonsFrame.pack(padx=5,pady=5)
Button(layerButtonsFrame,text='New network',width=15,command=MNIST.newNetwork,background=color,highlightbackground=color).grid(column=0,padx=5,pady=5)
Button(layerButtonsFrame,text='Delete layer',width=15,background=color,highlightbackground=color,command=MNIST.delLayer).grid(row=0,column=1,padx=5,pady=5)
Button(layerButtonsFrame,text='Add layer',width=15,background=color,highlightbackground=color,command=MNIST.addLayer).grid(row=0,column=2,padx=5,pady=5)

# --------------------------TAB 3: Training-----------------------------

# Training Progress
trainingProgressFrame = LabelFrame(MNIST.mainNotebook.tab[3],text=' Training progress ',background=color)
trainingProgressFrame.pack(side=TOP,padx=5,pady=5)
MNIST.training_progress_plot.accFrame = figureFrame(trainingProgressFrame,(7,3),'top',False,color)
MNIST.training_progress_plot.lossFrame = figureFrame(trainingProgressFrame,(7,3),'bottom',False,color)

# Training settings
trainingSettingsFrame = Frame(MNIST.mainNotebook.tab[3],background=color)
trainingSettingsFrame.pack(padx=5,pady=5)
MNIST.trainNotebook = noteBook(trainingSettingsFrame,['Direct learning','Transfer learning'],'left','left')
# TAB 3.0
MNIST.buildBatchSize_0 = labelWithOption(MNIST.trainNotebook.tab[0],'Batch size',[16,32,64,128,256,512],128,0,0,W2,color)
MNIST.buildEpochsNum_0 = labelWithOption(MNIST.trainNotebook.tab[0],'Epochs',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,75,100,150,200],10,0,2,W2,color)
MNIST.dataSet_0 = labelWithOption(MNIST.trainNotebook.tab[0],'Data set',['mnist_data'],'mnist_data',0,4,W1,color)
# TAB 3.1
# Pretrain
MNIST.buildBatchSize_1 = labelWithOption(MNIST.trainNotebook.tab[1],'Batch size',[16,32,64,128,256,512],128,0,0,W2,color)
MNIST.buildEpochsNum_1 = labelWithOption(MNIST.trainNotebook.tab[1],'Epochs',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,75,100,150,200],10,0,2,W2,color)
MNIST.dataSet_1 = labelWithOption(MNIST.trainNotebook.tab[1],'Data set',['mnist_data'],'mnist_data',0,4,W1,color)
# Retrain
MNIST.buildBatchSize_2 = labelWithOption(MNIST.trainNotebook.tab[1],'Batch size',[16,32,64,128,256,512],128,1,0,W2,color)
MNIST.buildEpochsNum_2 = labelWithOption(MNIST.trainNotebook.tab[1],'Epochs',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,75,100,150,200],10,1,2,W2,color)
MNIST.dataSet_2 = labelWithOption(MNIST.trainNotebook.tab[1],'Data set',['mnist_data'],'mnist_data',1,4,W1,color)
MNIST.trainingMode_2 = labelWithOption(MNIST.trainNotebook.tab[1],'Training mode',['New layers','Reuse layers'],'New layers',0,6,W1,color)
MNIST.layerBreak_2 = labelWithOption(MNIST.trainNotebook.tab[1],'Retrain from layer',['No network'],'No network',1,6,W1,color)
# Start training
trainingStartFrame = Frame(MNIST.mainNotebook.tab[3],background=color)
trainingStartFrame.pack(padx=5,pady=5)
Button(trainingStartFrame,text='Reset network',width=15,background=color,highlightbackground=color,command=MNIST.testFun).grid(row=0,column=0,padx=5,pady=5)
Button(trainingStartFrame,text='Train',width=15,command=MNIST.trainNetwork,background=color,highlightbackground=color).grid(row=0,column=1,padx=5,pady=5)

# --------------------------TAB 4: Results------------------------------
# Confusion matrix
confusionMatrixFrame = LabelFrame(MNIST.mainNotebook.tab[4],text=' Confusion matrix ',background=color)
confusionMatrixFrame.pack(side=TOP,padx=5,pady=5)
MNIST.confusionMatrixPlot = figureFrame(confusionMatrixFrame,(5,5),'top',False,color)
MNIST.confusionMatrixPlot.cm_plot = MNIST.confusionMatrixPlot.ax.imshow(np.zeros((10,10)),interpolation='nearest',cmap=cmaps.Blues)
MNIST.confusionMatrixPlot.ax.set_xlabel('Predicted categories')
MNIST.confusionMatrixPlot.ax.set_ylabel('True categories')
MNIST.confusionMatrixPlot.ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
MNIST.confusionMatrixPlot.ax.set_yticks([0,1,2,3,4,5,6,7,8,9])
# Test network by drawing a number
testFrame = LabelFrame(MNIST.mainNotebook.tab[4],text=' Draw a figure ',background=color)
testFrame.pack(padx=5,pady=5)
MNIST.writeANumberFrame = figureFrame(testFrame,(2,2),'left',True,color)
MNIST.writeANumberFrame.canvas._tkcanvas.bind('<B1-Motion>',MNIST.drawANumber)
MNIST.writeANumberImage = MNIST.writeANumberFrame.ax.imshow(MNIST.drawnNumber,interpolation='nearest')
MNIST.redrawNumber()
predictionFrame = Frame(testFrame,background=color)
predictionFrame.pack(padx=5,pady=5)
Label(predictionFrame,text=' ',background=color).grid(row=0,column=0,padx=5,pady=5)
MNIST.predTestVar = StringVar()
MNIST.predTestVar.set('---')
Label(predictionFrame,textvariable=MNIST.predTestVar,width=12,relief=SUNKEN,bg='white').grid(row=1,column=0,padx=5,pady=5)
Button(predictionFrame,text='Predict',width=15,background=color,highlightbackground=color,command=MNIST.predictDrawnNumber).grid(row=2,column=0,padx=5,pady=5)
Button(predictionFrame,text='Clear',width=15,command=MNIST.clearDrawnNumber,background=color,highlightbackground=color).grid(row=3,column=0,padx=5,pady=5)

# --------------------------TAB 5: Activations--------------------------

activationFrame = LabelFrame(MNIST.mainNotebook.tab[5],text=' Activations ',background=color)
activationFrame.pack(side=TOP,padx=5,pady=5)
MNIST.activations = figureFrame(activationFrame,(7,7),'left',True,color)

layerChoiceFrame = Frame(MNIST.mainNotebook.tab[5],background=color)
layerChoiceFrame.pack(padx=5,pady=5)
MNIST.layerChoice = labelWithOption(layerChoiceFrame,'Layer',['No network'],'No network',0,0,W1,color)
Button(layerChoiceFrame,text='Show activations of layer',command=MNIST.plotActivations).grid(row=0,column=2,padx=5,pady=5)

# --------------------------Status bar----------------------------------

statusBar = Frame(bottomFrame,background=color)
statusBar.pack(side=BOTTOM,fill=X)
Button(statusBar,text='Quit',width=15,command=root.quit,background=color,highlightbackground=color).pack(side=RIGHT,padx=5,pady=5)

# --------------------------Mainloop------------------------------------

root.mainloop()
