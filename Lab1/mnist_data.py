import matplotlib.pyplot as pl
from matplotlib.patches import Rectangle
import numpy as np
import random
import datetime
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist
from tkinter import ttk, messagebox, filedialog

class createData(object):

    def __init__(self):

        self.num_pixels_row = 28
        self.num_pixels_cols = 28

    def loadMNIST(self):

        # Load data

        (self.train_x,self.train_y),(self.test_x,self.test_y) = mnist.load_data()

    def loadFashionMNIST(self):

        # Load data

        (self.train_x,self.train_y),(self.test_x,self.test_y) = fashion_mnist.load_data()

    def prepareData(self):

        # Reshape data

        self.train_x = self.train_x.reshape(self.train_x.shape[0], self.num_pixels_row, self.num_pixels_cols, 1)
        self.test_x = self.test_x.reshape(self.test_x.shape[0], self.num_pixels_row, self.num_pixels_cols, 1)
        self.input_shape = (self.num_pixels_row, self.num_pixels_cols, 1)

        self.train_x = self.train_x.astype('float32')
        self.test_x = self.test_x.astype('float32')
        self.train_x /= 255
        self.test_x /= 255

        # Convert class vectors to binary class matrices

        self.train_y = to_categorical(self.train_y,10)
        self.test_y = to_categorical(self.test_y,10)

    def saveMNIST(self):

        np.savez('mnist_data.npz',self.train_x,self.train_y,self.test_x,self.test_y)

    def saveFashionMNIST(self):

        np.savez('fashion_mnist_data.npz',self.train_x,self.train_y,self.test_x,self.test_y)

    def loadPrepMNIST(self):

        # load MNIST
        data = np.load('mnist_data.npz')
        (self.train_x,self.train_y,self.test_x,self.test_y) = [data[idx] for idx in data]

    def loadPrepF_MNIST(self):

        # load fashion MNIST
        data = np.load('fashion_mnist_data.npz')
        (self.train_x,self.train_y,self.test_x,self.test_y) = [data[idx] for idx in data]

    def distortImage(self,data_x):

        def permute(steps):
            nums = np.arange(28)
            new_nums = list(nums)
            if steps > 0:
                #new_nums[0:steps] = nums[(len(nums)-steps):]
                new_nums[steps:] = nums[0:(len(nums)-steps)]
            if steps < 0:
                new_nums[0:(len(nums)+steps)] = nums[-steps:]
                #new_nums[(len(nums)+steps):] = nums[0:-steps]
            
            return new_nums

        row_translation = permute(random.choice([-4,-3,-2,-1,0,1,2,3,4]))
        column_translation = permute(random.choice([-4,-3,-2,-1,0,1,2,3,4]))
        image_x = data_x[row_translation,:,:]                                                                       # Perform translation for rows
        image_x = image_x[:,column_translation,:] + 0.2*np.random.rand(28,28,1) - 0.2 + random.uniform(-0.2,0.2)    # Perform translation for columns; Add noise and basline shift
        image_x = (image_x - np.min(image_x))/np.max(image_x)                                                       # Normalization to [0,1]

        return image_x

    def augmentDataSet(self,augment_factor_train,augment_factor_test):
        
        def augmentData(self,data_x,data_y,augment_factor):
        
            data_length = np.shape(data_x)[0]
            augmented_data_x = np.zeros((augment_factor*data_length,28,28,1))
            augmented_data_y = np.zeros((augment_factor*data_length,10))
            for iteration in np.arange(augment_factor):
                for idx in np.arange(data_length):
                    augmented_data_x[iteration*data_length+idx,:,:,:] = self.distortImage(data_x[idx,:,:,:])
                    augmented_data_y[iteration*data_length+idx,:] = data_y[idx]

            return augmented_data_x, augmented_data_y
        
        self.augmented_train_x, self.augmented_train_y = augmentData(self,self.train_x,self.train_y,augment_factor_train)
        self.augmented_test_x, self.augmented_test_y = augmentData(self,self.test_x,self.test_y,augment_factor_test)

    def saveAugData(self):

        file_name = filedialog.asksaveasfilename(initialdir = "/media/proboscis/Seagate 4 TB/Course_UMU",title="Select file", filetypes = (("npz-files","*.npz"),("all files","*.*")))
        np.savez(file_name,self.augmented_train_x,self.augmented_train_y,self.augmented_test_x,self.augmented_test_y)

class syntheticData(object):

    def __init__(self):

        self.start_time = time.time()
        self.fonts = ['serif','sans-serif','monospace']
        self.style = ['normal','italic','oblique']
        self.width = [ 'ultralight','light','normal','regular','book','medium','roman','semibold','demibold','demi','bold','heavy','extra bold','black']

    def generateNumber(self):
    
        # Create figure and configure

        fig,ax = pl.subplots()
        fig.set_size_inches(0.28,0.28)
        fig.patch.set_facecolor('black')
        
        # Randomize font parameters

        num = random.randint(0,9)
        rotation = random.randint(-20,20)
        x = random.uniform(0,0.3)
        y = random.uniform(0,0.3)
        font_size = random.randint(10,18)
        font_type = random.randint(0,2)
        font_style = random.randint(0,2)
        font_width = random.randint(0,13)
        
        # Plot number

        ax.add_patch(Rectangle((-5,-5),35,35,alpha=1,color='black'))
        ax.text(x,y,str(num),fontsize=font_size,fontname=self.fonts[font_type],fontweight=self.width[font_width],fontstyle=self.style[font_style],rotation=rotation,backgroundcolor='black',color='white')
        fig.canvas.draw()

        # Create array of image

        image = np.array(fig.canvas.renderer._renderer)
        image = image[:,:,0]/255
        image = image[:,:,np.newaxis]
        
        # Close figure to clean up
        
        pl.close(fig)

        # Return image and category

        return image, num

    def generateBatch(self,size_train,size_test):

        image = syntheticData()
        self.synthetic_train_x = np.zeros((size_train,28,28,1))
        self.synthetic_train_y = np.zeros(size_train)
        self.synthetic_test_x = np.zeros((size_test,28,28,1))
        self.synthetic_test_y = np.zeros(size_test)
        for i in np.arange(size_train):
            self.synthetic_train_x[i,:,:,:],self.synthetic_train_y[i] = image.generateNumber()
            if i%1000 == 0:
                passed_time = time.time() - self.start_time
                passed_time = str(datetime.timedelta(seconds=round(passed_time)))
                print('1000 examples generated after',passed_time)
        for i in np.arange(size_test):
            self.synthetic_test_x[i,:,:,:],self.synthetic_test_y[i] = image.generateNumber()
            if i%1000 == 0:
                passed_time = time.time() - self.start_time
                passed_time = str(datetime.timedelta(seconds=round(passed_time)))
                print('1000 examples generated after',passed_time)
        self.synthetic_train_y = to_categorical(self.synthetic_train_y,10)
        self.synthetic_test_y = to_categorical(self.synthetic_test_y,10)

    def save(self):

        file_name = filedialog.asksaveasfilename(initialdir = "/media/proboscis/Seagate 4 TB/Course_UMU",title="Select file", filetypes = (("npz-files","*.npz"),("all files","*.*")))
        np.savez(file_name,self.synthetic_train_x,self.synthetic_train_y,self.synthetic_test_x,self.synthetic_test_y)


# Loads MNIST/Fashion MNIST from TensorFlow, reshapes and saves to a file ('mnist_data.npz' or fashion_mnist_data.npz) in current folder

# mnist = createData()
# mnist.loadMNIST()
# mnist.prepareData()
# mnist.saveMNIST()

# fashion_mnist = createData()
# fashion_mnist.loadFashionMNIST()
# fashion_mnist.prepareData()
# fashion_mnist.saveFashionMNIST()

# Creates augmented data from original MNIST set and saves to a file ('augmented_data.npz') in current folder
# ====================== Creating large data sets may take long time =========================================

# augmented_mnist.loadPrepMNIST()
# augmented_mnist.augmentDataSet(100000,10000)              # 1st number is the factor of augmentation of training set and the 2nd for testing set
# augmented_mnist.saveAugData()

# augmented_fashion_mnist.loadPrepF_MNIST()
# augmented_fashion_mnist.augmentDataSet(100000,10000)              # 1st number is the factor of augmentation of training set and the 2nd for testing set
# augmented_fashion_mnist.saveAugData()

# Creates synthetic data from fonts and saves to a file ('synthetic_data.npz') in current folder
# ====================== Creating large data sets may take ***VERY*** long time =========================================

# synthetic_data = syntheticData()
# synthetic_data.generateBatch(100000,10000)                # 1st number is the number of examples for the training set and the 2nd for testing set
# synthetic_data.save()