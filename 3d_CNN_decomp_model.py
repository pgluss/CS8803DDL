import tensorflow as tf
import numpy as np


from os.path import join
from os import listdir
import pickle
import random
from pprint import pprint

import imageio


class SCL(tf.keras.Model):

    def __init__(self, name=None):
        # Initialize layers needed for Spatial Convolution Module
        super(SCL, self).__init__(name=name)
        self.c1 = tf.keras.layers.Conv2D(96, 7, 2)
        self.relu = tf.keras.layers.ReLU()
        self.norm = tf.nn.local_response_normalization
        self.p1 = tf.keras.layers.MaxPool2D(3, 2)
        self.c2 = tf.keras.layers.Conv2D(256, 5, 2)
        self.p2 = tf.keras.layers.MaxPool2D(3, 2)
        self.c3 = tf.keras.layers.Conv2D(512, 3, 1)
        self.c4 = tf.keras.layers.Conv2D(512, 3, 1)
        

    def call(self, inputs):
        # First Convolution
        temp = self.c1(inputs)
        # ReLU and local response normalization
        temp2 = self.norm(self.relu(temp), 5, 2, 5 * 10**-4, 0.75)
        # Pooling
        o1 = self.p1(temp2)
        # Conv, ReLU, norm, pooling
        o2 = self.p2(self.norm(self.relu(self.c2(o1)),   5, 2, 5 * 10**-4, 0.75))
        # Convolution3, notice no pooling or relu
        o3 = self.c3(o2)
        #Convolution4
        return self.c4(o3)

class TCL(tf.keras.Model):
    
    def __init__(self, name= None):
        def __init__(self, name= None):
        # Set up layers for Temporal Convolutions
        super(TCL, self).__init__(name=name)
        self.c1 = tf.keras.layers.Conv2D(128, 3, 1)
        self.p1 = tf.keras.layers.MaxPool2D(3,3)
        
        self.ct1 = tf.keras.layers.Conv1D(32, 3, 1)
        self.relu = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(0.5)

        # Comment out either the first or second time convolution to stop 
        # warning when only using one of the layers
        
        #self.ct2 = tf.keras.layers.Conv1D(32, 5, 1)

        self.fc1 = tf.keras.layers.Dense(4096)
        self.fc2 = tf.keras.layers.Dense(2048)


    def call(self, inputs):
        # Pre layers in temporal path

        o1 = self.c1(inputs)
        o2 = self.p1(o1)
        
        # Do reshaping according to the paper so time convolution is explicit

        shape = tf.shape(o2)
        t, x, y, f = shape[0], shape[1], shape[2], shape[3]
        # May need to change 128 for new video set
        o2 = tf.reshape(o2, [t, x * y, 128])
        
        #shape: BS, 16, 32
        #to1 = self.dropout(self.relu(self.ct1(o2)))
        #shape: BS, 14, 32
        to2 = self.dropout(self.relu(self.ct2(o2)))
        
        # Need to adjust the 2 value when switching layers, may change when changing 
        # sampling values
        t_final =  tf.reshape(to2, [tf.shape(to2)[0],2 * 32])

        l1 = self.fc1(t_final)
        return self.fc2(l1)

class SCL_extra(tf.keras.Model):
    
    def __init__(self, name=None):
        # Initialize extra Spatial Layers

        super(SCL_extra, self).__init__(name=name)
        self.c1 = tf.keras.layers.Conv2D(128, 3, 1)
        self.p1 = tf.keras.layers.MaxPool2D(3,3)
        self.fc1 = tf.keras.layers.Dense(4096, input_shape = (2304,))
        self.fc2 = tf.keras.layers.Dense(2048, input_shape = (4096,))
        
    def call(self, inputs):
        # Conv and pool

        o1 = self.p1(self.c1(inputs))
        # Reshape for linear layer, may change if you change sampling values
        o1 = tf.reshape(o1, [tf.shape(o1)[0], 768])
        o2 = self.fc1(o1)
        return self.fc2(o2)
    
class Final(tf.keras.Model):
    
    def __init__(self, name=None):
        super(Final, self).__init__(name=name)
        num_classes = 51
        self.fc1 = tf.keras.layers.Dense(2048, input_shape = (2048, ))
        self.fc2 = tf.keras.layers.Dense(num_classes, input_shape = (2048,))
        self.softmax = tf.keras.layers.Softmax(axis=1)
        
    def call(self, inputs1, inputs2 = None):
        # Maybe should use tf.concate, but its killing my kernel on the reshape
        inputs = inputs1 + inputs2
        # Linear Layer and softmax
        o1 = self.fc1(inputs)
        o2 = self.softmax(self.fc2(o1))
        return o2

class Overall(tf.keras.Model):
    def __init__(self, name=None):
        super(Overall, self).__init__(name=name)
        self.final = Final(name='Final')
        self.scl_extra = SCL_extra(name='SCL_extra')
        self.tcl = TCL(name='TCL')
        self.scl = SCL(name='SCL')
        
    def call(self, inputs):
        # Call layers in order
        # Now due to sampling we need to keep track of two different outputs
        # V_clip is fed into the extra SCL
        # V_clip diff is fed into the temporal layer
        out_scl1, out_scl2 = self.scl(inputs)

        out_tcl = self.tcl(out_scl1)
        out_extra = self.scl_extra(out_scl2)
        # Feed both into final layer
        out_final = self.final(out_tcl, out_extra)
        return out_final

"""
Previous Method: Classify Actions on Static Video Frames
"""

def findDataNames(searchDir, cacheFile = "cache.pickle"):
    """
    findDataNames generates a list of the full paths to all video files
    for usage in later steps. 
    Inputs:
     - searchDir = path to the directory containing the video data
     - cacheFile = name of the pickle file to save compiled data to and to 
                   read compiled data from
    Outputs:
     - videoFilenameList = list of tuples where each tuple corresponds to a video
                           in the data set. The tuple contains the path to the video,
                           its label, and a nest tuple containing the shape
     - labelMap = a mapping between the string representation of the label name and an
                  integer representation of that label
    We cache data to a pickle file because it is slow to iterate through
    every file on every run. Remembering this information makes it easier 
    to tune and test the code as it increases running speed.
    """

    # First we want to get the video name, path, and label
    try:
        # Load our cache file if it exists
        videoFilenameList, labelMap = pickle.load(open(cacheFile, "rb"))
    except (OSError, IOError) as e:
        # If cache file doesn't exist, then we will build up up the data explicity
       videoFilenameList = []
       labelMap = {}
       counter = 0
        print("Error")
        videoFilenameList = None
        labelMap = None
       for label in listdir(searchDir):
           # Add label and corresponding integer to the label dictionary
           labelMap[label] = counter
           counter += 1

           labelDir = join(searchDir, label, label)
           for videoFilename in listdir(labelDir):
               videoPath = join(labelDir, videoFilename)
               print(videoPath)

               videoData = imageio.get_reader(videoPath)
               for i,frame in enumerate(videoData):
                    # We're counting the number of frames and due to issues with ffmpeg, we're required
                    # to iterate through the frames instead of just getting the length
                   pass

               videoFilenameList.append((videoPath, label, (i, frame.shape[0], frame.shape[1], frame.shape[2])))

       pickle.dump((videoFilenameList, labelMap), open(cacheFile, "wb"))
    
    return videoFilenameList, labelMap

def buildModel():
    """
    buildModel sets up a convolutional 2D network using a reLu activation function
    Outputs:
     - model = model object to be used later for training and classification
    """
    # We need to incrementally train the model so we'll set it up before preparing the data
    test_model = Overall(name='Overall')
    my_optimizer = tf.keras.optimizers.Adam(0.0005, 0.9)
    test_model.compile(optimizer=my_optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return test_model

def trainModel(model, videoFilenameList, n = 1):
    """
    trainModel trains the built model using chunks of data of size n videos
    Inputs:
     - model = model object to be trained
     - videoFilenameList = list of tuples where each tuple corresponds to a video
                           in the data set. The tuple contains the path to the video,
                           its label, and a nest tuple containing the shape
     - n = integer value for how many videos to act on at a time
    """
# Switch n to 1 so we can do the video clip sampling per video

    # Do some initial steps to prepare for training/validating model
    chunks = [videoFilenameList[i * n:(i + 1) * n] for i in range((len(videoFilenameList) + n - 1) // n )] 

    # We're now going to iterate over our chunks
    xTest = []
    yTest = []
    for chunk in chunks:
        # Sum all frames in a given chunk
        chunkSum = sum([c[2][0] for c in chunk])

        # Set up numpy arrays for the video frame data and respective labels
        x = np.zeros([chunkSum, 240, 592, 3])
        y = np.zeros(chunkSum)
        # How much we want to reduce the dimension of our clips
        div = 4/3
        v = np.zeros([chunkSum, int(240/div), int(592/div), 3])
        frameNum = 0 # Initialize frame number for the chunk
        dt = 9
        num_done = -1
        for videoPath,label,shape in chunk:
            videoData = imageio.get_reader(videoPath)
            # Initialize values in order to to do sampling
            # lt in paper, how many to sample temporally
            lt = 5
            st = 0
            # tst is total times st has gone thorugh all lt values
            tst = 0
            # How videos you have done
            num_done += 1
            for i,frame in enumerate(videoData):
                if frameNum >= chunkSum:
                    break
                padTotal  = 592 - frame.shape[1] # Difference in 1th dimension shape and maximum shape
                padBefore = padTotal // 2
                padAfter  = padTotal - padBefore
                paddedFrame = np.pad(frame, [(0,0), (padBefore, padAfter), (0,0)])

                x[frameNum, :, :, :] = paddedFrame
                y[frameNum]= labelMap[label]
                frameNum += 1
                # Dimensions to reduce
                lx, ly = int(np.floor(paddedFrame.shape[0]/div)), int(np.floor(paddedFrame.shape[1]/div))
                # Variables need to be incremented/reset 
                # Basically st = st mod 5
                if st == 5:
                    st = 0
                    tst += 1
                    lt -= 1
                # Once lt hits 0 stop sampling
                if lt != 0:
                    v[num_done * lt * st + tst * st + st, :, :, :] = paddedFrame[0:lx, 0:ly, :]
                    st += 1
            # Get m random indices to sample
            m = np.random.choice(np.shape(x)[0], 1 + x.shape[0]//2) 
            # Take diff of v vector and sample it m times
            vdiff = np.abs(np.diff(x[min(m):max(m) + dt, 0:lx, 0:ly,:], dt, axis = 2))
            vdiff_final = vdiff[m-min(m), :, :, :]
            # Sample v, y as well
            v_final = v[m, :, :, :]
            y_samp = y[m]

        # Need two training and test sets now for v_clip diff and v_clip
        vTrain1 = vdiff_final[0:int(0.8*vdiff_final.shape[0])]
        vTrain2 = v_final[0:int(0.8*v_final.shape[0])]
        yTrain = y_samp[0:int(0.8*y_samp.shape[0])]
        vTest1 = vdiff_final[int(0.8*vdiff_final.shape[0]):]
        vTest2 = v_final[int(0.8*v_final.shape[0]):]
        yTest = y_samp[int(0.8*y_samp.shape[0]):]
         

 #       xTest += [x[int(0.8*x.shape[0]):]]
#        yTest += [y[int(0.8*y.shape[0]):]]

        # Train the model using cross-validation (so we don't need to explicitly do CV outside of training)
        model.fit([vTrain1, vTrain2], yTrain, batch_size = 32, epochs = 2)
        model.evaluate([vTest1, vTest2], yTest)
  #  print("Testing")
   # for loop_ind in range(len(xTest)):
    #    model.evaluate(xTest[loop_ind], yTest[loop_ind])

if __name__ == "__main__": 
    videoFilenameList, labelMap = findDataNames("/home/ubuntu/video-dataset/","videoFilenameList.pickle")
    pprint(f"Number of videos in data set: {len(videoFilenameList)}")

    # Build and train model
    model = buildModel()

    random.shuffle(videoFilenameList) # Shuffle so training in batches doesn't always have the same label
    trainModel(model, videoFilenameList, 10)

