from os.path import join
from os import listdir
import pickle
import random
from pprint import pprint

import numpy as np
import imageio

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical

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
    model = Sequential()

    # Add layers to the model
    model.add(Conv2D(64, kernel_size = 3, activation = "relu", input_shape=(240, 592, 3)))
    model.add(Conv2D(32, kernel_size = 3, activation = "relu"))
    model.add(Flatten())
    model.add(Dense(51, activation = "softmax"))

    # Compile model and use accuracy to measure performance
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

    return model

def trainModel(model, videoFilenameList, n = 10):
    """
    trainModel trains the built model using chunks of data of size n videos

    Inputs:
     - model = model object to be trained
     - videoFilenameList = list of tuples where each tuple corresponds to a video
                           in the data set. The tuple contains the path to the video,
                           its label, and a nest tuple containing the shape
     - n = integer value for how many videos to act on at a time
    """

    # Do some initial steps to prepare for training/validating model
    chunks = [videoFilenameList[i * n:(i + 1) * n] for i in range((len(videoFilenameList) + n - 1) // n )] 

    # We're now going to iterate over our chunks
    for chunk in chunks:
        # Sum all frames in a given chunk
        chunkSum = sum([c[2][0] for c in chunk])

        # Set up numpy arrays for the video frame data and respective labels
        x = np.zeros([chunkSum, 240, 592, 3])
        y = np.zeros([chunkSum,51])
        frameNum = 0 # Initialize frame number for the chunk

        for videoPath,label,shape in chunk:
            videoData = imageio.get_reader(videoPath)

            for i,frame in enumerate(videoData):
                if frameNum >= chunkSum:
                    break
                padTotal  = 592 - frame.shape[1] # Difference in 1th dimension shape and maximum shape
                padBefore = padTotal // 2
                padAfter  = padTotal - padBefore
                paddedFrame = np.pad(frame, [(0,0), (padBefore, padAfter), (0,0)])

                x[frameNum, :, :, :] = paddedFrame
                y[frameNum, labelMap[label]] = 1
                frameNum += 1

        # Split x and y into training and validation sets
        xTrain = x[0:int(0.8*x.shape[0])]
        yTrain = y[0:int(0.8*y.shape[0])]

        xTest = x[int(0.8*x.shape[0]):]
        yTest = y[int(0.8*y.shape[0]):]

        # Train the model using cross-validation (so we don't need to explicitly do CV outside of training)
        model.fit(xTrain, yTrain, validation_data = (xTest, yTest), epochs = 2)


if __name__ == "__main__": 
    videoFilenameList, labelMap = findDataNames("/home/ubuntu/bucket/Original-Data/", "cache.pickle")
    pprint(f"Number of videos in data set: {len(videoFilenameList)}")

    # Build and train model
    model = buildModel()

    random.shuffle(videoFilenameList) # Shuffle so training in batches doesn't always have the same label
    trainModel(model, videoFilenameList, 10)




