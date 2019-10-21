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

# First we want to get the video name, path, and label
try:
    videoFilenameList, labelMap = pickle.load(open("videoFilenameList.pickle", "rb"))
except (OSError, IOError) as e:
    videoFilenameList = []
    labelMap = {}
    counter = 0

    videoSetDir = "/home/ubuntu/bucket/Original-Data/"
    for label in listdir(videoSetDir):
        # Add label and corresponding integer to the label dictionary
        labelMap[label] = counter
        counter += 1

        labelDir = join(videoSetDir, label, label)
        for videoFilename in listdir(labelDir):
            videoPath = join(labelDir, videoFilename)
            print(videoPath)

            videoData = imageio.get_reader(videoPath)
            for i,frame in enumerate(videoData):
                pass

            videoFilenameList.append((videoPath, label, (i, frame.shape[0], frame.shape[1], frame.shape[2])))

    pickle.dump((videoFilenameList, labelMap), open("videoFilenameList.pickle", "wb"))

pprint(f"Number of videos in data set: {len(videoFilenameList)}")

# We need to incrementally train the model so we'll set it up before preparing the data
model = Sequential()

# Add layers to the model
model.add(Conv2D(64, kernel_size = 3, activation = "relu", input_shape=(240, 592, 3)))
model.add(Conv2D(32, kernel_size = 3, activation = "relu"))
model.add(Flatten())
model.add(Dense(51, activation = "softmax"))

# Compile model and use accuracy to measure performance
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# Do some initial steps to prepare for training/validating model
random.shuffle(videoFilenameList) # Shuffle so training in batches doesn't always have the same label

n = 10 # Number of videos to act on at a time
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

    print(yTest.shape)


    # Train the model
    model.fit(xTrain, yTrain, validation_data = (xTest, yTest), epochs = 2)

    print(videoPath)







