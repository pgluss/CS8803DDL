# Action Classification in Video Data
This project is for the Georgia Institute of Technology's CS 8803 DDL class. The course covers studying and developing techniques for large-scale video analytics with deep learning and covers deep learning architectures with a focus on developing end-to-end modeling systems.

## Introduction
### Background 
Action detection in videos is a rising topic in the machine learning world as application opportunities continue to grow. Some applications include:
- Autonomous systems capable of monitoring crowds of people and sending alerts if suspicious activity is detected.
- Sports analytics systems for studying games played to find patterns in play style. Being able to ingest hours of recorded game data would allows coaches to drastically improve their players' training routines.

### Approach
The planned approach for this project is to implement a 3D CNN using a 2D and 1D CNN. This is an intuitive approach that allows us to take advantage of the signiﬁcant prior work done in 2D image classiﬁcation using CNNs while expanding it to ﬁt our problem space. A 3D CNN can be simulated with a traditional 2D ﬁlter for spatial data and a 1D ﬁlter takes this data in to capture the temporal element.

Decomposing a 3D CNN into a 2D CNN (spacial) and 1D CNN (temporal) with multiple kernels reduces the high cost of a full 3D CNN and reduces the number of features.

### Data
The data for this project is from the Human Motion Database from Brown University. It contains 6849 video clips divided into 51 action classes and each class contains at least 100 clips.

## Usage
This code currently uses the following Python packages:

- pickle: pickle is being used to cache video data so that it does not have to be reloaded each time the code is run. This speeds up tuning and testing by reducing the overall run time.
- imageio: this is being used to read in video data and convert to still frames and is an interface with ffmpeg.
- keras: framework for building deep learning models.

The code currently expects the video data to be in a nested folder structure.

Data_Folder --> Label_Folder --> Label_Folder --> Actual_Video

## Task List
### Majors 
[x] Set up environment for video processing on AWS leveraging EC2 and S3
[x] Complete first method of classification -- converting videos into still frames and classifying the images rather than the whole video itself
[x] Understand and be able to articulate the strengths and weaknesses of a static frame classification approach
[x] Design architecture for video classification with spatial and temportal dimensions.
[] Complete implementation of a spatio-temporal classifier using CNNs
[] Complete implementation of a spatio-temporal classifier using a CNN and RNN
[] Integrate model into the EVA video querying engine
[] Test system on new videos introduced to the data set

### Minors
[] Massage video file structure to reduce nested folders
[] Generalize for variable video sizes (currently this is hard-coded)