# AudioSurfer_Game_AI
A 'bot' composed of two AI models that can see and play the video game called 'Audio Surfer'. 

## Purpose
This project was created as part of an independent study course that had the goal of being able to create neural networks that could play a given video game of my choice using computer vision that was granted only by screenshots and keyboard inputs. 

## Requirements
Requires the following python3 modules:
* pandas
* numpy
* cv2
* mss
* keyboard
* tensorflow
  * keras
  * object_detection.utils (this will likely require protobuf if you haven't set up the models module)

## How It Works
The project uses two neural network models (AI) and scripts that extract, transform, and load data between the source (the computer peripherals) and the models. 

To summarize what happens: one model determines where the objects we need to know about when making our choices are located on the screen, then the second model will make a prediction of what arrow-key to press based on where these objects are on screen. For any given arrangement of objects on the screen, it will make a prediction based on the gameplay it was trained on and mimics that gameplay (i.e. the AI will mimic all the choices i made during the gameplay i trained it on, so it's a copycat from watching me play). The predicted arrow-key is then pushed for you. This can be repeated constantly, allowing the bot to play the game on its own.

This bot's two AI models is: 
* an object detection model (ODM) that takes input from the screen then outputs object locations
* a long short-term memory model (LTSMM) that takes in a feature vector and outputs a predicted feature vector

The full loop of data goes as follows:
1. capture an image from the screen
2. feed the image into the ODM and get the resulting object locations
3. filter and transform the objects location data into a feature vector
   * the feature vector's composition can be found in the section below (Information and Notes for Geeks)
4. capture the keyboard arrow-key input and add it to the current feature vector
5. feed the feature vector to the LTSMM and get the resulting predicted feature vector
6. extract the predicted arrow-key to be pushed from the predicted feature vector
7. give the predicted arrow-key as a keyboard input to the computer on our behalf
8. repeat

## Information and Notes for Geeks
The game rules of `Audio Surfer` is very simple. We have three lanes that extend vertically with the our single `ship` occupying a single lane at the very bottom. The `ship` can choose to be in the left, middle, or right lane by pushing the `left arrow key`, `no arrow key`, or the `right arrow key` respectively. `block` objects can appear at the top and slide down a given lane, which we want collide our ship with for points. `spike` objects can appear at the top and slide down a given lane as well, but we want to avoid touching them with our ship. multiple `block` and `spike` objects can appear on the screen at the same time, but they won't overlap and must be aligned with the center of any given lane.

the feature vector is an array composed of 4 features: the location of a `block`, `spike`, `ship`, and the arrow-key pressed. 
*  the current 4 features is represented as 10 values:
1. the x coordinate of the lowest detected `block`
2. the y coordinate of the lowest detected `block`
3. the x coordinate of the lowest detected `spike`
4. the y coordinate of the lowest detected `spike`
5. (5.,6.,7.) the location of the `ship` represented as 'one hot encoding' for `left lane`, `middle lane`, and `right lane`
   * [a one hot array is used to have a binary representation for all possible catagories](https://en.wikipedia.org/wiki/One-hot)
   * the purpose of this is to make predictions easier and more accurate by having 'off' and 'on' values for each catagory, which [can be further explained here](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
8. (8.,9.,10.) the key input represented as 'one hot encoding' for `left arrow-key`, `no key`, and `right arrow key`

These values were chosen as both blocks and spikes can appear on screen and the vertical order they appear can affect what choice we make, along with the x coordinate determining what lane they may be in (however we cannot one-hot encode this easily due to parallax effecting x coordinates when the ship changes lane). The ship can be one-hot encodded as it tends to be at the same x coordinates for each lane and uneffected by any parallax and will always be at the very bottom (so no consideration is needed for the y coordinate).

The ODM has currently an overall accuracy of >80%, with typically >95% confidence scores when it recognizes the actual object (i.e. it has very high confidence for true-positive matches, but infrequently has false-negatives for an expected `block` or `spike` object, and very rarely a false-negative for the `ship` object.) False positives with high confidence are very, very rare.

The LTSMM has a training accuracy of ~85% and testing accuracy of ~80%. This inaccuracy typically resulted from predicting to enter or leave the expected lane slightly sooner than expected or slightly later than expected, but this perfectly acceptable as it is in the expected/desired lane when it mattered and had negligable impact on gameplay results.

details about the respective models can be found documented within the python files of the project.

## Personal Take on the Project
This project has taught me quite a lot about python, computer environments, as well as the various parts in configuring neural networks and training them. Overall, I feel the models I trained were a success with my object detection model having a high accuracy rating without too many anomalies that I had faced in previous model attempts. My LSTM model also feels rather successful as the accuracy of my second version ended up around 83% accuracy, which is rather impeccable as after I compared the prediction, they were notably similar where it mattered with a few variations (mostly being inaccurate due to a slight delay or quickness in changing lanes compared to when I would've, but this is overall good as it followed the major trend and didn't predict weird or unbelievably.) Notably, only the test data accuracy consistently converged on a '66.7% accuracy' for all configurations, which jumped to 83% accuracy after I shifted the expected keys column up by one row before comparing to the predicted keys column. 

The hardest parts faced during this project mostly was figuring out how to set up my environments in order to get tensorflow and its modules to work, but after enough dedicated effort and research I was able to work through it successfully. The longest part was probably labeling all the images for training the object detection model and determining what my dependencies were for creating a useful feature vector that would give me accurate results. 

### Going Foward
Going forward, I hope to make further optimizations that may give better results. I plan on adding a script that will implement the full loop as well as adding a setup script for installing the dependencies that my scripts require.
