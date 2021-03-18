# Recurrent GRF Prediction

## Contents

## Train Network
`Train_RNN.ipynb`

*Model was trained using Google Colab because they allow for use of their GPUs.* 


## Feature Generation 
1. LSTM models require data to be divided into windows instead of feeding it the whole signal at once. I have to determine the
number of windows, the size of the windows, and the number of features to be calculated in each window. I took a 
many-to-one approach, where *many* frames of the accelerometer signal were used to predict *one* frame of the vGRF signal.
This requires the number of overlapping windows to be equal to the number of frames in the vGRF signal. More specifically, 
I wanted the window of the accelerometer signal to be centered on the predicted vGRF frame. This required the signal to 
be padded with zeros at the beginning/end. Here's an illustration:
```
  Acceleration Signal

        |zero-padded|    
Frame   |-3 |-2 | -1| 0 | 1 | 2 | 3 | 4 |  
        |---|---|---|---|---|---|---|---|
Window 1 \____________*________/
Window 2     \____________+________/
Window 3         \____________#_______/
...

  vGRF Signal:        *   +   #
Frame               | 0 | 1 | 2 | 3 | 4 |  
```
Window size can have interesting effects on prediction accuracy. If the window is only 1 frame, then the model is only 
getting one frame's worth of information to make the prediction. This can lead to poor predictions if acceleration and
vGRF data don't match up consistently. However, if the window size is so large, say the entire length of stance phase,
then there's too much data to sift through just to predict one frame. I initially thought a window size < 10% stance phase
would work well, but found a window size of about 0.006 s (3% of stance phase if contact time is 0.2 s) to work best. At
1000 Hz, that is a window size of 6 frames. 

With the number of windows and size of windows determined, I had to decide on the number of features to be calculated
in each window. On the simple side of things, the features could have been 6 frames of the signal itself. On the complex
side, there are python packages that could have calculated >1,000 features from each 6-frame window. I decided to calculate
the following descriptive features from the acceleration signal for each window:
1. Mean
1. Standard Deviation
1. Range

These features were calculated for each window of both the AP axis and Vertical axis of the sacrum accelerometer data. 
In addition to these features, I included information about the condition and participant:
1. Subject Height (cm)
1. Subject Mass (kg)
1. Running Velocity (m/s)
1. Running Slope (degrees)
1. Foot strike Pattern (% Forefoot, % Midfoot, % Rearfoot Strike)

Which is a total of 13 features: (3 features for AP axis data, 3 features of vertical axis data, and 7 subject/condition 
features).

### LSTM model design
**The cool thing about LSTM models is that the input can be of variable lengths!** This means that even though this model
was trained on signals that were X seconds long, it will work on longer or shorter signals, as long as the number of 
features and size of windows are the same. This is a huge advantage over the prior use of artificial neural networks (ANN)
because ANNs require standardized input lengths. That's why you'll always see researchers normalizing data to 100% stance
phase and processing steps separately. This requires step-splitting and doesn't allow for calculation of temporal variables
like contact time, swing time, step frequency, etc. You can do all this with LSTM models though!

Below I've included an overview of the model structure. From the top down, there is one input layer `acceleration_subcond`,
the LSTM layer `lstm`, a dropout layer `dropout`, a dense layer `dense`, and then the output layer `dense_1`. Data is 
passed from the top-most layer down the the bottom-most layer, with the shape of the data changing along the way:

If we look at the input/output shapes, we can tell that 18 values are fed into the model and eventually the model 
returns a single value. This is performed for each window fed into the model and the LSTM model is cool because it uses
the output from past windows to influence the prediction of the current window. 

### Training Parameters
* Loss function: mean square error
* Optimizer: Adam algorithm with learning rate of 0.001
* Early Stopping Settings:
    - `patience=30`
    - `min_delta=0.001`
    - `restore_best_weights=True`
* Max epochs: 1000 (all subjects 60 - 160 epochs though)
* Validation split: 10%
* Batch Size = 32

### Prediction Evaluation
