# order of operations
## Organize Data
1. `pickle_data.py`
    - Crop data to 10,000 frames per trial (common denominator across trials/subs)
1. `store_all_sub_data.py`
    - writes csv files for each variable where each row is a trial defined in Sub_Info.csv. 10,000 columns
## Train Model
1. `LSTM_calgary_updownhill.ipynb`

Model was trained using Google Colab because they allow for use of their GPUs. 

### Data Cleaning
1. Fill in missing `AGE` or `HEIGHT` data with group median for 2 subjects.
1. Remove walking conditions (S1C29 and S1C27)
1. Remove 3 conditions for Subject 11 with only noise or confusing signal patterns (S1C18, S1C19, S2C5)
1. Swap `Sacrum_AP` with `Sacrum_Vert` for Subject 11. I have reason to believe these sensors were switched during
data collection.

### Data Processing
1. Downsample data from 2000 Hz to 1000 Hz. Reduced computational cost and 1000 Hz is more common.
1. Normalize vGRF to body weight (BW)
1. Filter Sacrum (AP and Vertical axes) data
    - 4th order zero-lag low-pass butterworth (corrected per Research Methods in Biomechanics (2e) pg 288)
    - 20 Hz cutoff was determined from FFT

### Prepare data for LSTM
1. Split Train/Test by doing Leave-One-Subject-Out. Computationally expensive, but a good choice for smaller datasets.
1. Concatenate training AP/Vert Sacrum signals and fit to a scaler (`sklearn.preprocessing.StandardScaler()`).
 This standardization method removes the mean and scales data so that it has a mean of 0 and SD of 1. More info can
 be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). Scaler is
 then used to transform the training and testing acceleration data.
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
1. Maximum Value
1. Median
1. Range
1. Minimum Value
1. Sum of signal values

These features were calculated for each window of both the AP axis and Vertical axis of the sacrum accelerometer data. 
In addition to these features, I included information about the condition and participant:
1. Subject Height (cm)
1. Subject Mass (kg)
1. Running Velocity (m/s)
1. Running Slope (degrees)

Which is a total of 18 features: (7 features for AP axis data, 7 features of vertical axis data, and 4 subject/condition 
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

![](model.png)

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
After model is trained on n-1 subjects, it's used to predict the vGRF data of the remaining subject. RMSE is calculated 
for the entire trial, even though the error in the first couple frames is sometimes a bit wonky due to padding.

I'm thinking about splitting steps and calculating RMSE for each step (stance + following aerial) as past work has done.


  