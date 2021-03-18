# Recurrent GRF Prediction

![](README_image.png)

This repository contains an example of how a Recurrent Neural Network (RNN) can be used to predict ground reaction
force (GRF) data from accelerometer data during running. This approach is implemented in the following manuscript:

*Predicting continuous ground reaction forces from accelerometers during uphill and downhill running: A recurrent 
neural network solution"* by Alcantara et al. (In Press)

## Repository Files
- `data/`: Contains example accelerometer data, GRF data, condition/demographic data, and RNN model file. 
- `Train_RNN.ipynb`: Notebook example of how to prepare data and train an RNN to predict GRFs from accelerometer data.
If you're going to run this notebook using [Google Colab](https://colab.research.google.com/) (recommended), make sure 
you utilize their GPU Runtime Type. You will need to adjust the path to `data/` depending on how files are uploaded in
Google Colab.
- `pre_processing.py`: Some functions used in `Train_RNN.ipynb`.

## Feature Engineering 
RNNs require time series data to be divided into subsequences instead of feeding it the whole time series at once. I took a 
many-to-one approach, where *many* frames of the accelerometer signal were used to predict *one* frame of the GRF signal.
This requires the number of overlapping windows to be equal to the number of frames in the GRF signal. More specifically, 
 the window of the accelerometer signal is (almost) centered on the predicted vGRF frame. This required the signal to 
be padded at the beginning/end:
```
  Acceleration Signal

        |   padded  |    
Frame # |-3 |-2 | -1| 0 | 1 | 2 | 3 | 4 |  
        |---|---|---|---|---|---|---|---|
Window 0 \____________*________/
Window 1     \________|___+________/
Window 2         \____|___|___#_______/
...                   |   |   |
                      |   |   |
  GRF Signal:         *   +   #
Frame #             | 0 | 1 | 2 | 3 | 4 |  
```
Window size can have interesting effects on prediction accuracy. If the window is only 1 frame, then the model is only 
getting one frame's worth of information to make the prediction. This can lead to poor predictions if acceleration and
GRF data don't match up consistently. However, if the window size is so large, say the entire length of stance phase,
then there's too much data to sift through just to predict one frame. I found that a window size of 6 frames (12 ms at 
500 Hz) worked well for my running data, but try different window sizes and see what works for you. Window size can be 
modified in `Train_RNN.ipynb` by adjusting the `window_size` variable.

In addition to window size, I  had to decide on the number of features to be calculated from each window of accelerometer
data. On the simple side of things, the features could have been 6 frames of the signal itself. On the more complex
side, there are python packages that could have calculated >1,000 features from each 6-frame window. I started with many
 features and took a stepwise approach to keeping only 3 features calculated for each window. This reduced my computational
 cost compared to just using the 6 frames in the window and maintained prediction accuracy:
1. Mean
1. Standard Deviation
1. Range

These features were calculated for each window of both the Anteroposterior (AP) axis and Vertical axis of the sacrum 
accelerometer data, a total of 6 features. In addition to these features, I included information about the condition and
runner:
1. Runner Height (cm)
1. Runner Mass (kg)
1. Running Velocity (m/s)
1. Running Slope (degrees)
1. Foot strike Pattern (% of steps in trial classified as Forefoot, Midfoot,or Rearfoot strike)

This is a total of 13 features: (3 features from the AP accelerometer data, 3 features from the vertical accelerometer 
data, and 7 discrete features).

## Model Design
The cool thing about RNNs is that the input can be of variable lengths! This means that even though this network
was trained on signals that were X seconds long, it will work on longer or shorter signals, as long as the number of 
features and size of windows are the same. This is advantageous over artificial neural networks (ANNs) that predict the 
whole time series at once. The input signal length must be consistent, so that's why it's common to normalize accelerometer
and GRF data to 100% stance phase and process each stance phase separately. This approach requires stance phase 
identification and doesn't allow for calculation of temporal variables like contact time, swing time, step frequency, 
etc. You can do all this with RNNs, though! RNNs can predict 0.6 seconds of accelerometer data as easily as 6 hours of 
accelerometer data because predictions are made iteratively (recurrently) based on the windowed time series data. The 
RNN in `Train_RNN.ipynb` predicts a single frame of GRF data from 13 features that represent the runner, condition, and 
summary statistics for 6 corresponding frames of accelerometer data.

## Questions?
[Open an issue](https://github.com/alcantarar/Recurrent_GRF_Prediction/issues/new) if you have a question or if 
something is broken. 
