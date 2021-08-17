# Predicting continuous ground reaction forces from accelerometers during uphill and downhill running: A recurrent neural network solution
![visitors](https://visitor-badge.laobi.icu/badge?page_id=alcantarar.Recurrent_GRF_Prediction)

The final models and data supporting [Predicting continuous ground reaction 
forces from accelerometers during uphill and downhill running: A recurrent neural network 
solution](https://www.biorxiv.org/content/10.1101/2021.03.17.435901) are archived 
via Zenodo [here](https://zenodo.org/record/4995574). `Train_LSTM.ipynb` is the notebook that
generates the model from the archived data.

This repository also provides an example of how a Recurrent Neural Network (RNN) can be used to 
predict ground reaction force (GRF) data from accelerometer data during running:

## Tutorial Contents
- `data/`: Contains example accelerometer data, GRF data, condition/demographic data, and RNN model file. 
- `LSTM_Example.ipynb`: Notebook example of how to prepare data and train an RNN to predict GRFs from accelerometer data.
If you're going to run this notebook using [Google Colab](https://colab.research.google.com/) (recommended), make sure 
you utilize their GPU Runtime Type. You will need to adjust the path to `data/` depending on how files are uploaded in
Google Colab.
- `pre_processing.py`: Some functions used in `LSTM_Example.ipynb`.

## RNN vs MLP
The benefit of using RNNs is that the input can be of variable lengths! Even though the network in this repository
was trained on signals that were *n* seconds long, it will work on longer or shorter signals, as long as the number of 
features and size of windows are the same. This is advantageous over multilayer perceptrons (MLPs) that are used to 
predict the whole time series at once. The input signal length must be consistent during training and inference, so 
that's why it's common to normalize accelerometer and GRF data to 100% stance phase and process each stance phase 
separately when using MLPs. This approach requires preliminary stance phase identification and doesn't allow for 
calculation of temporal variables like contact time, swing time, step frequency, etc. 

## Questions?
[Open an issue](https://github.com/alcantarar/Recurrent_GRF_Prediction/issues/new) if you have a question or if 
something is broken. 
