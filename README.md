# Predicting continuous ground reaction forces from accelerometers during uphill and downhill running: A recurrent neural network solution
![visitors](https://visitor-badge.laobi.icu/badge?page_id=alcantarar.Recurrent_GRF_Prediction)

The final models and data supporting the published manuscript,[Predicting continuous ground reaction 
forces from accelerometers during uphill and downhill running: A recurrent neural network 
solution](https://peerj.com/articles/12752/), are archived 
via Zenodo [here](https://zenodo.org/record/5213939). `Train_LSTM.ipynb` is the notebook that
generates the model from the archived data. `Test_LSTM.ipynb` is a notebook that shows you how to use
the trained LSTM to predict GRFs from your own accelerometer data.

This repository also provides an example of how a Long Short-Term Memory Network (LSTM) can be used to 
predict ground reaction force (GRF) data from accelerometer data during running:

## Tutorial Contents
- `data/`: Contains example accelerometer data, GRF data, condition/demographic data, and LSTM model file. 
- `LSTM_Example.ipynb`: Notebook example of how to prepare data and train an LSTM to predict GRFs from accelerometer data.
If you're going to run this notebook using [Google Colab](https://colab.research.google.com/) (recommended), make sure 
you utilize their GPU Runtime Type. You will need to adjust the path to `data/` depending on how files are uploaded in
Google Colab.
- `pre_processing.py`: Some functions used in `LSTM_Example.ipynb`.

## Questions?
[Open an issue](https://github.com/alcantarar/Recurrent_GRF_Prediction/issues/new) if you have a question or if 
something is broken. 
