# Predicting continuous ground reaction forces from accelerometers during uphill and downhill running: A recurrent neural network solution
![visitors](https://visitor-badge.laobi.icu/badge?page_id=alcantarar.Recurrent_GRF_Prediction)

This repository supports [Predicting continuous ground reaction 
forces from accelerometers during uphill and downhill running: A recurrent neural network 
solution](https://peerj.com/articles/12752/).

The final models and data supporting the published manuscript are archived [here](https://zenodo.org/record/5224624). 

## Contents

`Train_LSTM.ipynb` is a notebook that generates the model from the archived data. 

`Test_LSTM.ipynb` is a notebook that shows you how to use the trained LSTM to predict GRFs from your own accelerometer data.

`Example_LSTM.ipynb` is a notebook that provides a tutorial of how a Long Short-Term Memory Network (LSTM) can be used to 
predict ground reaction force (GRF) data from accelerometer data during running.

`pre_processing.py` contains helper functions used in `LSTM_Example.ipynb` and `Test_LSTM.ipynb`.

`data/` Contains example accelerometer data, GRF data, condition/demographic data, and LSTM model file. Supports `Test_LSTM.ipynb` and `Example_LSTM.ipynb`. 

If you're going to train an LSTM model using [Google Colab](https://colab.research.google.com/) (recommended), make sure 
you utilize their GPU Runtime Type. You will need to adjust the path to `data/` depending on how files are uploaded in
Google Colab.

## Questions?
[Open an issue](https://github.com/alcantarar/Recurrent_GRF_Prediction/issues/new) if you have a question or if 
something is broken. You can also email me at the address listed in the associated publication.
