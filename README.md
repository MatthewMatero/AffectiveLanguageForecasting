# AffectiveLanguageForecasting

This repository stores the data used in [Autoregressive Affective Language Forecasting: A Self-Supervised Task](). This repository is actively maintained, if you have questions feel free to email the authors or leave an issue on GH.

The data used in the paper has 2 versions, one at a weekly time resolution and one at a daily time resolution. Additionally, 2 helper data files are supplied that contain the user id and tweet id of the individual tweets before aggregating to daily/weekly values. This is done for those who want to download the raw messages and use some other form of data extraction (i.e. type of embedding).

Data is supplied in the form of CSV files in the **/data** directory. Namely, daily data can be found in **/data/daily_all_labels.csv** and weekly in **/data/weekly_all_labels.csv**. In the paper, we mostly investigate at the weekly level where we use more labels (Affect/Intensity and 6 basic emotions) as well as aggregated embeddings:


- user id
- week 
- w2v
- bert
- affect
- intensity
- anger
- disgust
- joy
- fear
- sadness
- surprise

The daily version is much smaller containing only the following:

- user id
- day id
- affect
- intensity



In the root directory of this repository there is a small setup script that goes about loading the data, chunking it into specified history lengths, and training/testing a simple model. This is supplied to give you an idea of how to go about setting up the baseline models (AR Ridge, Linear SVM, etc) as they are much easier than the various RNNs used in the paper. 
