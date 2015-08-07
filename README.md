# kaggle's Otto Group Product Classification Challenge

## Introduction

This is my code for [kaggle's Product Classification Challenge][1]. I had a [write-up][2] about the solution in my blog. The final model uses an ensemble of two levels. 30 runs can get 0.4192 on private LB (top 5%). Training data for level 1 and level 2 was roughly split into 1:1 and one can fine tune this to get a better result.

This competition attracted me because:

* Lots of competitors and lots of solutions in the forum to learn from.
* Feature engineering was quite limited, thus I could focus on trying to use different models.
* A good chance to use model ensembling.

## Dependencies

* CUDA Toolkit 7.0
* Python 2.7.6
    * Lasagne 0.1.dev
    * nolearn 0.5
    * numpy 1.8.2
    * pandas 0.13.1
    * scikit-learn 0.16.1
    * scipy 0.13.3
    * Theano 0.7.0
    * xgboost 0.4

## Run

* Put data into the `data` dir
* run `python ensembler.py`

[1]: https://www.kaggle.com/c/otto-group-product-classification-challenge
[2]: http://jianghao.org/blog/20150803/otto-challenge.html