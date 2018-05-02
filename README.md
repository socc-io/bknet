# BKnet

#### Authors

* Baek Seohyeon (becxer)
* Kim Youngjin (smilu97)

## This repository contains

This repository contains python source code using tensorflow, written for AI-hackathon hold on NAVER

## Detail of contents

### Problems

There was two problems which participants solve

#### KIN - Comparing similarity between two sentences

#### MOVIE - Predicting rating of review sentence

### File structure

![depgraph](statics/depgraph.png)

### Detail

`main_xxx.py` create Dataset instance in `dataset.py`, split minibatch data, and run session with graph that starts with layer in `core_layer.py`.

`dataset.py` contains iterable class that read raw string data from file, and preprocess data (char -> phoneme indices, char -> syllable indices, word -> wordvec)