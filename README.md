# Deep Protein Scoring

Spring BLUR 2018 Project

Last Updated: 4/6/18

Lead Maintainers:

- [Rafael Zamora](https://github.com/rz4) - rz4@hood.edu

- [Thomas Corcoran](https://tjosc.github.io/) - tomcorcoran@protonmail.com

This project is exploring how deep learning can be applied to protein structure
scoring. Protein structure prediction is the process of predicting the
3-dimensional conformation or structure of a protein from its unique amino-acid
sequence. In order to validate the accuracy of computationally generated structures
(often referred to as 'decoys’), scoring function are created to determine how
close a decoy is relative to its true native structure. The goal of this project
is to create and train deep learning models which predict structural accuracy
using protein structural data in an effort to improve the current state of scoring
functions.

The current methodology involves using convolutional neural networks (CNNS) to
classify decoy structures into GDT-TS score groups. GDT-TS is a measure of
similarity between a decoy and target structure, and ranges between the values
of 0.0 and 1.0. In order to train the CNNs, we need to generate 2-dimensional
representations of the 3-dimensional decoy structures which can be fed into the
networks. We will also require labeled data in the form of GDT-TS scores for
each given decoy structure. With this information, we can separate the decoy
structures into different score groups and train the networks to classify
between them.

This collection of utilities is used to aid the generation of trainable data from
large set of computationally generated structures. Scripts have been written using
MPI to run on HPC systems. Network training employs Keras/Tensorflow.

This project is still evolving rapidly and this documentation, as well as the
code contained in this repository, is subject to rapid change.

## Getting Started

In order to form a trainable dataset, input representations and GDT-TS scores must
be generated. Scripts data processing as well as documentation on how to run Scripts
can be found under [src/data_processing](src/data_processing).

Once your dataset has been generated, network training scripts can be found under
[src/network_training](src/network_training). We have currently employed a simple
binary classifer model which split the dataset along a given GDT-TS threshold,
and a cascading model which splits the dataset along a set of binned GDT-TS thresholds.
