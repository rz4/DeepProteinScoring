# Deep Protein Scoring

Spring SULI 2018 Project

Last Updated: 4/23/18

Lead Maintainers:

- [Rafael Zamora](https://github.com/rz4) - rz4@hood.edu

- [Thomas Corcoran](https://tjosc.github.io/) - tomcorcoran@protonmail.com

This project is exploring how deep learning can be applied to protein structure scoring.
Protein structure prediction techniques has been used to predict the native 3-dimensional
conformation or structure of a protein from its unique amino-acid sequence, and
these methods are capable of generating a large quantity of possible structures. In
order to validate the accuracy of these computationally generated structures
(often referred to as 'decoys’), scoring function are created to determine how
accurate a decoy is relative to its true native structure. The goal of this project
is to define and train deep learning models which can predict accuracy
from a decoys structure in an effort to improve the current state of scoring
functions which will improve decoy selection and evaluation.

Our current methodology involves using convolutional neural networks (CNNs) to
classify decoy structures into Global Distance Test Total Score (GDT-TS) thresholds.
GDT-TS is a measure of similarity between a decoy and target structure, and
ranges between the values of 0.0 and 1.0. We have created a pipeline which
generates 2-dimensional feature maps of the 3-dimensional decoy structures, and
provides tools to calculate GDT-TS on HPC systems. With our feature maps as
input data and GDT-TS scores as labels, we can divide decoy structures into
different score groups and train the networks to classify between them.

This collection of utilities is used to aid the generation of trainable data from
large set of computationally generated structures provided through. Scripts have been written using
MPI to run on HPC systems. Network training employs Keras/Tensorflow.

This project is still evolving rapidly and this documentation, as well as the
code contained in this repository, is subject to rapid change.

## Getting Started

In order to form a trainable dataset, input representations and GDT-TS scores must
be generated. Scripts for data processing as well as documentation on how to run scripts
can be found under [src/data_processing](src/data_processing).

Once your dataset has been generated, network training scripts can be found under
[src/network_training](src/network_training). We have currently employed a simple
binary classifer model which split the dataset along a given GDT-TS threshold,
and a cascading model which splits the dataset along a set of binned GDT-TS thresholds.
