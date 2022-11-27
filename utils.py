from tensorflow.contrib.keras import preprocessing
from collections import defaultdict
import itertools
import os, pdb
import tensorflow as tf
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pad_sequences = preprocessing.sequence.pad_sequences

def read_data(event_train_file, event_test_file, time_train_file, time_test_file, dist_train_file, dist_test_file, pad=True):

    with open(event_train_file, 'r') as in_file:
        eventTrain = [[int(y) for y in x.strip().split()] for x in in_file]
    with open(event_test_file, 'r') as in_file:
        eventTest = [[int(y) for y in x.strip().split()] for x in in_file]

    with open(time_train_file, 'r') as in_file:
        timeTrain = [[float(y) for y in x.strip().split()] for x in in_file]
    with open(time_test_file, 'r') as in_file:
        timeTest = [[float(y) for y in x.strip().split()] for x in in_file]

    with open(dist_train_file, 'r') as in_file:
        distTrain = [[float(y) for y in x.strip().split()] for x in in_file]
    with open(dist_test_file, 'r') as in_file:
        distTest = [[float(y) for y in x.strip().split()] for x in in_file]

    assert len(timeTrain) == len(eventTrain)
    assert len(eventTest) == len(timeTest)

    unique_samples = set()

    for x in eventTrain + eventTest:
        unique_samples = unique_samples.union(x)

    maxTime = max(itertools.chain((max(x) for x in timeTrain), (max(x) for x in timeTest)))
    minTime = min(itertools.chain((min(x) for x in timeTrain), (min(x) for x in timeTest)))

    maxDist = max(itertools.chain((max(x) for x in distTrain), (max(x) for x in distTest)))
    minDist = min(itertools.chain((min(x) for x in distTrain), (min(x) for x in distTest)))

    eventTrainIn = [x[:-1] for x in eventTrain]
    eventTrainOut = [x[1:] for x in eventTrain]

    timeTrainIn = [[(y - minTime) / (maxTime - minTime) for y in x[:-1]] for x in timeTrain] 
    timeTrainOut = [[(y - minTime) / (maxTime - minTime) for y in x[1:]] for x in timeTrain]

    distTrainIn = [[(y - minDist) / (maxDist - minDist) for y in x[:-1]] for x in distTrain] 
    distTrainOut = [[(y - minDist) / (maxDist - minDist) for y in x[1:]] for x in distTrain]

    if pad:
        train_event_in_seq = pad_sequences(eventTrainIn, padding='post')
        train_event_out_seq = pad_sequences(eventTrainOut, padding='post')
        train_time_in_seq = pad_sequences(timeTrainIn, dtype=float, padding='post')
        train_time_out_seq = pad_sequences(timeTrainOut, dtype=float, padding='post')
        train_dist_in_seq = pad_sequences(distTrainIn, dtype=float, padding='post')
        train_dist_out_seq = pad_sequences(distTrainOut, dtype=float, padding='post')

    else:
        train_event_in_seq = eventTrainIn
        train_event_out_seq = eventTrainOut
        train_time_in_seq = timeTrainIn
        train_time_out_seq = timeTrainOut
        train_dist_in_seq = distTrainIn
        train_dist_out_seq = distTrainOut

    eventTestIn = [x[:-1] for x in eventTest]
    eventTestOut = [x[1:] for x in eventTest]
    timeTestIn = [[(y - minTime) / (maxTime - minTime) for y in x[:-1]] for x in timeTest]
    timeTestOut = [[(y - minTime) / (maxTime - minTime) for y in x[1:]] for x in timeTest]
    distTestIn = [[(y - minDist) / (maxDist - minDist) for y in x[:-1]] for x in distTest]
    distTestOut = [[(y - minDist) / (maxDist - minDist) for y in x[1:]] for x in distTest]

    if pad:
        test_event_in_seq = pad_sequences(eventTestIn, padding='post')
        test_event_out_seq = pad_sequences(eventTestOut, padding='post')
        test_time_in_seq = pad_sequences(timeTestIn, dtype=float, padding='post')
        test_time_out_seq = pad_sequences(timeTestOut, dtype=float, padding='post')
        test_dist_in_seq = pad_sequences(distTestIn, dtype=float, padding='post')
        test_dist_out_seq = pad_sequences(distTestOut, dtype=float, padding='post')
    else:
        test_event_in_seq = eventTestIn
        test_event_out_seq = eventTestOut
        test_time_in_seq = timeTestIn
        test_time_out_seq = timeTestOut
        test_dist_in_seq = distTestIn
        test_dist_out_seq = distTestOut

    return {
        'train_event_in_seq': train_event_in_seq,
        'train_event_out_seq': train_event_out_seq,
        'train_time_in_seq': train_time_in_seq,
        'train_time_out_seq': train_time_out_seq,
        'train_dist_in_seq': train_dist_in_seq,
        'train_dist_out_seq': train_dist_out_seq,
        'test_event_in_seq': test_event_in_seq,
        'test_event_out_seq': test_event_out_seq,
        'test_time_in_seq': test_time_in_seq,
        'test_time_out_seq': test_time_out_seq,
        'test_dist_in_seq': test_dist_in_seq,
        'test_dist_out_seq': test_dist_out_seq,
        'num_categories': len(unique_samples)
    }


def MAE(time_preds, time_true, events_out):
    seq_limit = time_preds.shape[1]
    clipped_time_true = time_true[:, :seq_limit]
    clipped_events_out = events_out[:, :seq_limit]
    is_finite = np.isfinite(time_preds) & (clipped_events_out > 0)
    return np.mean(np.abs(time_preds - clipped_time_true)[is_finite]), np.sum(is_finite)

def ACC(event_preds, event_true):
    clipped_event_true = event_true[:, :event_preds.shape[1]]
    is_valid = clipped_event_true > 0
    highest_prob_ev = event_preds.argmax(axis=-1) + 1

    return np.sum((highest_prob_ev == clipped_event_true)[is_valid]) / np.sum(is_valid)
