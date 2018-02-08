import csv
import numpy as np
import tensorflow as tf


def get_data(path):
    l = []

    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        spamreader.__next__()
        for row in spamreader:
            separated = row[0].split(',')
            l.append([float(val) for val in separated])

    return np.array(l)


LEARNING_RATE = 0.01
EPOCHS = int(1e5)
EPOCH_RAPORT = int(1e3)

H_LAYER = 20

TRIAL_NAME = 'try3'
COST = 'cost_' + TRIAL_NAME

forgery_path = '/media/bartek/120887D50887B5EF/POLITECHNIKA/Magisterka/SUSigP/DataProcessed/BlindSubCorpus/FORGERY/001_f.csv'
genuine_path = '/media/bartek/120887D50887B5EF/POLITECHNIKA/Magisterka/SUSigP/DataProcessed/BlindSubCorpus/GENUINE/001_g.csv'

forg = get_data(forgery_path)
outs = len(forg) * [[0.]]
gen = get_data(genuine_path)
outs.extend(len(gen) * [[1.]])
outs = np.array(outs)

trX = np.concatenate((forg, gen))
trY = np.array(outs)

teX = trX[::2]
teY = trY[::2]
trX = np.array(trX[1::2])
trY = np.array(trY[1::2])

XOR_X = np.array(trX)
XOR_Y = np.array(trY)
INPUTS_AMOUNT = XOR_X.shape[-1]
OUTPUTS_AMOUNT = XOR_Y.shape[-1]