import numpy as np
import csv
import os
from math import isnan, isfinite
from features.ReadData import read_signature, SignatureParams


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def extract_features(files_urls, signatory_nr='', in_dir_path=None, out_dir_path=None, get_features=None, draw_plots=None, write_data=None):
    np.seterr(divide='ignore', invalid='ignore')
    signatures = []

    for file_url in files_urls:
        raw_measures = read_signature(in_dir_path + file_url)
        signatures.append(SignatureParams(raw_measures))

    os.makedirs(os.path.dirname(out_dir_path), exist_ok=True)
    with open(out_dir_path + signatory_nr + '.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['signatureDuration', 'penDownRatio', 'aRatio', 'cursiviness'])
        for signature in signatures:
            spamwriter.writerow(signature.features_container)


def calculate_means(mins, maxes):
    means = []
    fullArea = 0
    for i in range(mins):

        if isnan(mins[i].x) or isnan(maxes[i].x):
            continue

        if isfinite(maxes[i].y - mins[i].y) and maxes[i].y != mins[i].y:
            fullArea += (maxes[i].y - mins[i].y)

            means.append(Point(mins[i].x, (maxes[i].y + mins[i].y) / 2))

    return means, fullArea
