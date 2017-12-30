# This function is provided as it is, without any warranty.
# Parameters:
# FileName: file name of the signature.
# ShowSig: if set to non 0 value displays signature.

# Output:
# X: x coordinates
# Y: y coordinates
# TStamps: point acquisition time in msec(relative to 1'st point)
# Pressure: self explanatory
# EndPts: 1 indicates end of a signature segment, 0 otherwise

# from pylab import *
import csv

import pylab as pl
import matplotlib.pyplot as plt
import numpy as np

from features.AuxiliaryFunctions import smooth


def get_derivatives(x, y, t_stamp):
    t_steps = pl.diff(t_stamp)
    d_x = pl.diff(x)
    d_y = pl.diff(y)
    v_x = d_x / t_steps

    return d_x, d_y, t_steps, v_x


class RawPoint:
    def __init__(self, x, y, t_stamp, pressure, end_pts):
        self.x = x
        self.y = y
        self.t_stamp = t_stamp
        self.pressure = pressure
        self.end_pts = end_pts


class SignPlot:
    def __init__(self, y, x, title=None):
        self.y = y
        self.x = x
        if title:
            self.title = title

    def dom(self):
        return self.x

    def val(self):
        return self.y

    def plot(self):
        if self.title:
            pl.title(self.title)
        pl.plot(self.x, self.y, marker='o', markersize=2)
        pl.show()


class SignatureParams:
    features_container = []

    # basic values
    x_val = None
    y_val = None
    p_val = None
    t_stamp = None

    # basic plots objects
    x_plot = None
    y_plot = None
    p_plot = None

    # velocities
    v_x_plot = None
    v_y_plot = None
    v_p_plot = None
    t_steps = None

    # derivatives
    d_x = None
    d_y = None
    d_p = None
    v_x = None
    v_y = None
    v_p = None

    # additional parameters
    slant = None
    path_traveled = None
    pen_velocity = None
    pen_acceleration = None

    def __init__(self, raw_points):
        self.rawPoints = raw_points
        self.fill_basic_fields()

        self.basic_calculates()

        self.calculate_features(
            visualise_signature=True
        )

    def fill_basic_fields(self):
        self.x_val = x_val = []
        self.y_val = y_val = []
        self.p_val = p_val = []
        self.t_stamp = t_val = []

        try:
            for raw_point in self.rawPoints:
                x_val.append(raw_point.x)
                y_val.append(raw_point.y)
                p_val.append(raw_point.pressure)
                t_val.append(raw_point.t_stamp)
        except Exception:
            print(Exception)

        self.x_plot = SignPlot(x_val, t_val, "x(t)")
        self.y_plot = SignPlot(y_val, t_val, "y(t)")
        self.p_plot = SignPlot(p_val, t_val, "p(t)")

    def basic_calculates(self):
        # derivatives and velocities
        self.t_steps = t_steps = pl.diff(self.x_plot.dom())
        self.d_x = d_x = smooth(pl.diff(self.x_plot.val()))
        self.d_y = d_y = smooth(pl.diff(self.y_plot.val()))
        self.d_p = d_p = smooth(pl.diff(np.asarray(self.p_plot.val(), dtype=int)))
        self.v_x_plot = SignPlot(d_x / t_steps, t_steps, "v_x(t)")
        self.v_y_plot = SignPlot(d_y / t_steps, t_steps, "v_y(t)")
        self.v_p_plot = SignPlot(d_p / t_steps, t_steps, "v_p(t)")
        # slant
        slant = np.divide(self.d_y, self.d_x)
        slant = np.delete(slant, np.argwhere(np.isnan(slant)))
        self.slant = np.degrees(np.arctan(slant))
        # sign path features
        path_traveled = np.cumsum((np.sqrt(self.d_x ** 2 + self.d_y ** 2)))
        self.path_traveled = path_traveled = np.insert(path_traveled, 0, 0)
        self.pen_velocity = pen_velocity = pl.diff(path_traveled) / self.t_steps
        self.pen_acceleration = pl.diff(pen_velocity) / self.t_steps[1:]

    def calculate_features(
            self,
            visualise_signature=False,
            signature_center=True,
            signature_duration=True,
            component_time_spacing=True,
            pen_down_ratio=True,
            horizontal_length=True,
            aspect_ratio=True,
            pen_ups=True,
            cursiviness=True):
        # print('slant: \n', self.slant)
        # plot(self.rawPoints.y, self.rawPoints.x)
        # plot(self.t_stamp, self.y)
        # figure()
        # plot(self.x, self.y, marker='o', markersize=2)
        # figure()
        # plot(self.t_stamp[1:], self.pen_velocity)
        # plot(self.t_stamp[1:-1], self.pen_acceleration)
        # # ion()
        # show()
        # # center of signature
        # xCenter = mean(Xmeans);
        # yCenter = mean(Ymeans);
        # fullVector = [xCenter;
        # yCenter];
        if not visualise_signature:
            # plt.ion()
            plt.plot(self.x_plot.val(), self.y_plot.val())
            plt.gca().invert_yaxis()
            plt.title("Signature visualisation")
            plt.show()
            self.x_plot.plot()
            self.y_plot.plot()

        # signature center
        x_center = min(self.x_val) + (max(self.x_val) - min(self.x_val)) / 2
        y_center = min(self.y_val) + (max(self.y_val) - min(self.y_val)) / 2

        # signature duration
        signatureDuration = self.t_stamp[-1]

        # Component Time Spacing
        penUpTime = sum([t_step for t_step in self.t_steps if t_step > 2 * self.t_steps[0]])

        # # pen-down ratio
        penDownTime = self.t_stamp[-1] - penUpTime
        penDownRatio = penDownTime / self.t_stamp[-1]

        # horizontal length
        hLength = max(self.x_val) - min(self.x_val)

        # aspect ratio
        aRatio = hLength / (max(self.y_val) - min(self.y_val))

        # pen-ups
        penUps = sum(self.t_steps > self.t_steps[1]) + 1

        # cursiviness
        cursiviness = hLength / penUps

        # top heaviness
        topHeav = pl.mean(self.y_val) / pl.median(self.y_val)

        # Horizontal Dispersion
        horDisp = pl.mean(self.x_val) / pl.median(self.x_val)

        # curvature
        curvature = sum(pl.sqrt(self.d_x ** 2 + self.d_y ** 2)) / hLength

        # Strokes
        # smoothedX = smoothen_plot(self.x_val, 5)
        # smoothedY = smoothen_plot(self.y_val, 5)
        # [XlocExtr, XlocExtrInd] = getExtremes(smoothedX)
        # [YlocExtr, YlocExtrInd] = getExtremes(smoothedY)
        # hStrokes = length(XlocExtrInd) + 1
        # vStrokes = length(YlocExtrInd) + 1

        # Maximum velocity
        maxVelocity = max(self.pen_velocity)

        # Average velocity
        meanVelocity = pl.mean(self.pen_velocity)

        # print('signatureDuration: ', signatureDuration,
        #       '\npenDownRatio: ', penDownRatio,
        #       '\naRatio: ', aRatio, '\ncursiviness: ', cursiviness)

        self.features_container = [signatureDuration, penDownRatio, aRatio, cursiviness]
        # with open('test.csv', 'w') as csvfile:
        #     spamwriter = csv.writer(csvfile)
        #     spamwriter.writerow(['signatureDuration', 'penDownRatio', 'aRatio', 'cursiviness'])
        #     spamwriter.writerow([signatureDuration, penDownRatio, aRatio, cursiviness])


def read_signature(file_name):
    try:
        file = open(file_name, 'r')
    except FileNotFoundError:
        print('File ' + str(file_name) + ' not found')
        return

    file.readline()
    num_of_points = int(file.readline()[10:])
    # print(num_of_points)

    raw_points = []

    for line in file:
        values = line.split(' ')
        raw_points.append(RawPoint(
            x=int(values[0]),
            y=int(values[1]),
            t_stamp=int(values[2]),
            pressure=values[3],
            end_pts=values[4]
        ))
    file.close()

    return raw_points
