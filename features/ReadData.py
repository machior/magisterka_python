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

import matplotlib.pyplot as plt

from features.AuxiliaryFunctions import *
from features.Constants import MAX_TIME


def get_derivatives(x, y, t_stamp):
    t_steps = pl.diff(t_stamp)
    d_x = pl.diff(x)
    d_y = pl.diff(y)
    v_x = d_x / t_steps

    return d_x, d_y, t_steps, v_x


def process_points(raw_points):
    values = ValuesArrays()

    for raw_point in raw_points:
        values.append(raw_point)

    return values.get_values(), values.get_relatified_values()


def calculate_vel_acc(dimensions, t_stamp):
    t_steps = pl.diff(t_stamp)
    v = []
    a = []

    for dimension in dimensions:
        derivative = dimension / t_steps
        v.append(derivative)
        a.append(pl.diff(derivative) / t_steps[1:])

    return v, a


class SignatureParams:
    features_container = []
    features_names = [
        'signature_duration_ratio', 'a_Ratio',
        'pen_ups_ratio', 'pen_up_time_ratio', 'pen_down_ratio',
        'mass_center_x', 'mass_center_y', 'mass_center_p',
        'mean_med_x', 'mean_med_y', 'mean_med_p',
        'points_above_center_r', 'points_left_to_center_r', 'harder_points_r',
        'std_dev_x', 'std_dev_y', 'std_dev_p',
        'x_travel_ratio',
        'x_a_trend', 'y_a_trend', 'p_a_trend',
        'x_b_trend', 'y_b_trend', 'p_b_trend'
    ]

    # basic values
    x_abs = None
    y_abs = None
    p_abs = None
    t_stamp = None

    # relatived values
    x_r = None
    y_r = None
    p_r = None
    t_r = None

    # basic plots objects
    x_plot = None
    y_plot = None
    p_plot = None

    # velocities
    v_x_plot = None
    v_y_plot = None
    v_p_plot = None

    # accelerations
    a_x_plot = None
    a_y_plot = None
    a_p_plot = None

    # derivatives
    t_steps = None
    d_x = None
    d_y = None
    d_p = None
    v_x = None
    v_y = None
    v_p = None
    a_x = None
    a_y = None
    a_p = None

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
            visualise_signature=False
        )

    def fill_basic_fields(self):
        [x_abs, y_abs, p_abs, t_abs], [x_r, y_r, p_r, t_r] = process_points(self.rawPoints)

        self.x_r = x_r
        self.y_r = y_r
        self.p_r = p_r
        self.t_r = t_r

        self.x_abs = x_abs
        self.y_abs = y_abs
        self.p_abs = p_abs
        self.t_stamp = t_abs

        self.x_plot = SignPlot(x_abs, t_abs, "x(t)")
        self.y_plot = SignPlot(y_abs, t_abs, "y(t)")
        self.p_plot = SignPlot(p_abs, t_abs, "p(t)")

    def basic_calculates(self):
        # derivatives and velocities
        self.t_steps = pl.diff(self.t_stamp)
        derivatives = derive_and_smooth([self.x_abs, self.y_abs, self.p_abs])
        [self.d_x, self.d_y, self.d_p] = derivatives
        # [self.v_x, self.v_y, self.v_p], [self.a_x, self.a_y, self.a_p] = calculate_vel_acc(derivatives, self.t_stamp)

        # self.v_x_plot = SignPlot(self.v_x, self.t_stamp[1:], "v_x(t)")
        # self.v_y_plot = SignPlot(self.v_y, self.t_stamp[1:], "v_y(t)")
        # self.v_p_plot = SignPlot(self.v_p, self.t_stamp[1:], "v_p(t)")

        # self.a_x_plot = SignPlot(self.a_x, self.t_stamp[1:-1], "v_x(t)")
        # self.a_y_plot = SignPlot(self.a_y, self.t_stamp[1:-1], "v_y(t)")
        # self.a_p_plot = SignPlot(self.a_p, self.t_stamp[1:-1], "v_p(t)")
        # slant
        slant = self.d_y / self.d_x
        slant = np.delete(slant, np.argwhere(np.isnan(slant)))
        self.slant = np.mean(np.degrees(np.arctan(slant)))
        # sign path features
        path_traveled = np.cumsum((np.sqrt(self.d_x ** 2 + self.d_y ** 2)))
        self.path_traveled = path_traveled = np.insert(path_traveled, 0, 0)
        self.pen_velocity = pen_velocity = pl.diff(path_traveled) / self.t_steps
        self.pen_acceleration = pl.diff(pen_velocity) / self.t_steps[1:]

    def calculate_features(
            self,
            visualise_signature=False
    ):
        if visualise_signature:
            plt.plot(self.x_plot.val(), self.y_plot.val())
            plt.gca().invert_yaxis()
            plt.title("Signature visualisation")
            plt.show()
            self.x_plot.plot()
            self.y_plot.plot()

        ##################################################
        # GENERAL DEPENDENCIES
        ##################################################
        # signature duration
        signatureDuration = self.t_stamp[-1]
        signature_duration_ratio = 1 if signatureDuration < 1 else 1 / signatureDuration

        # Component Time Spacing and Pen-Ups
        pen_ups = [t_step for t_step in self.t_steps if t_step > 2 * self.t_steps[0]]
        pen_ups_ratio = 1 / (len(pen_ups) + 1)
        penUpTime = sum(pen_ups)
        pen_up_time_ratio = penUpTime / MAX_TIME if penUpTime < MAX_TIME else 1

        # pen-down ratio
        penDownTime = signatureDuration - penUpTime
        pen_down_ratio = penDownTime / signatureDuration

        ##################################################
        # y(x) DEPENDENCIES
        ##################################################
        # aspect ratio
        hLength = max(self.x_abs) - min(self.x_abs)
        vHeight = max(self.y_abs) - min(self.y_abs)
        a_Ratio = normalise_aRatio(hLength / vHeight)

        # mass center / mean value
        mass_center_x = np.mean(self.x_r)
        mass_center_y = np.mean(self.y_r)
        mass_center_p = np.mean(self.p_r)

        # upper part advantage
        points_above_center = len([y_r for y_r in self.y_r if y_r > 0.5])
        points_above_center_r = points_above_center / len(self.y_r)

        # left part advantage
        points_left_to_center = len([x_r for x_r in self.x_r if x_r > 0.5])
        points_left_to_center_r = points_left_to_center / len(self.x_r)

        # harder part advantage
        harder_points = len([p_r for p_r in self.p_r if p_r > 0.5])
        harder_points_r = harder_points / len(self.p_r)

        # std deviation
        std_dev_x = np.std(self.x_r)
        std_dev_y = np.std(self.y_r)
        std_dev_p = np.std(self.p_r)

        # cumsum axis x / abs ( 1 / curvature )
        path_traveled_x = sum(abs(self.d_x))
        x_travel_ratio = path_traveled_x / max(self.path_traveled)


        ##################################################
        # x(t) DEPENDENCIES
        ##################################################
        # x trend line
        x_trend = np.polyfit(self.x_r, self.t_r, 1)
        x_a_trend = normalise_trend(x_trend[0])
        x_b_trend = normalise_trend(x_trend[1])

        # y trend line
        y_trend = np.polyfit(self.y_r, self.t_r, 1)
        y_a_trend = normalise_trend(y_trend[0])
        y_b_trend = normalise_trend(y_trend[1])

        # y trend line
        p_trend = np.polyfit(self.p_r, self.t_r, 1)
        p_a_trend = normalise_trend(p_trend[0])
        p_b_trend = normalise_trend(p_trend[1])

        # median value
        mean_med_x = pl.median(self.x_r)
        mean_med_y = pl.median(self.y_r)
        mean_med_p = pl.median(self.p_r)

        # mean / median velocity
        # mean_max_vel = pl.mean(self.pen_velocity) / pl.median(self.pen_velocity)

        # Average velocity
        # meanVelocity = pl.mean(self.pen_velocity)

        self.features_container = [
            signature_duration_ratio, a_Ratio,
            pen_ups_ratio, pen_up_time_ratio, pen_down_ratio,
            mass_center_x, mass_center_y, mass_center_p,
            mean_med_x, mean_med_y, mean_med_p,
            points_above_center_r, points_left_to_center_r, harder_points_r,
            std_dev_x, std_dev_y, std_dev_p,
            x_travel_ratio,
            x_a_trend, y_a_trend, p_a_trend,
            x_b_trend, y_b_trend, p_b_trend,
        ]


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
            pressure=int(values[3]),
            end_pts=values[4]
        ))
    file.close()

    return raw_points
