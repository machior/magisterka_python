import numpy as np
import pylab as pl
from numpy.random.mtrand import randn


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


class ValuesArrays:
    def __init__(self):
        self.x = []
        self.y = []
        self.p = []
        self.t = []

    def append(self, point):
        self.x.append(point.x)
        self.y.append(point.y)
        self.p.append(point.pressure)
        self.t.append(point.t_stamp)

    def get_values(self):
        return self.x, self.y, self.p, self.t

    @staticmethod
    def relativify_field(field):
        f_max = max(field)
        f_min = min(field)
        f_r = []
        for f in field:
            f_r.append((f - f_min) / (f_max - f_min))

        return f_r

    def get_relatified_values(self):
        x_r = self.relativify_field(self.x)
        y_r = self.relativify_field(self.y)
        p_r = self.relativify_field(self.p)
        t_r = self.relativify_field(self.t)

        return x_r, y_r, p_r, t_r


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        # w = np.hanning(window_len)
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    l_diff = len(y) - len(x)
    start_diff = int(l_diff / 2)
    end_diff = l_diff - start_diff
    return y[start_diff: -end_diff]


def smooth_demo():
    t = np.linspace(-4, 4, 100)
    x = np.sin(t)
    xn = x + randn(len(t)) * 0.1
    y = smooth(x)

    ws = 31

    pl.subplot(211)
    pl.plot(np.ones(ws))

    windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    pl.hold(True)
    for w in windows[1:]:
        eval('pl.plot(' + w + '(ws) )')

    pl.axis([0, 30, 0, 1.1])

    pl.legend(windows)
    pl.title("The smoothing windows")
    pl.subplot(212)
    pl.plot(x)
    pl.plot(xn)
    for w in windows:
        pl.plot(smooth(xn, 10, w))
    l = ['original signal', 'signal with noise']
    l.extend(windows)

    pl.legend(l)
    pl.title("Smoothing a noisy signal")
    pl.show()


def normalise_trend(val):
    if val > 1:
        val = 1
    elif val < -1:
        val = -1

    return (val + 1) / 2


def normalise_aRatio(val):
    MAX_RATIO = 10
    if val > MAX_RATIO:
        val = MAX_RATIO

    return val / MAX_RATIO


def derive_and_smooth(arrays):
    result = []
    for array in arrays:
        result.append(smooth(pl.diff(array)))

    return result
