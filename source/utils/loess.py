from scipy import signal


def loessline(ls):
    smoothed_r = signal.savgol_filter(ls, 111, 9)

    return smoothed_r
