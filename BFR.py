# -*- coding: utf-8 -*-
"""
Creation date: 06.04.2018

Katarzyna Filipiuk
katarzyna.filipiuk@student.uw.edu.pl
Urszula Romaniuk
urszula.romaniuk@student.uw.edu.pl
Izabela Szopa
im.szopa@student.uw.edu.pl

Version: 3.0
Date: 19.04.2018
"""
import numba
from timeit import timeit
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import buttord, butter, filtfilt, hilbert
from scipy.io import loadmat

name = "AL2442-STA-wires.mat"
name_of_matlab_matrix = "rfOut_1"
aperture = 64
f0 = 5.5e6
fs = 50e6
pitch = 0.21e-3
c = 1490

temp = loadmat(name)
data = np.array(temp[name_of_matlab_matrix])
no_transducers = data.shape[0]
no_samples = data.shape[1]
no_events = data.shape[2]
depth = no_samples * c / (2 * fs)
dx = depth / no_samples


#@numba.jit
def _highpass(matrix, wp=6e4, ws=1e4, gpass=3, gstop=20):
    """
    Filtering out low frequencies of signal
    :param wp: Passband edge frequency [Hz]
    :param ws: Stopband edge frequency [Hz]
    :param gpass: The maximum loss in the passband [dB]
    :param gstop: The minimum attenuation in the stopband [dB]
    :return: filtered matrix
    """

    n, wn = buttord(wp * 2. / fs, ws * 2. / fs, gpass, gstop)
    b, a = butter(n, wn, btype='high')
    return filtfilt(b, a, matrix, axis=1)


#@numba.jit
def _hilbert_transform(matrix):
    """
    Calculating an envelope of a RF signal
    :param matrix: data which hilbert transform should be calculated
    :return: hilbert transform of a given matrix
    """

    return np.abs(hilbert(matrix, 2 * no_samples,
                          axis=1))[:, :no_samples]


#@numba.jit
def _delay(column_distance, h):
    """
    Calculate delay for one transducer in aperture
    :param column_distance: Distance between middle column of an aperture 
                            and receiver column
    :param h: Depth of a receiver focal point
    :return: List of delays in samples for every transducer in an aperture
    """

    delay = (np.sqrt(h ** 2 + (column_distance * pitch) ** 2) - h) / \
            c * fs
    delay = int(np.round((-1) * delay, 0))
    return delay


#@numba.jit
def _generate_delays_profile(r=16):
    """
    Calculate delays profile for all transducers in aperture
    :param r: Distance in samples between emission of a wave form a first
              transducer of an aperture and emission of a wave form a 
              middle transducer
    :return: List of delays in samples for every transducer in an aperture
    """

    t_max = r / fs
    focal_point = ((aperture * pitch / 2.) ** 2 - t_max ** 2 * \
                   c ** 2) / (2 * c * t_max)

    delay_profile = np.zeros((aperture), dtype=int)
    for i in range(aperture):
        column_distance = np.abs(aperture / 2. - i)
        delay_profile[i] = _delay(column_distance,
                                  focal_point)
    return delay_profile


#@numba.jit
def _bfr(matrix):
    """
    Calculate reconstruction of a USG data
    :param matrix: filtered USG data
    :return: reconstruction of a USG data
    """

    delay_profile = _generate_delays_profile()

    reconstruction = np.zeros((no_events - aperture,
                               no_samples))
    half_aperture = aperture // 2

    for i in range(half_aperture, no_events - half_aperture):
        temp = matrix[i - half_aperture: i + half_aperture, :, i]
        for j in range(aperture):
            temp[j, :] = np.roll(temp[j, :], delay_profile[j])
        reconstruction[i - half_aperture, :] = np.sum(temp, axis=0)
    reconstruction_envelope = _hilbert_transform(reconstruction)
    return reconstruction_envelope


#@numba.jit
def _db_conversion(matrix):
    """
    Converting a matrix to log scale
    :param matrix: data which should be converted to log scale
    :return: a given matrix in log scale
    """

    norm = np.max(matrix)
    return 20 * np.log10(matrix / norm)


#@numba.jit
def _plot_reconstruction(matrix, from_sample=300, to_sample=-300,
                         cutoff=-50, from_transducer=0, to_transducer=0):
    """
    Plot reconstruction of a USG signal
    :param matrix:
    :param from_sample:
    :param cutoff:
    :return:
    """

    plt.figure()
    plt.imshow(matrix[:, from_sample:to_sample].T, cmap="Greys_r",
               interpolation='bilinear', vmin=cutoff, vmax=0,
               extent=[(0 + from_transducer) * pitch * 100,
                       pitch * (no_transducers + to_transducer) \
                       * 100, (no_samples + to_sample) * dx * \
                       100, from_sample * dx * 100])
    plt.title("Reconstruction", fontsize=24)
    plt.xlabel("Horizon [cm]", fontsize=20)
    plt.ylabel("Depth [cm]", fontsize=20)
    plt.colorbar()
    plt.subplots_adjust(left=0.0, right=0.86)


data_centered_around_0 = _highpass(data)
bfr = _bfr(data_centered_around_0)
bfr = _db_conversion(bfr)
_plot_reconstruction(bfr, cutoff=-40,
                     from_transducer=aperture // 2,
                     to_transducer=-(aperture // 2))
plt.show()
