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

name = "AL2442-CPWI-wires.mat"
name_of_matlab_matrix = "rfOut_0"
delay_in_samples = 45
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
def _transmit_delay(row, column_distance, transducers_delay):
    """
	Calculate time required for a sound wave to travel to receiver
	:param row: Row of matrix
	:param column_distance: Distance between column containing chosen pixel
							and receiver column
	:param transducers_delay: Distance in samples between emission of a 
							  wave form a first transducer and emission of
							  a wave form a last transducer
	:return: Time required for a sound wave to travel to receiver 
			 in samples
	"""

    temp1 = row * dx * np.sqrt((transducers_delay * dx) ** 2 / \
                               (no_transducers * pitch) ** 2
                               + 1)
    temp2 = np.sqrt((column_distance * pitch) ** 2 + (row * dx) ** 2)

    delay = (temp1 + temp2) / c * fs
    delay = int(np.round(delay, 0))
    if delay >= no_samples:
        return no_samples - 1
    return delay


#@numba.jit
def _cpwi_for_angle(matrix, transducers_delay):
    """
	Calculate reconstruction for a given angle
	:param matrix: data from single event
	:param transducers_delay: Distance in samples between emission of a 
							  wave form a first transducer and emission of 
							  a wave form a last transducer
	:return: reconstruction for a given angle
	"""

    reconstruction = np.zeros((no_transducers, no_samples))

    for i in range(no_samples):
        delays = [_transmit_delay(i, k, transducers_delay)
                  for k in range(-no_transducers + 1,
                                 no_transducers)]
        for j in range(no_transducers):
            reconstruction[j, i] = np.sum(matrix[
                                              list(range(no_transducers)),
                                              delays[no_transducers - j
                                                     - 1: 2 * no_transducers - j
                                                          - 1]])
    return reconstruction


##@numba.jit
def _cpwi(matrix):
    """
	Calculate reconstruction of a USG data
	:param matrix: filtered USG data
	:return: reconstruction of a USG data
	"""

    transducers_delays = [i * delay_in_samples for i in
                          range(-(no_events // 2),
                                no_events // 2 + 1)]
    reconstruction = np.zeros_like(matrix)

    for i, transducers_delay in enumerate(transducers_delays):
        reconstruction[:, :, i] = _cpwi_for_angle(matrix[:, :, i],
                                                  transducers_delay)
        print("Computing PWI for", i, "event")
    reconstruction = np.sum(reconstruction, axis=2)
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


##@numba.jit
def run():
    """
	Create full reconstruction of a USG data
	"""

    data_centered_around_0 = _highpass(data)

    cpwi = _cpwi(data_centered_around_0)
    cpwi = _db_conversion(cpwi)
    _plot_reconstruction(cpwi, from_sample=1000, to_sample=-300,
                         cutoff=-50, from_transducer=0,
                         to_transducer=0)
    plt.show()


run()
