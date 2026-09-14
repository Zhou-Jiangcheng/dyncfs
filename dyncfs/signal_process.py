import numpy as np

from pygrnwang.signal_process import resample


def correct_zero_frequency(data, srate, A0, f_c, tc1, tc2, ratio_interp=0):
    """
    Replace the zero-frequency value of data[tc1:tc2] with A0 and taper the
    amplitudes of the lowest f_c frequency bins towards |A0|.

    :param data: 1D array (e.g. stress rate), not modified in place.
    :param srate: Sampling rate in Hz.
    :param A0: Zero-frequency value, i.e. the integral of data over time.
    :param f_c: Number of frequency bins used for the transition.
    :param tc1: Start index of the corrected window.
    :param tc2: End index (exclusive) of the corrected window, clipped to len(data).
    :param ratio_interp: Interpolation ratio before FFT, 0 means no interpolation.
    :return: Corrected data with the same length as the input.
    """
    data = np.asarray(data)
    N_data = len(data)
    tc1 = max(0, int(tc1))
    tc2 = min(N_data, int(tc2))
    if tc2 - tc1 < 2:
        return data.copy()
    data = data[tc1:tc2].copy()
    N_win = len(data)
    data[0] = 0
    data[-1] = 0
    # data = taper(data)
    if ratio_interp > 0:
        # u = np.concatenate([np.zeros(pad_len // 2), data.copy(), np.zeros(pad_len - pad_len // 2)])
        # l = ratio_interp * (tc2 - tc1)
        u = resample(data=data, srate_old=srate, srate_new=srate * ratio_interp)
        uf = np.fft.fft(u) / (srate * ratio_interp)
    else:
        u = data.copy()
        uf = np.fft.fft(u) / srate
    uf_correct = uf.copy()
    uf_correct[0] = A0

    N = len(uf)
    A_f = np.abs(uf)
    phi_f = np.angle(uf)

    # f = np.fft.fftfreq(N, 1 / srate)[:N // 2]
    # f_c = max(2, np.argmin(np.abs(cut_freq - f)))
    # print(f_c)

    # positive and negative frequency bins must not overlap
    f_c = int(min(f_c, (N - 1) // 2))
    if f_c > 0:
        w = np.zeros(N)
        w[0 : f_c + 1] = 1 - 1 / 2 * (1 + np.cos(np.pi * np.arange(f_c + 1) / f_c))  # 0->1
        w[-f_c:] = w[1 : f_c + 1][::-1]

        uf_correct[1 : f_c + 1] = (
            (1 - w[1 : f_c + 1]) * np.abs(A0) + w[1 : f_c + 1] * A_f[1 : f_c + 1]
        ) * np.exp(1j * np.complex128(phi_f[1 : f_c + 1]))
        uf_correct[-f_c:] = (
            (1 - w[-f_c:]) * np.abs(A0) + w[-f_c:] * A_f[-f_c:]
        ) * np.exp(1j * np.complex128(phi_f[-f_c:]))

    if ratio_interp > 0:
        u_correct = np.real(np.fft.ifft(uf_correct)) * srate * ratio_interp
        # u_correct = filter_butter(data=u_correct, srate=srate * ratio_interp,
        #                           freq_band=[0, srate / 2])
        u_correct = resample(
            data=u_correct, srate_old=srate * ratio_interp, srate_new=srate
        )
    else:
        u_correct = np.real(np.fft.ifft(uf_correct)) * srate
    # keep the window length after resampling
    if len(u_correct) >= N_win:
        u_correct = u_correct[:N_win]
    else:
        u_correct = np.concatenate([u_correct, np.zeros(N_win - len(u_correct))])
    # u_correct = u_correct - u_correct[0]
    # u_correct = taper(u_correct, taper_len)
    u_correct = np.concatenate([np.zeros(tc1), u_correct, np.zeros(N_data - tc2)])
    return u_correct
