import numpy as np
from scipy import signal


def taper(data, taper_length=None, max_percentage=0.05) -> np.ndarray:
    data = data.copy()
    if taper_length is None:
        taper_length = max(2, round(len(data) * max_percentage))
    taper_window = signal.windows.hann(2 * taper_length)
    data[:taper_length] = data[:taper_length] * taper_window[:taper_length]
    data[-taper_length:] = data[-taper_length:] * taper_window[-taper_length:]
    return data


def cal_sos(srate, freq_band, butter_order=4):
    fn = srate / 2
    if (freq_band[0] == 0) and (freq_band[1] != 0) and (freq_band[1] / fn < 1):
        sos = signal.butter(
            butter_order, freq_band[1] / fn, btype="lowpass", output="sos"
        )
    elif (freq_band[0] != 0) and ((freq_band[1] == 0) or (freq_band[1] / fn >= 1)):
        sos = signal.butter(
            butter_order, freq_band[0] / fn, btype="highpass", output="sos"
        )
    elif (freq_band[0] != 0) and (freq_band[1] != 0) and (freq_band[1] / fn < 1):
        sos = signal.butter(
            butter_order,
            [freq_band[0] / fn, freq_band[1] / fn],
            btype="bandpass",
            output="sos",
        )
    else:
        sos = None
    return sos


def filter_butter(data: np.ndarray, srate, freq_band, butter_order=4, zero_phase=False):
    data = data.copy()
    sos = cal_sos(srate, freq_band, butter_order)
    if sos is not None:
        if zero_phase:
            data = signal.sosfiltfilt(sos, data)
        else:
            data = signal.sosfilt(sos, data)
    return data


def resample(data, srate_old: float, srate_new: float, zero_phase=True):
    if zero_phase:
        if float(srate_old).is_integer() and float(srate_new).is_integer():
            srate_old = int(srate_old)
            srate_new = int(srate_new)
            gcd = np.gcd(srate_new, srate_old)
            p = srate_new // gcd
            q = srate_old // gcd
            data = signal.resample_poly(data, p, q)
        else:
            data = signal.resample(data, round(len(data) * srate_new / srate_old))
            data = filter_butter(
                data=data,
                srate=srate_new,
                freq_band=[0, srate_old / 2],
                zero_phase=zero_phase,
            )
    else:
        q = srate_old / srate_new
        if srate_new < srate_old:
            if q.is_integer():
                data = signal.decimate(
                    data, q=int(q), ftype="fir", zero_phase=zero_phase
                )
            else:
                data = filter_butter(
                    data=data,
                    srate=srate_old,
                    freq_band=[0, srate_new / 2],
                    zero_phase=zero_phase,
                )
                data = signal.resample(x=data, num=round(len(data) * q))
        elif srate_new > srate_old:
            data = signal.resample(data, round(len(data) * q))
            data = filter_butter(
                data=data,
                srate=srate_new,
                freq_band=[0, srate_old / 2],
                zero_phase=zero_phase,
            )
    return data


def linear_interp(data, N_new) -> np.ndarray:
    points_loc = np.arange(0, len(data))
    points_loc_new = np.linspace(0, len(data), N_new, endpoint=False)
    data_new = np.interp(points_loc_new, points_loc, data)
    return data_new


def correct_zero_frequency(data, srate, A0, f_c, tc1, tc2, ratio_interp=0):
    N_data = len(data)
    data = data[tc1:tc2]
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

    w = np.zeros(N)
    w[0 : f_c + 1] = 1 - 1 / 2 * (1 + np.cos(np.pi * np.arange(f_c + 1) / f_c))  # 0->1
    w[-f_c:] = w[1 : f_c + 1][::-1]

    uf_correct[1 : f_c + 1] = (
        (1 - w[1 : f_c + 1]) * np.abs(A0) + w[1 : f_c + 1] * A_f[1 : f_c + 1]
    ) * np.exp(1j * np.complex128(phi_f[1 : f_c + 1]))
    uf_correct[-f_c:] = ((1 - w[-f_c:]) * np.abs(A0) + w[-f_c:] * A_f[-f_c:]) * np.exp(
        1j * np.complex128(phi_f[-f_c:])
    )

    if ratio_interp > 0:
        u_correct = np.real(np.fft.ifft(uf_correct)) * srate * ratio_interp
        # u_correct = filter_butter(data=u_correct, srate=srate * ratio_interp,
        #                           freq_band=[0, srate / 2])
        u_correct = resample(
            data=u_correct, srate_old=srate * ratio_interp, srate_new=srate
        )
    else:
        u_correct = np.real(np.fft.ifft(uf_correct)) * srate
    # u_correct = u_correct - u_correct[0]
    # u_correct = taper(u_correct, taper_len)
    u_correct = np.concatenate([np.zeros(tc1), u_correct, np.zeros(N_data - tc2)])
    return u_correct
