"""Tools for spectral analysis.
"""

import jax.numpy as np
#from . import signaltools
from .windows import get_window
from .arraytools import const_ext, even_ext, odd_ext, zero_ext
import warnings


__all__ = ['stft', 'istft']


def stft(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None,
         detrend=False, return_onesided=True, boundary='zeros', padded=True,
         axis=-1):
    r"""
    Compute the Short Time Fourier Transform (STFT).

    STFTs can be used as a way of quantifying the change of a
    nonstationary signal's frequency and phase content over time.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to 256.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`. When
        specified, the COLA constraint must be met (see Notes below).
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to `False`.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        'zeros', for zero padding extension. I.e. ``[1, 2, 3, 4]`` is
        extended to ``[0, 1, 2, 3, 4, 0]`` for ``nperseg=3``.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `True`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`, as is the
        default.
    axis : int, optional
        Axis along which the STFT is computed; the default is over the
        last axis (i.e. ``axis=-1``).

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Zxx : ndarray
        STFT of `x`. By default, the last axis of `Zxx` corresponds
        to the segment times.

    See Also
    --------
    istft: Inverse Short Time Fourier Transform
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint
                is met
    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met
    welch: Power spectral density by Welch's method.
    spectrogram: Spectrogram by Welch's method.
    csd: Cross spectral density by Welch's method.
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT in
    `istft`, the signal windowing must obey the constraint of "Nonzero
    OverLap Add" (NOLA), and the input signal must have complete
    windowing coverage (i.e. ``(x.shape[axis] - nperseg) %
    (nperseg-noverlap) == 0``). The `padded` argument may be used to
    accomplish this.

    Given a time-domain signal :math:`x[n]`, a window :math:`w[n]`, and a hop
    size :math:`H` = `nperseg - noverlap`, the windowed frame at time index
    :math:`t` is given by

    .. math:: x_{t}[n]=x[n]w[n-tH]

    The overlap-add (OLA) reconstruction equation is given by

    .. math:: x[n]=\frac{\sum_{t}x_{t}[n]w[n-tH]}{\sum_{t}w^{2}[n-tH]}

    The NOLA constraint ensures that every normalization term that appears
    in the denomimator of the OLA reconstruction equation is nonzero. Whether a
    choice of `window`, `nperseg`, and `noverlap` satisfy this constraint can
    be tested with `check_NOLA`.

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.
    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from
           Modified Short-Time Fourier Transform", IEEE 1984,
           10.1109/TASSP.1984.1164317

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly
    modulated around 3kHz, corrupted by white noise of exponentially
    decreasing magnitude sampled at 10 kHz.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2 * np.sqrt(2)
    >>> noise_power = 0.01 * fs / 2
    >>> time = np.arange(N) / float(fs)
    >>> mod = 500*np.cos(2*np.pi*0.25*time)
    >>> carrier = amp * np.sin(2*np.pi*3e3*time + mod)
    >>> noise = np.random.normal(scale=np.sqrt(noise_power),
    ...                          size=time.shape)
    >>> noise *= np.exp(-time/5)
    >>> x = carrier + noise

    Compute and plot the STFT's magnitude.

    >>> f, t, Zxx = signal.stft(x, fs, nperseg=1000)
    >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
    >>> plt.title('STFT Magnitude')
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()
    """

    freqs, time, Zxx = _spectral_helper(x, x, fs, window, nperseg, noverlap,
                                        nfft, detrend, return_onesided,
                                        scaling='spectrum', axis=axis,
                                        mode='stft', boundary=boundary,
                                        padded=padded)

    return freqs, time, Zxx


def istft(Zxx, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
          input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2):
    r"""
    Perform the inverse Short Time Fourier transform (iSTFT).

    Parameters
    ----------
    Zxx : array_like
        STFT of the signal to be reconstructed. If a purely real array
        is passed, it will be cast to a complex data type.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window. Must match the window used to generate the
        STFT for faithful inversion.
    nperseg : int, optional
        Number of data points corresponding to each STFT segment. This
        parameter must be specified if the number of data points per
        segment is odd, or if the STFT was padded via ``nfft >
        nperseg``. If `None`, the value depends on the shape of
        `Zxx` and `input_onesided`. If `input_onesided` is `True`,
        ``nperseg=2*(Zxx.shape[freq_axis] - 1)``. Otherwise,
        ``nperseg=Zxx.shape[freq_axis]``. Defaults to `None`.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`, half
        of the segment length. Defaults to `None`. When specified, the
        COLA constraint must be met (see Notes below), and should match
        the parameter used to generate the STFT. Defaults to `None`.
    nfft : int, optional
        Number of FFT points corresponding to each STFT segment. This
        parameter must be specified if the STFT was padded via ``nfft >
        nperseg``. If `None`, the default values are the same as for
        `nperseg`, detailed above, with one exception: if
        `input_onesided` is True and
        ``nperseg==2*Zxx.shape[freq_axis] - 1``, `nfft` also takes on
        that value. This case allows the proper inversion of an
        odd-length unpadded STFT using ``nfft=None``. Defaults to
        `None`.
    input_onesided : bool, optional
        If `True`, interpret the input array as one-sided FFTs, such
        as is returned by `stft` with ``return_onesided=True`` and
        `numpy.fft.rfft`. If `False`, interpret the input as a a
        two-sided FFT. Defaults to `True`.
    boundary : bool, optional
        Specifies whether the input signal was extended at its
        boundaries by supplying a non-`None` ``boundary`` argument to
        `stft`. Defaults to `True`.
    time_axis : int, optional
        Where the time segments of the STFT is located; the default is
        the last axis (i.e. ``axis=-1``).
    freq_axis : int, optional
        Where the frequency axis of the STFT is located; the default is
        the penultimate axis (i.e. ``axis=-2``).

    Returns
    -------
    t : ndarray
        Array of output data times.
    x : ndarray
        iSTFT of `Zxx`.

    See Also
    --------
    stft: Short Time Fourier Transform
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint
                is met
    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met

    Notes
    -----
    In order to enable inversion of an STFT via the inverse STFT with
    `istft`, the signal windowing must obey the constraint of "nonzero
    overlap add" (NOLA):

    .. math:: \sum_{t}w^{2}[n-tH] \ne 0

    This ensures that the normalization factors that appear in the denominator
    of the overlap-add reconstruction equation

    .. math:: x[n]=\frac{\sum_{t}x_{t}[n]w[n-tH]}{\sum_{t}w^{2}[n-tH]}

    are not zero. The NOLA constraint can be checked with the `check_NOLA`
    function.

    An STFT which has been modified (via masking or otherwise) is not
    guaranteed to correspond to a exactly realizible signal. This
    function implements the iSTFT via the least-squares estimation
    algorithm detailed in [2]_, which produces a signal that minimizes
    the mean squared error between the STFT of the returned signal and
    the modified STFT.

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.
    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from
           Modified Short-Time Fourier Transform", IEEE 1984,
           10.1109/TASSP.1984.1164317

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    Generate a test signal, a 2 Vrms sine wave at 50Hz corrupted by
    0.001 V**2/Hz of white noise sampled at 1024 Hz.

    >>> fs = 1024
    >>> N = 10*fs
    >>> nperseg = 512
    >>> amp = 2 * np.sqrt(2)
    >>> noise_power = 0.001 * fs / 2
    >>> time = np.arange(N) / float(fs)
    >>> carrier = amp * np.sin(2*np.pi*50*time)
    >>> noise = np.random.normal(scale=np.sqrt(noise_power),
    ...                          size=time.shape)
    >>> x = carrier + noise

    Compute the STFT, and plot its magnitude

    >>> f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg)
    >>> plt.figure()
    >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
    >>> plt.ylim([f[1], f[-1]])
    >>> plt.title('STFT Magnitude')
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.yscale('log')
    >>> plt.show()

    Zero the components that are 10% or less of the carrier magnitude,
    then convert back to a time series via inverse STFT

    >>> Zxx = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)
    >>> _, xrec = signal.istft(Zxx, fs)

    Compare the cleaned signal with the original and true carrier signals.

    >>> plt.figure()
    >>> plt.plot(time, x, time, xrec, time, carrier)
    >>> plt.xlim([2, 2.1])
    >>> plt.xlabel('Time [sec]')
    >>> plt.ylabel('Signal')
    >>> plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
    >>> plt.show()

    Note that the cleaned signal does not start as abruptly as the original,
    since some of the coefficients of the transient were also removed:

    >>> plt.figure()
    >>> plt.plot(time, x, time, xrec, time, carrier)
    >>> plt.xlim([0, 0.1])
    >>> plt.xlabel('Time [sec]')
    >>> plt.ylabel('Signal')
    >>> plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
    >>> plt.show()

    """

    # Make sure input is an ndarray of appropriate complex dtype
    Zxx = np.asarray(Zxx) + 0j
    freq_axis = int(freq_axis)
    time_axis = int(time_axis)

    if Zxx.ndim < 2:
        raise ValueError('Input stft must be at least 2d!')

    if freq_axis == time_axis:
        raise ValueError('Must specify differing time and frequency axes!')

    nseg = Zxx.shape[time_axis]

    if input_onesided:
        # Assume even segment length
        n_default = 2*(Zxx.shape[freq_axis] - 1)
    else:
        n_default = Zxx.shape[freq_axis]

    # Check windowing parameters
    if nperseg is None:
        nperseg = n_default
    else:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    if nfft is None:
        if (input_onesided) and (nperseg == n_default + 1):
            # Odd nperseg, no FFT padding
            nfft = nperseg
        else:
            nfft = n_default
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    # Rearrange axes if necessary
    if time_axis != Zxx.ndim-1 or freq_axis != Zxx.ndim-2:
        # Turn negative indices to positive for the call to transpose
        if freq_axis < 0:
            freq_axis = Zxx.ndim + freq_axis
        if time_axis < 0:
            time_axis = Zxx.ndim + time_axis
        zouter = list(range(Zxx.ndim))
        for ax in sorted([time_axis, freq_axis], reverse=True):
            zouter.pop(ax)
        Zxx = np.transpose(Zxx, zouter+[freq_axis, time_axis])

    # Get window as array
    if isinstance(window, str) or type(window) is tuple:
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if win.shape[0] != nperseg:
            raise ValueError('window must have length of {0}'.format(nperseg))

    ifunc = np.fft.irfft if input_onesided else np.fft.ifft
    xsubs = ifunc(Zxx, axis=-2, n=nfft)[..., :nperseg, :]

    # Initialize output and normalization arrays
    outputlength = nperseg + (nseg-1)*nstep
    x = np.zeros(list(Zxx.shape[:-2])+[outputlength], dtype=xsubs.dtype)
    norm = np.zeros(outputlength, dtype=xsubs.dtype)

    if np.result_type(win, xsubs) != xsubs.dtype:
        win = win.astype(xsubs.dtype)

    xsubs *= win.sum()  # This takes care of the 'spectrum' scaling

    # Construct the output from the ifft segments
    # This loop could perhaps be vectorized/strided somehow...
    for ii in range(nseg):
        # Window the ifft
        x[..., ii*nstep:ii*nstep+nperseg] += xsubs[..., ii] * win
        norm[..., ii*nstep:ii*nstep+nperseg] += win**2

    # Remove extension points
    if boundary:
        x = x[..., nperseg//2:-(nperseg//2)]
        norm = norm[..., nperseg//2:-(nperseg//2)]

    # Divide out normalization where non-tiny
    if np.sum(norm > 1e-10) != len(norm):
        warnings.warn("NOLA condition failed, STFT may not be invertible")
    x /= np.where(norm > 1e-10, norm, 1.0)

    if input_onesided:
        x = x.real

    # Put axes back
    if x.ndim > 1:
        if time_axis != Zxx.ndim-1:
            if freq_axis < time_axis:
                time_axis -= 1
            x = np.rollaxis(x, -1, time_axis)

    time = np.arange(x.shape[0])/float(fs)
    return time, x


def _spectral_helper(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
                     nfft=None, detrend='constant', return_onesided=True,
                     scaling='density', axis=-1, mode='psd', boundary=None,
                     padded=False):
    """
    Calculate various forms of windowed FFTs for PSD, CSD, etc.

    This is a helper function that implements the commonality between
    the stft, psd, csd, and spectrogram functions. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Parameters
    ---------
    x : array_like
        Array or sequence containing the data to be analyzed.
    y : array_like
        Array or sequence containing the data to be analyzed. If this is
        the same object in memory as `x` (i.e. ``_spectral_helper(x,
        x, ...)``), the extra computations are spared.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross
        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
        and `y` are measured in V and `fs` is measured in Hz.
        Defaults to 'density'
    axis : int, optional
        Axis along which the FFTs are computed; the default is over the
        last axis (i.e. ``axis=-1``).
    mode: str {'psd', 'stft'}, optional
        Defines what kind of return values are expected. Defaults to
        'psd'.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        `None`.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `False`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`.
    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of times corresponding to each data segment
    result : ndarray
        Array of output data, contents dependent on *mode* kwarg.

    Notes
    -----
    Adapted from matplotlib.mlab

    .. versionadded:: 0.16.0
    """
    if mode not in ['psd', 'stft']:
        raise ValueError("Unknown value for mode %s, must be one of: "
                         "{'psd', 'stft'}" % mode)

    boundary_funcs = {'even': even_ext,
                      'odd': odd_ext,
                      'constant': const_ext,
                      'zeros': zero_ext,
                      None: None}

    if boundary not in boundary_funcs:
        raise ValueError("Unknown boundary option '{0}', must be one of: {1}"
                         .format(boundary, list(boundary_funcs.keys())))

    # If x and y are the same object we can save ourselves some computation.
    same_data = y is x

    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is 'stft'")

    axis = int(axis)

    # Ensure we have np.arrays, get outdtype
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)
        outdtype = np.result_type(x, y, np.complex64)
    else:
        outdtype = np.result_type(x, np.complex64)

    if not same_data:
        # Check if we can broadcast the outer axes together
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = np.broadcast(np.empty(xouter), np.empty(youter)).shape
        except ValueError:
            raise ValueError('x and y cannot be broadcast together.')

    if same_data:
        if x.size == 0:
            return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)
    else:
        if x.size == 0 or y.size == 0:
            outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
            emptyout = np.rollaxis(np.empty(outshape), -1, axis)
            return emptyout, emptyout, emptyout

    if x.ndim > 1:
        if axis != -1:
            x = np.rollaxis(x, axis, len(x.shape))
            if not same_data and y.ndim > 1:
                y = np.rollaxis(y, axis, len(y.shape))

    # Check if x and y are the same length, zero-pad if necessary
    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = np.concatenate((x, np.zeros(pad_shape)), -1)
            else:
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = np.concatenate((y, np.zeros(pad_shape)), -1)

    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    # parse window; if array like, then set nperseg = win.shape
    win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    # Padding occurs after boundary extension, so that the extended signal ends
    # in zeros, instead of introducing an impulse at the end.
    # I.e. if x = [..., 3, 2]
    # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
    # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]

    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg//2, axis=-1)
        if not same_data:
            y = ext_func(y, nperseg//2, axis=-1)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
        nadd = (-(x.shape[-1]-nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = np.concatenate((y, np.zeros(zeros_shape)), axis=-1)

    # Handle detrending and window functions
    if not detrend:
        def detrend_func(d):
            return d
    elif not hasattr(detrend, '__call__'):
        raise NotImplementedError()
        # def detrend_func(d):
        #     return signaltools.detrend(d, type=detrend, axis=-1)
    elif axis != -1:
        # Wrap this function so that it receives a shape that it could
        # reasonably expect to receive.
        def detrend_func(d):
            d = np.rollaxis(d, -1, axis)
            d = detrend(d)
            return np.rollaxis(d, axis, len(d.shape))
    else:
        detrend_func = detrend

    if np.result_type(win, np.complex64) != outdtype:
        win = win.astype(outdtype)

    if scaling == 'density':
        scale = 1.0 / (fs * (win*win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum()**2
    else:
        raise ValueError('Unknown scaling: %r' % scaling)

    if mode == 'stft':
        scale = np.sqrt(scale)

    if return_onesided:
        if np.iscomplexobj(x):
            sides = 'twosided'
            warnings.warn('Input data is complex, switching to '
                          'return_onesided=False')
        else:
            sides = 'onesided'
            if not same_data:
                if np.iscomplexobj(y):
                    sides = 'twosided'
                    warnings.warn('Input data is complex, switching to '
                                  'return_onesided=False')
    else:
        sides = 'twosided'

    if sides == 'twosided':
        freqs = np.fft.fftfreq(nfft, 1/fs)
    elif sides == 'onesided':
        freqs = np.fft.rfftfreq(nfft, 1/fs)

    # Perform the windowed FFTs
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)

    if not same_data:
        # All the same operations on the y data
        result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft,
                               sides)
        result = np.conjugate(result) * result_y
    elif mode == 'psd':
        result = np.conjugate(result) * result

    result *= scale
    if sides == 'onesided' and mode == 'psd':
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[..., 1:-1] *= 2

    time = np.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1,
                     nperseg - noverlap)/float(fs)
    if boundary is not None:
        time -= (nperseg/2) / fs

    result = result.astype(outdtype)

    # All imaginary parts are zero anyways
    if same_data and mode != 'stft':
        result = result.real

    # Output is going to have new last axis for time/window index, so a
    # negative axis index shifts down one
    if axis < 0:
        axis -= 1

    # Roll frequency axis back to axis where the data came from
    result = np.rollaxis(result, -1, axis)

    return freqs, time, result


def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):
    """
    Calculate windowed FFT, for internal use by
    scipy.signal._spectral_helper

    This is a helper function that does the main FFT calculation for
    `_spectral helper`. All input validation is performed there, and the
    data axis is assumed to be the last axis of x. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Returns
    -------
    result : ndarray
        Array of FFT data

    Notes
    -----
    Adapted from matplotlib.mlab

    .. versionadded:: 0.16.0
    """
    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        # https://stackoverflow.com/a/5568169
        step = nperseg - noverlap
        shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
        strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                 strides=strides)

    # Detrend each data segment individually
    result = detrend_func(result)

    # Apply window by multiplication
    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    if sides == 'twosided':
        func = np.fft.fft
    else:
        result = np.real(result)
        func = np.fft.rfft
    result = func(result, n=nfft)

    return result


def _triage_segments(window, nperseg, input_length):
    """
    Parses window and nperseg arguments for spectrogram and _spectral_helper.
    This is a helper function, not meant to be called externally.

    Parameters
    ----------
    window : string, tuple, or ndarray
        If window is specified by a string or tuple and nperseg is not
        specified, nperseg is set to the default of 256 and returns a window of
        that length.
        If instead the window is array_like and nperseg is not specified, then
        nperseg is set to the length of the window. A ValueError is raised if
        the user supplies both an array_like window and a value for nperseg but
        nperseg does not equal the length of the window.

    nperseg : int
        Length of each segment

    input_length: int
        Length of input signal, i.e. x.shape[-1]. Used to test for errors.

    Returns
    -------
    win : ndarray
        window. If function was called with string or tuple than this will hold
        the actual array used as a window.

    nperseg : int
        Length of each segment. If window is str or tuple, nperseg is set to
        256. If window is array_like, nperseg is set to the length of the
        6
        window.
    """

    # parse window; if array like, then set nperseg = win.shape
    if isinstance(window, str) or isinstance(window, tuple):
        # if nperseg not specified
        if nperseg is None:
            nperseg = 256  # then change to default
        if nperseg > input_length:
            warnings.warn('nperseg = {0:d} is greater than input length '
                          ' = {1:d}, using nperseg = {1:d}'
                          .format(nperseg, input_length))
            nperseg = input_length
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if input_length < win.shape[-1]:
            raise ValueError('window is longer than input signal')
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError("value specified for nperseg is different"
                                 " from length of window")
    return win, nperseg
