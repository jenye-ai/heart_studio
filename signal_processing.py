from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from sklearn.utils import resample

def process_sound(filename):
    # Read the audio file
    sample_rate, audio_data = wavfile.read(filename)
    # .shape returns an array of size 1 if there is one channel and an array of size 2 for more than 1 channel
    channels = len(audio_data.shape)

    # Stereo to Mono conversion
    if channels == 2:
       audio_data = stereo_to_mono(audio_data)
       channels = 1

    # Downsampling
    if sample_rate > 16000:
        sample_factor = 16000/sample_rate
        audio_data = resample(audio_data, n_samples=(int(audio_data.shape[0]*sample_factor)))
        sample_rate = 16000
    else:
        return

    # N is the number of channels, interested in frequencies 0 to 1000 Hz
    N = 9

    # Spacing between each channel
    spacing = 100

    low = 20
    high = 100
    output_signal = 0

    for i in range(N+1):
        # Increment the lower and upper cutoff frequencies by 100 Hz
        low = low + i*spacing
        high = low + spacing

        # BANDPASS FILTER
        filtered_signal = butter_bandpass_filter(audio_data, low, high, sample_rate, 4)
        central_freq = (low + high)/2

        # ENVELOPE EXTRACTION
        # Rectify the output signal
        rectified_signal = abs(filtered_signal)

        # Detect envelopes of rectified signals using a lowpass filter
        envelope = butter_lowpass_filter(rectified_signal, 400, sample_rate, 4)
        envelope = envelope.transpose()

        # Generate cos signal with central frequency of bandpass filters and length of rectified signals
        time_duration = rectified_signal.shape[0]/sample_rate
        time = np.linspace(0, time_duration, num=rectified_signal.shape[0])
        cos_signal = np.cos(2*np.pi*central_freq*time)

        # AMPLITUDE MODULATION
        am_signal = envelope*cos_signal

        # Add the amplitude modulated signals for each channel to obtain output signal
        output_signal += am_signal

    # Normalize the output signal by the max of its absolute value
    output_signal_norm = output_signal / np.max(abs(output_signal))

    # Normalize resampled audio by its max
    audio_data_norm = audio_data / np.max(abs(audio_data))
    print(rectified_signal.shape[0])
    print(time)

    # Magnitude of output signal
    #output_signal_db = 20*np.log10(abs(output_signal))
    #audio_data_db = 20*np.log10(abs(audio_data))

    #plt.figure()
    #plt.magnitude_spectrum(audio_data, Fs=sample_rate, scale='dB')

    #plt.figure()
    #plt.magnitude_spectrum(output_signal, Fs=sample_rate, scale='dB')

    # Plot the output signal
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(time, audio_data_norm)
    ax1.set_title("Original Audio (Resampled)")
    ax1.set(xlabel= "Sample Number", ylabel= "Amplitude")

    ax2.plot(time, output_signal_norm)
    ax2.set_title("Filtered Audio")
    ax2.set(xlabel="Sample Number", ylabel="Amplitude")
    fig.tight_layout(pad=2.0)

    plt.show()

    return output_signal


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    # Nyquist frequency
    nyq = 0.5 * fs

    # Cutoff frequency must be between 0 and 1 so if upper cutoff frequency
    # equals to the Nyquist frequency, decrease it slightly
    if highcut == nyq:
        highcut -= 0.000000001

    # Normalize lower and upper cutoff frequencies by dividing by Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq

    # Use Butterworth bandpass filter to obtain transfer function coefficients b,a
    b, a = sig.butter(order, [low, high], btype='band')

    # Obtain filtered signal
    y = sig.lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order):
    # Nyquist frequency
    nyq = 0.5*fs

    # Cutoff frequency must be between 0 and 1 so if upper cutoff frequency
    # equals to the Nyquist frequency, decrease it slightly
    if cutoff == nyq:
        cutoff -= 0.000000001

    # Normalize cutoff frequency by dividing by Nyquist frequency
    cutoff_freq = cutoff / nyq

    # Use Butterworth lowpass filter to obtain transfer function coefficients b,a
    b, a = sig.butter(order, cutoff_freq, btype="low")

    # Obtain filtered signal
    y = sig.lfilter(b, a, data)
    return y


# Convert stereo sounds to mono
def stereo_to_mono(audio_data):
    audio_data = audio_data.astype(float)
    new_audio_data = []

    for i in range(len(audio_data)):
        # Not sure whether division by 2 is necessary
        d = audio_data[:, 0]/2 + audio_data[:, 1]/2
        new_audio_data.append(d)

    return np.array(new_audio_data, dtype = "int16")


if __name__ == '__main__':
    filepath = "C:\\Users\\mavel\\Documents\\BME 261\\Testing Set\\"
    filename = "Normal1.wav"
    sound_file = filepath + filename
    #playsound(sound_file)
    process_sound(sound_file)
