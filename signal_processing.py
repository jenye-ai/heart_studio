from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig


def process_sound(filename):
    # Read the audio file
    sample_rate, audio_data = wavfile.read(filename)
    channels = len(audio_data.shape)

    # Stereo to Mono conversion if necessary
    #if channels == 2:
    #    audio_data = stereo_to_mono(audio_data)
    #    channels = 1

    orig_time_duration = audio_data.shape[0] / sample_rate

    # Downsample - not sure if the 16000 is significant or arbitrary
    if sample_rate > 16000:
        # Fix this, is decimate better?
        sig.resample(audio_data, 16000)
        sample_rate = 16000
    else:
        return

    # N is the number of channels
    # Interested in frequencies 0 to 1000 Hz
    N = 9

    # Spacing between each channel
    spacing = 100

    low = 20
    high = 100
    output_signal = 0

    # Loop through each channel
    for i in range(N+1):
        # Increment the lower and upper cutoff frequencies by 100 Hz
        low = low + i*spacing
        high = low + spacing

        # BANDPASS FILTER BANK
        filtered_signal = butter_bandpass_filter(audio_data, low, high, sample_rate, 4)
        central_freq = (low + high)/2

        # ENVELOPE EXTRACTION
        # Rectify the output signal
        rectified_signal = abs(filtered_signal)

        # Detect envelopes of rectified signals using a lowpass filter
        # What should the cutoff frequency be here?
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

        # Plotting
        if i == 1:
            plt.figure()
            plt.plot(filtered_signal)
            plt.title("Output Signal of Channel 1")
            plt.xlabel("Sample Number")
            plt.ylabel("Amplitude")

            #plt.figure()
            #b,a = butter_bandpass(low,high,sample_rate, 4)
            #w, h = sig.freqz(b, a, worN=2000)
            #plt.plot((sample_rate * 0.5 / np.pi) * w, abs(h), label="order = 5")
            #plt.xlim([0, 1000])
            plt.figure()
            plt.plot(envelope)
            plt.title("Envelope")
            plt.xlabel("Sample Number")
            plt.ylabel("Amplitude")

        elif i == N:
            plt.figure()
            plt.plot(filtered_signal)
            plt.title("Output Signal of Channel N")
            plt.xlabel("Sample Number")
            plt.ylabel("Amplitude")

            plt.figure()
            plt.plot(envelope)
            plt.title("Envelope")
            plt.xlabel("Sample Number")
            plt.ylabel("Amplitude")

    # Normalize the output signal by the max of its absolute value
    output_signal = output_signal / np.max(abs(output_signal))

    # Plot the output signal
    plt.figure()
    plt.plot(output_signal)
    plt.title("Output Signal")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")

    # Show all plots
    plt.show()

    sound = np.random.uniform(-1, 1, sample_rate)
    #wavfile.write('test1.wav', sample_rate, sound)

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


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a


# Plot signal and save it as a png (can change this later)
# Should plot be saved or
def plot_signal(audio_data, time_duration):
    time = np.linspace(0.,time_duration, audio_data.shape[0])
    plt.plot(time, audio_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()
    plt.savefig("PCG_plot.png")


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
    filename = "fo-4.wav"
    sound_file = filepath + filename
    process_sound(sound_file)
