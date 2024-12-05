from scipy import signal


def preprocess_ch(ch_data, fs):
    """Pre-process EEG data by applying a 0.5 Hz highpass filter, a 60  Hz lowpass filter and a 50 Hz notch filter,
    all 4th order Butterworth filters. The data is resampled to 200 Hz.

    Args:
        ch_data: a list or numpy array containing the data of an EEG channel
        fs: the sampling frequency of the data

    Returns:
        ch_data: a numpy array containing the processed EEG data
        fs_resamp: the sampling frequency of the processed EEG data
    """

    # fs_resamp = 200

    b, a = signal.butter(4, 0.5 / (fs / 2), "high")
    ch_data = signal.filtfilt(b, a, ch_data, axis=1)

    b, a = signal.butter(4, 60 / (fs / 2), "low")
    ch_data = signal.filtfilt(b, a, ch_data, axis=1)

    b, a = signal.butter(4, [49.5 / (fs / 2), 50.5 / (fs / 2)], "bandstop")
    ch_data = signal.filtfilt(b, a, ch_data, axis=1)

    # ch_data = signal.resample(ch_data, int(fs_resamp*len(ch_data)/fs))
    ch_data = ch_data / 1e6
    return ch_data


def get_conversion_order(default_labels):
    # default_labels = [
    #     "Fp1-F3",
    #     "F3-C3",
    #     "C3-P3",
    #     "P3-O1",
    #     "Fp1-F7",
    #     "F7-T3",
    #     "T3-T5",
    #     "T5-O1",
    #     "Fz-Cz",
    #     "Cz-Pz",
    #     "Fp2-F4",
    #     "F4-C4",
    #     "C4-P4",
    #     "P4-O2",
    #     "Fp2-F8",
    #     "F8-T4",
    #     "T4-T6",
    #     "T6-O2",
    # ]

    # Adjust the labels to match the new order
    def adjust_label(label):
        mapping = {"T3": "T7", "T5": "P7", "T4": "T8", "T6": "P8"}
        for old, new in mapping.items():
            label = label.replace(old, new)
        return label.upper()

    adjusted_default_labels = [adjust_label(label) for label in default_labels]

    # New order labels
    new_order_labels = [
        "FP1-F7",
        "F7-T7",
        "T7-P7",
        "P7-O1",
        "FP1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "FP2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "FP2-F8",
        "F8-T8",
        "T8-P8",
        "P8-O2",
        "FZ-CZ",
        "CZ-PZ",
    ]

    # Create a mapping from labels to indices
    label_to_index = {
        label: idx for idx, label in enumerate(adjusted_default_labels)
    }

    # Get the indices for the new order
    new_order_indices = [label_to_index[label] for label in new_order_labels]
    return new_order_indices
