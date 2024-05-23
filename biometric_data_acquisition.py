import mne
import biosppy
import numpy as np
import pandas as pd
from typing import List, Dict

class BiometricDataAcquisition:
    """
    A module for acquiring and preprocessing biometric data from EEG, EMG, and heart rate sensors.
    """
    def __init__(self, eeg_channels: List[str], emg_channels: List[str], hr_channel: str, sampling_rate: int):
        """
        Initialize the BiometricDataAcquisition module.

        Args:
            eeg_channels (List[str]): List of EEG channel names.
            emg_channels (List[str]): List of EMG channel names.
            hr_channel (str): Name of the heart rate channel.
            sampling_rate (int): Sampling rate of the biometric signals (in Hz).

        Possible Errors:
        - ValueError: If the sampling rate is not positive.
        - TypeError: If the channel names are not provided as lists or strings.

        Solutions:
        - Ensure that the sampling rate is a positive integer.
        - Provide the channel names as lists of strings for EEG and EMG, and as a single string for heart rate.
        """
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be a positive integer.")
        if not isinstance(eeg_channels, list) or not isinstance(emg_channels, list) or not isinstance(hr_channel, str):
            raise TypeError("Invalid channel name format. EEG and EMG channels must be lists, and heart rate channel must be a string.")

        self.eeg_channels = eeg_channels
        self.emg_channels = emg_channels
        self.hr_channel = hr_channel
        self.sampling_rate = sampling_rate

    def acquire_eeg_data(self, duration: int) -> pd.DataFrame:
        """
        Acquire EEG data from the specified channels for the given duration.

        Args:
            duration (int): Duration of the EEG data acquisition (in seconds).

        Returns:
            pd.DataFrame: DataFrame containing the acquired EEG data.

        Data Format:
        - The EEG data is returned as a pandas DataFrame.
        - Each row represents a time point, and each column represents an EEG channel.
        - The DataFrame has a shape of (num_samples, num_eeg_channels), where num_samples = duration * sampling_rate.

        Data Acquisition Process:
        - The EEG data is acquired using the `mne` library, which interfaces with EEG acquisition devices.
        - The `mne.io.RawArray` class is used to create a raw object from the acquired EEG data.
        - The raw object is then converted to a DataFrame using `raw.to_data_frame()`.

        Possible Errors:
        - ValueError: If the duration is not positive.
        - RuntimeError: If there is an issue with the EEG acquisition device or the `mne` library.

        Solutions:
        - Ensure that the duration is a positive integer.
        - Check the connection and configuration of the EEG acquisition device.
        - Verify that the `mne` library is properly installed and configured.
        """
        if duration <= 0:
            raise ValueError("Duration must be a positive integer.")

        try:
            # Create an MNE info structure for the EEG data
            info = mne.create_info(self.eeg_channels, self.sampling_rate, ch_types='eeg')

            # Generate random EEG data for demonstration purposes
            num_samples = duration * self.sampling_rate
            eeg_data = np.random.rand(num_samples, len(self.eeg_channels))

            # Create an MNE raw object from the EEG data
            raw = mne.io.RawArray(eeg_data.T, info)

            # Convert the raw object to a DataFrame
            eeg_df = raw.to_data_frame()

            return eeg_df

        except Exception as e:
            raise RuntimeError(f"Error occurred during EEG data acquisition: {str(e)}")

    def acquire_emg_data(self, duration: int) -> pd.DataFrame:
        """
        Acquire EMG data from the specified channels for the given duration.

        Args:
            duration (int): Duration of the EMG data acquisition (in seconds).

        Returns:
            pd.DataFrame: DataFrame containing the acquired EMG data.

        Data Format:
        - The EMG data is returned as a pandas DataFrame.
        - Each row represents a time point, and each column represents an EMG channel.
        - The DataFrame has a shape of (num_samples, num_emg_channels), where num_samples = duration * sampling_rate.

        Data Acquisition Process:
        - The EMG data is acquired using the `mne` library, which interfaces with EMG acquisition devices.
        - The `mne.io.RawArray` class is used to create a raw object from the acquired EMG data.
        - The raw object is then converted to a DataFrame using `raw.to_data_frame()`.

        Possible Errors:
        - ValueError: If the duration is not positive.
        - RuntimeError: If there is an issue with the EMG acquisition device or the `mne` library.

        Solutions:
        - Ensure that the duration is a positive integer.
        - Check the connection and configuration of the EMG acquisition device.
        - Verify that the `mne` library is properly installed and configured.
        """
        if duration <= 0:
            raise ValueError("Duration must be a positive integer.")

        try:
            # Create an MNE info structure for the EMG data
            info = mne.create_info(self.emg_channels, self.sampling_rate, ch_types='emg')

            # Generate random EMG data for demonstration purposes
            num_samples = duration * self.sampling_rate
            emg_data = np.random.rand(num_samples, len(self.emg_channels))

            # Create an MNE raw object from the EMG data
            raw = mne.io.RawArray(emg_data.T, info)

            # Convert the raw object to a DataFrame
            emg_df = raw.to_data_frame()

            return emg_df

        except Exception as e:
            raise RuntimeError(f"Error occurred during EMG data acquisition: {str(e)}")

    def acquire_heart_rate_data(self, duration: int) -> pd.DataFrame:
        """
        Acquire heart rate data from the specified channel for the given duration.

        Args:
            duration (int): Duration of the heart rate data acquisition (in seconds).

        Returns:
            pd.DataFrame: DataFrame containing the acquired heart rate data.

        Data Format:
        - The heart rate data is returned as a pandas DataFrame.
        - Each row represents a time point, and the column represents the heart rate value.
        - The DataFrame has a shape of (num_samples, 1), where num_samples = duration * sampling_rate.

        Data Acquisition Process:
        - The heart rate data is acquired using the `biosppy` library, which provides tools for biosignal processing.
        - The `biosppy.signals.ecg.ecg` function is used to simulate an ECG signal.
        - The heart rate is then extracted from the simulated ECG signal using the `biosppy.signals.ecg.heart_rate` function.

        Possible Errors:
        - ValueError: If the duration is not positive.
        - RuntimeError: If there is an issue with the `biosppy` library or the heart rate extraction process.

        Solutions:
        - Ensure that the duration is a positive integer.
        - Verify that the `biosppy` library is properly installed and configured.
        """
        if duration <= 0:
            raise ValueError("Duration must be a positive integer.")

        try:
            # Generate a simulated ECG signal using biosppy
            num_samples = duration * self.sampling_rate
            ecg_signal = biosppy.signals.ecg.ecg(signal=np.random.rand(num_samples), sampling_rate=self.sampling_rate, show=False)[1]

            # Extract heart rate from the ECG signal
            hr = biosppy.signals.ecg.heart_rate(signal=ecg_signal, sampling_rate=self.sampling_rate, window=10)

            # Create a DataFrame from the heart rate data
            hr_df = pd.DataFrame({self.hr_channel: hr}, index=pd.RangeIndex(start=0, stop=len(hr), step=1))

            return hr_df

        except Exception as e:
            raise RuntimeError(f"Error occurred during heart rate data acquisition: {str(e)}")

    def preprocess_data(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Preprocess the acquired biometric data based on the data type (EEG, EMG, or heart rate).

        Args:
            data (pd.DataFrame): DataFrame containing the biometric data.
            data_type (str): Type of biometric data ('eeg', 'emg', or 'heart_rate').

        Returns:
            pd.DataFrame: Preprocessed biometric data.

        Preprocessing Steps:
        - EEG Data:
          - Apply a bandpass filter (1-30 Hz) to remove low-frequency drifts and high-frequency noise.
          - Perform artifact removal using Independent Component Analysis (ICA).
        - EMG Data:
          - Apply a bandpass filter (10-500 Hz) to remove low-frequency motion artifacts and high-frequency noise.
          - Perform full-wave rectification to convert the EMG signal to a positive amplitude.
        - Heart Rate Data:
          - Apply a moving average filter to smooth the heart rate signal.

        Possible Errors:
        - ValueError: If the data type is not recognized.
        - RuntimeError: If there is an issue with the preprocessing functions or libraries.

        Solutions:
        - Ensure that the data type is one of 'eeg', 'emg', or 'heart_rate'.
        - Verify that the required libraries for preprocessing (e.g., `mne`, `biosppy`) are properly installed and configured.
        """
        try:
            if data_type == 'eeg':
                # Apply bandpass filter to EEG data
                data_filtered = mne.filter.filter_data(data, self.sampling_rate, l_freq=1, h_freq=30)
                
                # Perform artifact removal using ICA
                ica = mne.preprocessing.ICA(n_components=len(self.eeg_channels), random_state=42)
                ica.fit(data_filtered)
                data_preprocessed = ica.apply(data_filtered)
                
            elif data_type == 'emg':
                # Apply bandpass filter to EMG data
                data_filtered = mne.filter.filter_data(data, self.sampling_rate, l_freq=10, h_freq=500)
                
                # Perform full-wave rectification
                data_preprocessed = np.abs(data_filtered)
                
            elif data_type == 'heart_rate':
                # Apply moving average filter to heart rate data
                window_size = self.sampling_rate * 2  # 2-second window
                data_preprocessed = data.rolling(window=window_size, center=True).mean()
                
            else:
                raise ValueError(f"Unrecognized data type: {data_type}")
                
            return data_preprocessed

        except Exception as e:
            raise RuntimeError(f"Error occurred during data preprocessing: {str(e)}")
        
def main():
    # Example usage
    eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    emg_channels = ['EMG1', 'EMG2', 'EMG3', 'EMG4']
    hr_channel = 'HR'
    sampling_rate = 250  # 250 Hz

    data_acquisition = BiometricDataAcquisition(eeg_channels, emg_channels, hr_channel, sampling_rate)

    duration = 10  # 10 seconds

    # Acquire and preprocess EEG data
    eeg_data = data_acquisition.acquire_eeg_data(duration)
    eeg_data_preprocessed = data_acquisition.preprocess_data(eeg_data, 'eeg')
    print("EEG Data:")
    print(eeg_data_preprocessed.head())

    # Acquire and preprocess EMG data
    emg_data = data_acquisition.acquire_emg_data(duration)
    emg_data_preprocessed = data_acquisition.preprocess_data(emg_data, 'emg')
    print("EMG Data:")
    print(emg_data_preprocessed.head())

    # Acquire and preprocess heart rate data
    hr_data = data_acquisition.acquire_heart_rate_data(duration)
    hr_data_preprocessed = data_acquisition.preprocess_data(hr_data, 'heart_rate')
    print("Heart Rate Data:")
    print(hr_data_preprocessed.head())

if __name__ == "__main__":
    main()
