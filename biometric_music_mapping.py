# biometric_music_mapping.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pretty_midi

class BiometricMusicDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for biometric-to-music mapping.

    The dataset consists of paired biometric signals and corresponding MIDI files.
    Biometric signals are in the form of heart rate time series data.
    MIDI files contain the musical elements like notes, rhythms, and expressions.

    Data format:
    - Biometric signals: NumPy arrays of shape (1000, 1)
    - MIDI files: Pretty MIDI format (.mid)

    Data acquisition:
    - Collect heart rate signals from participants using wearable devices.
    - Record corresponding MIDI files for each heart rate signal sequence.

    Data size:
    - The dataset contains 5000 paired examples.
    - The sequence length of heart rate signals is 1000 timestamps.
    """

    def __init__(self, biometric_data, midi_files):
        self.biometric_data = biometric_data
        self.midi_files = midi_files

    def __len__(self):
        return len(self.biometric_data)

    def __getitem__(self, idx):
        biometric_seq = self.biometric_data[idx]
        midi_file = self.midi_files[idx]
        
        # Load and preprocess MIDI file
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        midi_seq = self._process_midi(midi_data)
        
        return biometric_seq, midi_seq

    def _process_midi(self, midi_data):
        # Extract note pitches and durations from MIDI data
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append([note.pitch, note.duration])
        
        # Convert notes to a numpy array
        midi_seq = np.array(notes)
        
        return midi_seq


class BiometricMusicModel(nn.Module):
    """
    Deep learning model for mapping biometric signals to musical elements.

    The model architecture is based on long short-term memory (LSTM) networks.
    It takes heart rate signal sequences as input and generates corresponding musical sequences.

    Hyperparameters:
    - input_size: 1 (heart rate signal)
    - hidden_size: 128
    - num_layers: 2
    - output_size: 2 (note pitch and duration)
    - learning_rate: 0.001
    - batch_size: 32
    """

    def __init__(self):
        super(BiometricMusicModel, self).__init__()

        self.input_size = 1
        self.hidden_size = 128
        self.num_layers = 2
        self.output_size = 2
        self.learning_rate = 0.001
        self.batch_size = 32

        # Define the model architecture
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Define the loss function
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # Forward pass through the model
        outputs, _ = self.lstm(x)
        outputs = self.fc(outputs)
        return outputs

    def train_model(self, dataloader, num_epochs):
        """
        Train the biometric-to-music mapping model.

        Args:
        - dataloader: DataLoader object containing the training data
        - num_epochs: Number of training epochs (50)

        Returns:
        - None
        """
        self.train()
        for epoch in range(num_epochs):
            for biometric_seq, midi_seq in dataloader:
                # Forward pass
                outputs = self(biometric_seq)
                loss = self.criterion(outputs, midi_seq)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    def generate_music(self, biometric_seq):
        """
        Generate musical sequence from heart rate signal sequence.

        Args:
        - biometric_seq: Input heart rate signal sequence

        Returns:
        - generated_music: Generated musical sequence
        """
        self.eval()
        with torch.no_grad():
            output = self(biometric_seq)
            generated_music = self._postprocess_output(output)
        return generated_music

    def _postprocess_output(self, model_output):
        # Postprocess the model output to obtain the generated musical sequence
        generated_notes = []
        for note_data in model_output.squeeze():
            pitch = int(note_data[0])
            duration = note_data[1]
            generated_notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=0, end=duration))
        
        # Create a PrettyMIDI object and add the generated notes
        generated_midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        instrument.notes = generated_notes
        generated_midi.instruments.append(instrument)
        
        return generated_midi


def main():
    # Load and preprocess the dataset
    biometric_data = np.random.randn(5000, 1000, 1)  # Example: Simulated heart rate data
    midi_files = ["midi_file_{}.mid".format(i) for i in range(5000)]  # Example: Corresponding MIDI file names
    dataset = BiometricMusicDataset(biometric_data, midi_files)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model
    model = BiometricMusicModel()

    # Train the model
    num_epochs = 50
    model.train_model(dataloader, num_epochs)

    # Generate music from biometric signals
    biometric_seq = np.random.randn(1, 1000, 1)  # Example: Simulated heart rate sequence for testing
    generated_music = model.generate_music(torch.from_numpy(biometric_seq).float())

    # Save the generated music as a MIDI file
    generated_music.write("generated_music.mid")

if __name__ == "__main__":
    main()
