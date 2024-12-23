from torch.utils.data import DataLoader

class MusicDataset:
    def __init__(self, text_descriptions, audio_waveforms):
        self.texts = text_descriptions
        self.audio = audio_waveforms
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.audio[idx]
