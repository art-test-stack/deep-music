from torch import nn
import torchaudio

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        print("Loading Wav2Vec2 model...")
        self.encoder = torchaudio.models.wav2vec2_model()

    def forward(self, audio_waveform):
        return self.encoder(audio_waveform).mean(dim=1)
