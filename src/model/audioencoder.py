import torch
import torch.nn as nn
import torchaudio


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        print("Loading Wav2Vec2 model...")
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.model = bundle.get_model()
    
    def forward(self, audio_waveform):
        with torch.no_grad(): 
            features = self.model.extract_features(audio_waveform)[0]
        return features.mean(dim=1)