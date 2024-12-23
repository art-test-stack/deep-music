from src.data.dataset import MusicDataset

from src.model.textencoder import TextEncoder
from src.model.audioencoder import AudioEncoder
from src.model.audiodecoder import MusicDecoder

from src.train.trainer import Trainer

from settings import *

import torch
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__":
    batch_size = 16
    lr = 1e-4 / batch_size
    print("Prepare data...")
    text_descriptions = ["happy music", "sad melody", "upbeat tempo", "relaxing ambient"]
    audio_waveforms = torch.randn(len(text_descriptions), 16000)  # Simulate audio waveforms

    dataset = MusicDataset(text_descriptions, audio_waveforms)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Create model...")

    text_encoder = TextEncoder()
    audio_encoder = AudioEncoder()
    decoder = MusicDecoder(input_dim=768, hidden_dim=512)

    optimizer = torch.optim.Adam(
    list(text_encoder.parameters()) + list(audio_encoder.parameters()) + list(decoder.parameters()), lr=lr)

    device = get_device()
    print(f"Training on {device}")
    trainer = Trainer(text_encoder, audio_encoder, decoder, optimizer, torch.nn.MSELoss, early_stopping_patience=20, device=device)
    
    print("Start training...")
    trainer.fit(train_loader, test_loader, epochs=100)

    trainer.plot_history()