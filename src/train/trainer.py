from src.data.dataset import MusicDataset
from settings import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_model = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            self.save_model = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.save_model = True


class Trainer:
    def __init__(
            self,
            text_encoder: nn.Module,
            audio_encoder: nn.Module,
            decoder: nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.Module,
            early_stopping_patience: int = 5,
            name: str = "model_v1",
            device: str = "cpu",
        ) -> None:
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.loss_fn = loss_fn(reduction="sum")
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.device = device
        self.history = { "train": [], "test": [] }
        self.path = MODEL_PATH / f"{name} / checkpoint.pt"
        self.history_path = MODEL_PATH / f"{name} / history.png"
        self.name = name
        try: 
            self.load()
        except:
            print("No model found at", self.path)
        self.to(device)

    def fit(
            self, 
            trainloader: DataLoader, 
            testloader: DataLoader,
            num_epochs: int
        ) -> None:
        self.to(self.device)
        with tqdm(range(num_epochs)) as pbar:
            for epoch in pbar:
                train_loss = self._train_step(trainloader)
                self.history["train"].append(train_loss)
                test_loss = self.evaluate(testloader)
                self.history["loss"].append(test_loss)
                pbar.set_postfix(
                    train_loss=train_loss, test_loss=test_loss)
                self.early_stopping(test_loss)
                if self.early_stopping.early_stop:
                    break
                if self.early_stopping.save_model:
                    self.save()
        self.plot_history(display=False)

    def save(self, path: str | Path = None):
        path = path or self.path
        self.path = path
        torch.save({
            'text_encoder': self.text_encoder.state_dict(),
            'audio_encoder': self.audio_encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self):
        checkpoint = torch.load(self.path)
        self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        self.audio_encoder.load_state_dict(checkpoint['audio_encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.to(self.device)
    
    def evaluate(self, testloader):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for text, audio in testloader:
                text, audio = text.to(self.device), audio.to(self.device)
                text_embeds, audio_embeds, generated_audio = self._forward(text, audio)

                reconstruction_loss = ((generated_audio - audio) ** 2).mean()
                contrastive_loss = torch.nn.functional.cosine_embedding_loss(text_embeds, audio_embeds, torch.ones(len(text)))
                loss = reconstruction_loss + contrastive_loss
                total_loss += loss.item()
        return total_loss / len(testloader)
    
    def predict(self, text):
        self.eval()
        with torch.no_grad():
            text_embeds = self.text_encoder(text)
            return self.decoder(text_embeds)
    
    def get_embeddings(self, text):
        self.eval()
        with torch.no_grad():
            text_embeds = self.text_encoder(text)
            return text_embeds
    
    def get_audio_embeddings(self, audio):
        self.eval()
        with torch.no_grad():
            audio_embeds = self.audio_encoder(audio)
            return audio_embeds
        
    def plot_history(self, save_path: str = None, display: bool = True):
        plt.plot(self.history["train"], label="train")
        plt.plot(self.history["test"], label="test")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        elif display:
            plt.show()
        else:
            plt.savefig(self.history_path)
        
    def _train_step(self, trainloader):
        self.train()
        for text, audio in trainloader:
            text, audio = text.to(self.device), audio.to(self.device)
            text_embeds, audio_embeds, generated_audio = self._forward(text, audio)

            reconstruction_loss = ((generated_audio - audio) ** 2).mean()
            contrastive_loss = torch.nn.functional.cosine_embedding_loss(text_embeds, audio_embeds, torch.ones(len(text)))
            loss = reconstruction_loss + contrastive_loss / len(trainloader.dataset)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()
    
    def _forward(self, text, audio):
        text_embeds = self.text_encoder(text)
        audio_embeds = self.audio_encoder(audio)
        generated_audio = self.decoder(text_embeds)
        return text_embeds, audio_embeds, generated_audio

    def to(self, device: str | None):
        if device is None and self.device is None:
            return self
        self.device = device or self.device
        self.text_encoder.to(device)
        self.audio_encoder.to(device)
        self.decoder.to(device)
        return self

    def _check_device(self, text, audio):
        assert text.device == self.text_encoder.weight.device, "Text input is not on the same device as text_encoder"
        assert audio.device == self.audio_encoder.weight.device, "Audio input is not on the same device as audio_encoder"
        assert text.device == self.decoder.weight.device, "Text input is not on the same device as decoder"
        assert audio.device == self.decoder.weight.device, "Audio input is not on the same device as decoder"

    def eval(self):
        self.text_encoder.eval()
        self.audio_encoder.eval()
        self.decoder.eval()
        return self
    
    def train(self):
        self.text_encoder.train()
        self.audio_encoder.train()
        self.decoder.train()
        return self