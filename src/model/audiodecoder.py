from torch import nn

class MusicDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MusicDecoder, self).__init__()
        print("Creating MusicDecoder...")
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1) 
        
    def forward(self, embeddings):
        outputs, _ = self.lstm(embeddings)
        return self.fc(outputs)
