import torch.nn as nn

class Generator(nn.Module):
    def __init__(
            self, 
            num_gpus: int = 0, 
            input_size: int = 100,
            feature_maps_size: int = 64,
            channel_size: int = 3,
        ):
        super(Generator, self).__init__()
        self.num_gpus = num_gpus
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_size, feature_maps_size * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(feature_maps_size * 16),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(feature_maps_size * 16, feature_maps_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps_size * 8, feature_maps_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps_size * 4, feature_maps_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps_size * 2, feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps_size, channel_size, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

