import torch.nn as nn

class Network(nn.Module):
    def __init__(self, nchannels, nclasses):
        super(Network, self).__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'A']
        self.features = make_layers(cfg, nchannels)
        self.classifier = nn.Sequential( nn.Linear( 512, 512 ), nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear( 512, nclasses))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, in_channels):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.5)]
        elif v == 'A':
            layers += [nn.MaxPool2d(kernel_size=4, stride=4), nn.Dropout(0.5)]
        else: #nn.BatchNorm2d(v)
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
