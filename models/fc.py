import torch.nn as nn

class Network(nn.Module):
    def __init__(self, nchannels, nclasses):
        super(Network, self).__init__()
        self.classifier = nn.Sequential(nn.Linear( nchannels * 32 * 32, 1024 ), nn.ReLU(inplace=True),
                                        nn.Linear( 1024, nclasses))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
