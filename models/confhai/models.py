import torch 

class Net(torch.nn.Module):
    def __init__(self, input_size, num_classes, hidden = 2):
        super(Net, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden, num_classes)
            )
        self.LRlayer = torch.nn.Sequential(
            torch.nn.Linear(input_size, num_classes)
            )
        self.fc = torch.nn.Linear(input_size, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        #out = self.layers(x)
        #out = self.LRlayer(x)
        out = self.fc(x)
        return out
    
class RNet(torch.nn.Module):
    def __init__(self, input_size, num_classes, hidden = 2):
        super(RNet, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden, num_classes)
            )
        self.LRlayer = torch.nn.Sequential(
            torch.nn.Linear(input_size, num_classes)
            )
        self.fc = torch.nn.Linear(input_size, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        #out = self.layers(x)
        #out = self.LRlayer(x)
        out = self.fc(x)
        return out


class LRNet(torch.nn.Module):
    def __init__(self, input_size, num_classes, hidden = 2):
        super(Net, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden, num_classes)
            )
        self.LRlayer = torch.nn.Sequential(
            torch.nn.Linear(input_size, num_classes)
            )
        self.fc = torch.nn.Linear(input_size, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        #out = self.layers(x)
        out = self.LRlayer(x)
        #out = self.fc(x)
        return out