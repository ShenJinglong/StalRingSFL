import torch

class ModelBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, start=0, stop=None):
        if stop == None:
            stop = self.block_num
        for block in self._blocks[start:stop]:
            x = block(x)
        return x

    def get_params(self, start=0, stop=None):
        if stop == None:
            stop = self.block_num
        return [param for param in self._blocks[start:stop].parameters()]
