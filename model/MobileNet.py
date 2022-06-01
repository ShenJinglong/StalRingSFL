import sys
sys.path.append("..")
import torch

from model.ModelBase import ModelBase

class MobileNet_Mnist(ModelBase):
    def __init__(self) -> None:
        super().__init__()
        self._blocks = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),

            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),

            torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),

            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),

            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),

            torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),

            torch.nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),

            torch.nn.AvgPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 10)
        ])
        self.block_num = len(self._blocks)

class MobileNetSimple_Mnist(ModelBase):
    def __init__(self) -> None:
        super().__init__()
        self._blocks = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),

            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),

            torch.nn.AvgPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(6272, 10)
        ])
        self.block_num = len(self._blocks)

if __name__ == "__main__":
    data = torch.randn((1, 1, 28, 28))
    model = MobileNet_Mnist()
    output = model(data)
    print(output)