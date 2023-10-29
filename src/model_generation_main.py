import torch
import torchvision
import torch.functional as F


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)

    def forward(self, inputs):
        return self.resnet(inputs)


def main() -> None:
    # TODO: replace this with your code
    model = TestModel()
    dummy_input = torch.randn(1, 3, 224, 224)
    scripted_model = torch.jit.script(model)
    scripted_model(dummy_input)
    torch.jit.save(scripted_model, "model.pt")

if __name__ == "__main__":
    main()
