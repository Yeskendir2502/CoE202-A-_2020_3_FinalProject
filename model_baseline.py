import torch
import torch.nn as nn
import torch.nn.functional as F


class cancer_classifier(nn.Module):
    def __init__(self):
        super(cancer_classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(3*64*64, 64)
        self.linear2 = nn.Linear(64, 2)
    
    def forward(self, x):
        out = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        out = F.max_pool2d(F.relu(self.conv2(out)), kernel_size=2, stride=2)
        out = F.relu(self.conv3(out))
        out = out.reshape([-1, 3*64*64])
        out = self.linear1(out)
        out = self.linear2(out)
        return out
    
    def cancer_classifier_function():
        import torchvision
        model = torchvision.models.resnet50(pretrained=True).cuda()
        return model


if __name__ == "__main__":
    batch = torch.zeros(64, 3, 256, 256)
    model = cancer_classifier()
    output = model(batch)
    print(output.size())