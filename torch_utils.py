import io
import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms 
from PIL import Image

# load model

model = torchvision.models.resnet50(weights=None)
model.fc = nn.Linear(in_features=2048, out_features=10, bias=False)


in_channels = 3
num_classes = 10
PATH = "pretrained_resNet50_cifar10.pth"
model.load_state_dict(torch.load(PATH, map_location=torch.device("cpu")))
model.eval()

# image -> tensor
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Resize((28,28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.49139968, 0.48215827 ,0.44653124), 
                                        (0.24703233, 0.24348505, 0.26158768))])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# predict
def get_prediction(image_tensor):
    images = image_tensor
    outputs = model(images)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted
