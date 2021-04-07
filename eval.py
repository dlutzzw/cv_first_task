import torch.nn as nn
from torchvision import models
import torch
import cv2
from torchvision import datasets, transforms

num_classes = 10
image_path = './test_images/IMG_1133.JPG'
model_PATH ='./last.pt'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn = models.resnet50(pretrained=True)
'''
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    cnn = nn.DataParallel(cnn)
cnn.to(device)
'''
cnn.fc = nn.Linear(2048, num_classes)
# cnn = cnn.to(device)
image = cv2.imread(image_path)
image=cv2.resize(image,(224,224))
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])
# tensor=torch.from_numpy(np.asarray(image)).permute(2,0,1).float()/255.0
tensor = test_transforms(image)
tensor=tensor.reshape((1,3,224,224))
# tensor=tensor.to(device)
cnn.load_state_dict(torch.load(model_PATH),False)
cnn.eval()
outputs = cnn(tensor)
_, predicted = torch.max(outputs.data, 1)
print('the result is:',predicted.item())
