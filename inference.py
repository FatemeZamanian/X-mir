import argparse
import time
import torch
from torchvision import transforms
import cv2
import numpy as np

import model


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--image", type=str, default='test1.png')
parser.add_argument("--model", type=str, default='densenet121')
args = parser.parse_args()

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])

device=args.device
if args.model=='densenet121':
    m=model.DenseNet121()
elif args.model=='resnet50':
    m=model.ResNet50()
else:
    print('undefined model!!!!')
    exit()


m.load_state_dict(torch.load('weight.pth',map_location=args.device))
m.eval()

img=cv2.imread(args.image)
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.resize(img,(28,28))
tensor=transform(img).unsqueeze(0)
tensor=tensor.to(device)
start=time.time()
pred=m(tensor)
print(f"{time.time()-start} sec")
# pred=pred.argmax()
pred=pred.cpu().detach().numpy()
print(pred)
# out=np.argmax(pred)
# print(out)