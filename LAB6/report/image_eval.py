import torch
from torchvision import transforms
from PIL import Image
import json
import os
from file.evaluator import evaluation_model  


# --- Need!!!  add your img dir & json file --- #

json_file = 'file/new_test.json'
image_dir = '/home/clu98753cs13/Desktop/DL/LAB6/res/images/epoch_460/new_test'  # 圖片命名需為 0.png, 1.png, ..., N.png

#################################################

evaluator = evaluation_model()
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
with open(json_file) as f:
    test_labels = json.load(f)
with open('file/objects.json') as f:
    object_map = json.load(f)

num_classes = len(object_map)
images = []
labels = []

for idx, label_names in enumerate(test_labels):
    img_path = os.path.join(image_dir, f"{idx}.png")
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    images.append(image)

    label_vector = torch.zeros(num_classes)
    for name in label_names:
        label_vector[object_map[name]] = 1.0
    labels.append(label_vector)

images = torch.stack(images)  # shape: (N, 3, 64, 64)
labels = torch.stack(labels)  # shape: (N, 24)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
images = images.to(device)
labels = labels.to(device)

print(json_file + f" accuracy = {evaluator.eval(images, labels):.4f}")
