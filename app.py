
# Loading libraries 

import streamlit as st

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# Define new uploaded image preprocessing

class ValidationDataset(Dataset):
    def __init__(self, upload_img, transform=None):
        #self.val_dir = [os.path.join(val_dir, img) for img in os.listdir(val_dir)]
        self.images = Image.open(upload_img).convert('L')
        self.transform = transform
    
    def __len__(self):
        #return len(self.images)
        return 1
    

    def __getitem__(self, idx):
        #image = Image.open(self.images[idx]).convert('L')

        if self.transform:
            image = self.transform(self.images)
        return image


# Define new data transforms

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])



# Model parameters
batch_size = 32
learning_rate = 0.001
num_epochs = 50

# define device type whether GPU or CPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define CNN prediction model class
class PlumeCNN(nn.Module):
    def __init__(self):
        super(PlumeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.pool(self.bn1(nn.functional.relu(self.conv1(x))))
        x = self.pool(self.bn2(nn.functional.relu(self.conv2(x))))
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = nn.functional.sigmoid(self.fc2(x))
        return x



# Load the saved model and putting in eval mode
model = PlumeCNN().to(device)

if device == "cuda": 
    model.load_state_dict(torch.load("/content/model_Epoch_50_AUC_90.pt"))
else: 
    model.load_state_dict(torch.load("/content/model_Epoch_50_AUC_90.pt", map_location=torch.device('cpu')))

model.eval()


# -----------------streamlit calls start here ------------------------------------------

st.title('Methane Classifier')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploaded',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 0

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file is not None:

    #if save_uploaded_file(uploaded_file):
        # display the file
        #display_image = Image.open(uploaded_file).convert('L')

        # new image dataloader
        #image_path = str(os.path.join(uploaded_file.name))
        #destination_empty_directory = "/content/empty_directory"

        # Create the destination directory if it doesn't exist
        #os.makedirs(destination_empty_directory, exist_ok=True)

        # Copy the image to the destination directory
        #shutil.copy2(image_path, destination_empty_directory)

        val_dataset = ValidationDataset(uploaded_file,  transform=test_transforms)
        val_loader = DataLoader(val_dataset, batch_size= len(val_dataset) , shuffle=False)

        # Iterate over each image in the folder
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)

                # If sigmoid_output >= 0.5, then positive, else negative

                # change to np-------
                output = model(images).squeeze(0).detach().numpy()
                st.text(f'Likelihood of Methane Plum Presence: {np.round(output[0],4)*100} %')

                # printing model output class
                predictions = np.round(output)
                st.text("prediction: "+ str(predictions))

                # Print the classification result for the image
                st.text(f'The image {"has a methane plume" if predictions==1 else "does not have a methane plume"}')


        # *****START CODE for grad cam
        
        use_cuda = False

        #  Get your intermediate layer
        target_layers = [model.conv1, model.conv2]

        input_tensor = images

        # Note: input_tensor can be a batch tensor with several images!
        # target_category = [ClassifierOutputTarget(0)]
        target_category = None

        # *****END CODE

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=target_category, eigen_smooth=True)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(np.array(images.squeeze(0).permute(1,2,0),np.float32), grayscale_cam, use_rgb=True)
        # st.image(image, caption='Sunrise by the mountains')


