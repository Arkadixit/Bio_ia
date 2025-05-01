import torch
import torchvision.transforms as transforms
from cnn import Net
import os
from PIL import Image

# Define Path to model and classes
PATH = 'model/model_20250430_154030_18'
classes = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}
class_names = ['Glioma', 'Meningioma','No Tumor','Pituitary']  # Names of classes in order of index

# Define transformation for from scratch
transform = transforms.Compose([
        transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

# Load the model
net = Net()
net.load_state_dict(torch.load(PATH))
device = torch.device('cpu')
net.to(device)
net.eval()  # Set the model to evaluation mode

# Statistics
correct = 0
total = 0

# Function to test a single image
def test_single_image(image_path):
    global correct, total
    image = Image.open(image_path).convert('RGB')  # Load the image with PIL
    image_tensor = transform(image).unsqueeze(0).to(device)  # Apply the transformation and add a batch dimension

    with torch.no_grad():  # No need to compute gradients
        output = net(image_tensor)
        _, predicted = torch.max(output, 1)  # Get the index of the predicted class
        predicted_class = classes[predicted.item()]

    # Get the actual class from the folder name
    actual_class = os.path.basename(os.path.dirname(image_path))

    # Increment the total and correct count
    total += 1
    if predicted_class == actual_class:
        correct += 1

# Iterate over the entire test folder
test_dir = 'Test/Testing'
for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):    # Filter for images
            image_path = os.path.join(root, file)
            test_single_image(image_path)

# Accuracy
if total > 0:
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f} %')
else:
    print("No images to test.")