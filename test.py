import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw
from complex_corner_detection_model import ComplexCornerDetectionModel  # Assuming your model definition is in a separate file

# Define the model architecture
model = ComplexCornerDetectionModel()  # Assuming your model architecture is defined in ComplexCornerDetectionModel
model_path = '/home/eyerov/workspace/corner_detector/corner_detector_model.pth'

# Load the saved state dictionary
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Define transformation for the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Process all images in the test folder
test_folder = '/home/eyerov/workspace/corner_detector/screenshots/test/img/'

for filename in os.listdir(test_folder):
    if filename.endswith('.png'):
        # Load and preprocess the test image
        test_image_path = os.path.join(test_folder, filename)
        test_image = Image.open(test_image_path).convert('RGB')
        test_image_tensor = transform(test_image).unsqueeze(0).to(device)

        # Make predictions
        with torch.no_grad():
            predicted_coords = model(test_image_tensor)

        # Mark the predicted point with a red circle
        predicted_coords = predicted_coords.squeeze().cpu().numpy()
        marked_image = test_image.copy()
        draw = ImageDraw.Draw(marked_image)
        draw.ellipse((predicted_coords[0] - 5, predicted_coords[1] - 5,
                      predicted_coords[0] + 5, predicted_coords[1] + 5), outline='red')

        # Show the marked image
        marked_image.show()

print("All images processed and displayed with predictions.")

