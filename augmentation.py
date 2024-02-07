import os
import sys
import cv2
import numpy as np
import random

# Set the paths for the original images and labels
images_folder = sys.argv[1]
labels_folder = sys.argv[2]

# Create a new folder for augmented images and labels
augmented_folder = "/home/eyerov/Documents/meera/screenshots/augmented_data"
os.makedirs(augmented_folder, exist_ok=True)

# Function to randomly move pixels within the specified range
def random_pixel_shift(image, max_shift):
    rows, cols, _ = image.shape
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)


    #matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    #shifted_image = cv2.warpAffine(image, matrix, (cols, rows))
    height=rows+ np.abs(dy)
    width=cols+  np.abs(dx)
    shifted_image = np.zeros((height,width,3), np.uint8)

    starty=0
    startx=0
    if dy>0:
        starty=dy
        endy=height
    else:
        starty=0
        endy=rows
    if dx>0:
        startx=dx
        endx=width
    else:
        startx=0
        endx=cols


    shifted_image[starty:endy,startx:endx]=image

    return shifted_image,startx,starty

# Augment the images
total_augmented_images = 300
max_pixel_shift = 500

# Get a list of image file names
image_files = os.listdir(images_folder)
cv2.namedWindow("augmented image",cv2.WINDOW_NORMAL) 

for i in range(total_augmented_images):
    # Randomly select an original image and its corresponding label
    random_index = random.randint(0, len(image_files) - 1)
    selected_image = image_files[random_index]
    selected_label = selected_image.replace(".png", ".txt")

    # Load the image
    image_path = os.path.join(images_folder, selected_image)
    original_image = cv2.imread(image_path)

    # Randomly move pixels
    augmented_image, startx, starty = random_pixel_shift(original_image, max_pixel_shift)

    # Save the augmented image
    augmented_image_path = os.path.join(augmented_folder, f"{i + 1}.png")
    cv2.imwrite(augmented_image_path, augmented_image)

    # Read the original label file
    with open(os.path.join(labels_folder, selected_label), 'r') as original_label_file:
        original_label_content = original_label_file.read()

    # Update the label content based on pixel shifts
    lines = original_label_content.split('\n')
    updated_lines = []

    for line in lines:
        
        # Extract original X and Y values
        original_x, original_y = map(int,line.split(","))
        
        # Update X and Y values based on pixel shifts
        updated_x = original_x + startx
        updated_y = original_y + starty
        cv2.circle(augmented_image, (updated_x,updated_y), radius=3, color=(0, 0, 255), thickness=1)

        # Append the updated line to the list
        updated_lines.append(f" {updated_x}, {updated_y}")
    
    cv2.imshow("augmented image",augmented_image)
    cv2.resizeWindow("augmented image", 1920, 1080) 
    cv2.waitKey(0)

    # Create a new filename for the augmented label
    augmented_label_filename = f"{i + 1}.txt"

    # Create the path for the augmented label
    augmented_label_path = os.path.join(augmented_folder, augmented_label_filename)

    # Write the modified content to the new file
    with open(augmented_label_path, 'w') as augmented_label_file:
        augmented_label_file.write('\n'.join(updated_lines))

print("Augmentation completed.")

