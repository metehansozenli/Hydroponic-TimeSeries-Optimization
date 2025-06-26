import os
import base64
from PIL import Image
from io import BytesIO
import numpy as np

# Create a simple plant icon as PNG
def create_plant_png(output_path):
    # Create a blank image with transparent background
    width, height = 200, 300
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Convert to numpy array for easier manipulation
    img_array = np.array(image)
    
    # Define colors (RGBA)
    stem_color = (74, 143, 41, 255)  # Dark green
    leaf_color = (106, 190, 48, 255)  # Light green
    
    # Draw stem
    stem_width = 10
    stem_x = width // 2 - stem_width // 2
    for x in range(stem_x, stem_x + stem_width):
        for y in range(height - 50, height):
            img_array[y, x] = stem_color
    
    # Draw leaves (simplified)
    # Leaf 1 (bottom right)
    leaf_height = 60
    leaf_width = 80
    leaf_y = height - 100
    
    # Draw a simple leaf shape
    for x in range(width // 2, width // 2 + leaf_width):
        for y in range(leaf_y, leaf_y + leaf_height):
            # Create oval shape
            dx = (x - (width // 2 + leaf_width // 2)) / (leaf_width // 2)
            dy = (y - (leaf_y + leaf_height // 2)) / (leaf_height // 2)
            dist = dx*dx + dy*dy
            if dist < 1.0:
                # Fade alpha at edge
                alpha = int(255 * (1 - dist*dist))
                if alpha > 0:
                    img_array[y, x] = (leaf_color[0], leaf_color[1], leaf_color[2], alpha)
    
    # Leaf 2 (bottom left)
    for x in range(width // 2 - leaf_width, width // 2):
        for y in range(leaf_y, leaf_y + leaf_height):
            dx = (x - (width // 2 - leaf_width // 2)) / (leaf_width // 2)
            dy = (y - (leaf_y + leaf_height // 2)) / (leaf_height // 2)
            dist = dx*dx + dy*dy
            if dist < 1.0:
                alpha = int(255 * (1 - dist*dist))
                if alpha > 0:
                    img_array[y, x] = (leaf_color[0], leaf_color[1], leaf_color[2], alpha)
    
    # Leaf 3 (middle right)
    leaf_y = height - 180
    for x in range(width // 2, width // 2 + leaf_width):
        for y in range(leaf_y, leaf_y + leaf_height):
            dx = (x - (width // 2 + leaf_width // 2)) / (leaf_width // 2)
            dy = (y - (leaf_y + leaf_height // 2)) / (leaf_height // 2)
            dist = dx*dx + dy*dy
            if dist < 1.0:
                alpha = int(255 * (1 - dist*dist))
                if alpha > 0:
                    img_array[y, x] = (leaf_color[0], leaf_color[1], leaf_color[2], alpha)
    
    # Leaf 4 (middle left)
    for x in range(width // 2 - leaf_width, width // 2):
        for y in range(leaf_y, leaf_y + leaf_height):
            dx = (x - (width // 2 - leaf_width // 2)) / (leaf_width // 2)
            dy = (y - (leaf_y + leaf_height // 2)) / (leaf_height // 2)
            dist = dx*dx + dy*dy
            if dist < 1.0:
                alpha = int(255 * (1 - dist*dist))
                if alpha > 0:
                    img_array[y, x] = (leaf_color[0], leaf_color[1], leaf_color[2], alpha)
    
    # Top leaf/bud
    leaf_y = 50
    leaf_height = 70
    leaf_width = 50
    for x in range(width // 2 - leaf_width // 2, width // 2 + leaf_width // 2):
        for y in range(leaf_y, leaf_y + leaf_height):
            dx = (x - width // 2) / (leaf_width // 2)
            dy = (y - (leaf_y + leaf_height // 2)) / (leaf_height // 2)
            dist = dx*dx + dy*dy
            if dist < 1.0:
                alpha = int(255 * (1 - dist*dist))
                if alpha > 0:
                    img_array[y, x] = (leaf_color[0], leaf_color[1], leaf_color[2], alpha)
    
    # Convert back to image and save
    plant_img = Image.fromarray(img_array)
    plant_img.save(output_path)
    print(f"Plant image saved to {output_path}")

if __name__ == "__main__":
    output_path = "static/images/plant.png"
    create_plant_png(output_path)
