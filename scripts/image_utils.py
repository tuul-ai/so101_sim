import torch
from PIL import Image
import numpy as np
import inspect
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display # Import for in-notebook rendering
import time
import sys

def reorder_tensor_dimensions(tensor):
    """
    Reorders tensor dimensions from (C, H, W) to (H, W, C) format if necessary.
    
    Args:
        tensor: Input tensor with shape (C, H, W) or (H, W, C)
        
    Returns:
        Tensor with shape (H, W, C)
    """
    if len(tensor.shape) == 3 and tensor.shape[0] == 3:
        # Permute dimensions from (C, H, W) to (H, W, C)
        tensor = tensor.permute(1, 2, 0)
    else:
        assert tensor.shape[2] == 3, "Invalid tensor shape"
    return tensor

def heic_to_pil(heic_path: str) -> Image.Image:
    """Convert a HEIC image to PIL Image"""
    with Image.open(heic_path) as img:
        return img

def tensor_to_pil(tensor: torch.Tensor | np.ndarray) -> Image.Image:
    """Convert a PyTorch tensor or numpy array to PIL Image"""
    # Handle both torch tensors and numpy arrays
    if isinstance(tensor, torch.Tensor):
        if len(tensor.shape) == 4:  # Remove batch dimension if present
            tensor = tensor[0]
        
        # Convert from CxHxW to HxWxC
        if len(tensor.shape) == 3 and tensor.shape[0] == 3:
            img_array = tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img_array = tensor.cpu().numpy()
    else:  # numpy array
        if len(tensor.shape) == 4:  # Remove batch dimension if present
            tensor = tensor[0]
        
        # Convert from CxHxW to HxWxC
        if len(tensor.shape) == 3 and tensor.shape[0] == 3:
            img_array = np.transpose(tensor, (1, 2, 0))
        else:
            img_array = tensor
    
    # Scale to 0-255 if needed
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

# Add to a new cell
def prepare_image_for_gemini(tensor_image):
    """Convert tensor image to PIL format for Gemini Vision API"""
    # Handle different tensor formats
    if len(tensor_image.shape) == 4:  # If it has a batch dimension
        tensor_image = tensor_image[0]
        
    # Convert to numpy and ensure proper range
    img_np = tensor_image.permute(1, 2, 0).cpu().numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    # Convert to PIL
    pil_image = Image.fromarray(img_np)
    return pil_image

def display_images(*images, titles=None, figsize=(15, 10), max_cols=4):
    """
    Display multiple images in a grid layout with automatic titles based on variable names.
    
    Args:
        *images: Variable number of images (2-8) as PIL Images, numpy arrays, or torch tensors
        titles: Optional list of custom titles. If None, variable names will be used
        figsize: Figure size as (width, height) tuple
        max_cols: Maximum number of columns in the grid
    
    Example:
        phone_im = tensor_to_pil(frame_data['observation.images.phone'])
        laptop_im = tensor_to_pil(frame_data['observation.images.laptop'])
        display_images(phone_im, laptop_im)  # Will use "phone_im" and "laptop_im" as titles
    """
    # Get variable names from the calling frame
    if titles is None:
        frame = inspect.currentframe().f_back
        calling_vars = frame.f_locals.items()
        titles = []
        
        # Find variable names by comparing their values with our images
        for img in images:
            found = False
            for var_name, var_val in calling_vars:
                if var_val is img:
                    titles.append(var_name)
                    found = True
                    break
            if not found:
                titles.append(f"Image {len(titles)+1}")
    
    # Validate number of images
    num_images = len(images)
    if num_images < 2 or num_images > 8:
        raise ValueError("Number of images must be between 2 and 8")
    
    # Determine grid layout
    cols = min(num_images, max_cols)
    rows = (num_images + cols - 1) // cols
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Display each image
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        
        # Convert image to displayable format if needed
        if isinstance(img, torch.Tensor):
            # Convert torch tensor to numpy array
            img_np = img.cpu().numpy()
            # Handle different tensor formats
            if len(img_np.shape) == 3:
                if img_np.shape[0] == 3:  # CHW format
                    img_np = np.transpose(img_np, (1, 2, 0))
            plt.imshow(img_np)
        elif isinstance(img, np.ndarray):
            plt.imshow(img)
        elif isinstance(img, Image.Image):
            plt.imshow(np.array(img))
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        
        plt.axis('off')
        plt.title(title)
    
    plt.tight_layout()
    plt.show()

class RealtimeRenderer:
    """
    A class for real-time rendering of robot observations.
    Can render in a standalone pop-up window or directly within a Jupyter notebook output.
    Press 'q' to close the window (standalone mode).
    """
    def __init__(self, camera_key: str = 'overhead_cam', title: str = "Robot View", interval: float = 0.03, in_notebook: bool = False):
        self.camera_key = camera_key
        self.title = title
        self.interval = interval
        self.in_notebook = in_notebook
        self.frame_count = 0
        
        if not self.in_notebook:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)
            self.img_display = None

    def update(self, obs: dict):
        """
        Updates the display with the current observation.
        """
        if self.camera_key not in obs:
            print(f"Observation does not contain '{self.camera_key}' image.")
            return

        current_img = tensor_to_pil(obs[self.camera_key])

        if self.in_notebook:
            display.clear_output(wait=True) # Clear previous output
            plt.imshow(current_img)
            plt.title(f"{self.title} - Frame {self.frame_count}")
            plt.axis('off')
            display.display(plt.gcf()) # Display current figure
            time.sleep(self.interval) # Pause for pacing
        else:
            if self.img_display is None:
                self.img_display = self.ax.imshow(current_img)
            else:
                self.img_display.set_data(current_img)
            self.ax.set_title(f"{self.title} - Frame {self.frame_count}")
            self.ax.axis('off')  # Hide axes
            plt.draw()
            plt.pause(self.interval)

            # Check for 'q' key press to close the window
            if plt.waitforbuttonpress(timeout=0.001):
                if plt.gcf().canvas.manager.key_press_handler.key == 'q':
                    print("'q' pressed. Closing display window.")
                    self.close()
                    sys.exit() # Exit the script
        
        self.frame_count += 1

    def close(self):
        """
        Closes the rendering window or clears notebook output.
        """
        if self.in_notebook:
            display.clear_output(wait=True)
            print("Notebook rendering cleared.")
        else:
            plt.close(self.fig)
            print("Rendering window closed.")
