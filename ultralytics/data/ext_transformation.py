import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

def lbp(img_dict):
  # Check if the input is a PIL Image, if so, convert to grayscale
  img = img_dict['img']
  print(img)
  if isinstance(img, Image.Image):
      img_gray = img.convert('L')
      img_gray = np.asarray(img_gray)
  elif isinstance(img, np.ndarray):
      # If it's a NumPy array, check if it's RGB and convert to grayscale
      if len(img.shape) == 3 and img.shape[2] == 3:  # RGB image
          img_gray = Image.fromarray(img).convert('L')
          img_gray = np.asarray(img_gray)
      elif len(img.shape) == 2:  # Already grayscale
          img_gray = img
      else:
          raise ValueError("Unsupported NumPy array shape.")
  else:
      raise ValueError("Input must be a PIL Image or NumPy array.")
  
  Rows, Cols = img_gray.shape
  
  # Convert the grayscale image to a PyTorch tensor and move it to the GPU
  x = torch.from_numpy(img_gray.reshape(1, Rows, Cols).astype(np.uint8)).to('cuda')
  
  # Pad the image to accommodate the 3x3 mask
  x = F.pad(input=x, pad=[1, 1, 1, 1], mode='constant')
  M, N = x.shape[1], x.shape[2]
  
  # Extract the 3x3 neighborhoods
  y00, y01, y02 = x[:, 0:M-2, 0:N-2], x[:, 0:M-2, 1:N-1], x[:, 0:M-2, 2:N]
  y10, y11, y12 = x[:, 1:M-1, 0:N-2], x[:, 1:M-1, 1:N-1], x[:, 1:M-1, 2:N]
  y20, y21, y22 = x[:, 2:M, 0:N-2], x[:, 2:M, 1:N-1], x[:, 2:M, 2:N]
  
  # Pre-create weights for the bitwise comparisons
  weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device='cuda', dtype=torch.uint8)
  
  # Perform the LBP calculation using bitwise operations
  comparisons = torch.stack([
      torch.ge(y01, y11),
      torch.ge(y02, y11),
      torch.ge(y12, y11),
      torch.ge(y22, y11),
      torch.ge(y21, y11),
      torch.ge(y20, y11),
      torch.ge(y10, y11),
      torch.ge(y00, y11)
  ], dim=0).type(torch.uint8)
  
  # Calculate the LBP value by summing the weighted bit comparisons
  lbp_value = torch.sum(comparisons * weights.view(-1, 1, 1), dim=0)
  img_dict['img'] = lbp_value.type(torch.FloatTensor)
  return img_dict


class ext_transformer:
  def __init__(self, func = lbp):
    print("External Transformer has been called")
    self.func = func
  def __call__(self, data):
      # Apply the transformation function to the input data
      return transforms.Lambda(self.func)(data)


"""transforms_lbp = [
    transforms.Lambda(lbp),
    transforms.ToPILImage(),
    transforms.ToTensor(),
]"""


