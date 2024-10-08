import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

def lbp(img_dict):
  # Check if the input is a PIL Image, if so, convert to grayscale
  img = img_dict['img']
  
  # Define transform
  transform = transforms.Grayscale()
   
  # Convert the image to grayscale
  img_gray = transform(img)
  
  
  
  Rows, Cols = img_gray.shape[1:]
  
  # Convert the grayscale image to a PyTorch tensor and move it to the GPU
  #pad image for 3x3 mask size
  x = F.pad(input=img_gray, pad = [1, 1, 1, 1], mode='constant')
  b=x.shape
  M=b[1]
  N=b[2]
  
  y=x
  #select elements within 3x3 mask 
  # y00  y01  y02
  # y10  y11  y12
  # y20  y21  y22
  
  y00=y[:,0:M-2, 0:N-2]
  y01=y[:,0:M-2, 1:N-1]
  y02=y[:,0:M-2, 2:N  ]
  #     
  y10=y[:,1:M-1, 0:N-2]
  y11=y[:,1:M-1, 1:N-1]
  y12=y[:,1:M-1, 2:N  ]
  #
  y20=y[:,2:M, 0:N-2]
  y21=y[:,2:M, 1:N-1]
  y22=y[:,2:M, 2:N ]      
  
     
  
  # Comparisons 
  # 1 ---------------------------------
  bit=torch.ge(y01,y11)
  tmp=torch.mul(bit,torch.tensor(1)) 
  
  # 2 ---------------------------------
  bit=torch.ge(y02,y11)
  val=torch.mul(bit,torch.tensor(2))
  val=torch.add(val,tmp)    
  
  # 3 ---------------------------------
  bit=torch.ge(y12,y11)
  tmp=torch.mul(bit,torch.tensor(4))
  val=torch.add(val,tmp)
  
  # 4 --------------------------------- 
  bit=torch.ge(y22,y11)
  tmp=torch.mul(bit,torch.tensor(8))   
  val=torch.add(val,tmp)
  
  # 5 ---------------------------------
  bit=torch.ge(y21,y11)
  tmp=torch.mul(bit,torch.tensor(16))   
  val=torch.add(val,tmp)
  
  # 6 ---------------------------------
  bit=torch.ge(y20,y11)
  tmp=torch.mul(bit,torch.tensor(32))   
  val=torch.add(val,tmp)
  
  # 7 ---------------------------------
  bit=torch.ge(y10,y11)
  tmp=torch.mul(bit,torch.tensor(64))   
  val=torch.add(val,tmp)
  
  # 8 ---------------------------------
  bit=torch.ge(y00,y11)
  tmp=torch.mul(bit,torch.tensor(128))   
  val=torch.add(val,tmp).type('torch.FloatTensor').repeat(3, 1, 1)
  
  
  img_dict['img'] = val
  
  return img_dict


class ext_transformer:
  def __init__(self, func = lbp):
    #print("External Transformer has been called")
    self.func = func
  def __call__(self, data):
      # Apply the transformation function to the input data
      return transforms.Lambda(self.func)(data)


"""transforms_lbp = [
    transforms.Lambda(lbp),
    transforms.ToPILImage(),
    transforms.ToTensor(),
]"""


