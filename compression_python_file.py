#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
from PIL import Image


# In[2]:


from ipywidgets import interact, interactive, interact_manual
import ipywidgets as widgets
from IPython.display import display


# In[3]:


# replace images with the image you want to compress
images = {
    "Lion": np.asarray(Image.open('animal2.jpg'))
}


# In[4]:


def show_images(img_name):
    'It will show image in widgets'
    print("Loading...")
    plt.title("Close this plot to open compressed image...")
    plt.imshow(images[img_name])
    plt.axis('off')
    plt.show()
    


# In[5]:


show_images('Lion')


# In[6]:


compressed_image = None


# In[7]:

    
    
def compress_image(img_name, k):
    print("processing...")
    global compressed_image
    img = images[img_name]
    
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    
    print("compressing...")
    ur,sr,vr = svd(r, full_matrices=False)
    ug,sg,vg = svd(g, full_matrices=False)
    ub,sb,vb = svd(b, full_matrices=False)
    rr = np.dot(ur[:,:k],np.dot(np.diag(sr[:k]), vr[:k,:]))
    rg = np.dot(ug[:,:k],np.dot(np.diag(sg[:k]), vg[:k,:]))
    rb = np.dot(ub[:,:k],np.dot(np.diag(sb[:k]), vb[:k,:]))
    
    print("arranging...")
    rimg = np.zeros(img.shape)
    rimg[:,:,0] = rr
    rimg[:,:,1] = rg
    rimg[:,:,2] = rb
    
    for ind1, row in enumerate(rimg):
        for ind2, col in enumerate(row):
            for ind3, value in enumerate(col):
                if value < 0:
                    rimg[ind1,ind2,ind3] = abs(value)
                if value > 255:
                    rimg[ind1,ind2,ind3] = 255

    compressed_image = rimg.astype(np.uint8)
    plt.title("Image Name: "+img_name+"\n")
    plt.imshow(compressed_image)
    plt.axis('off')
    plt.show()
    compressed_image = Image.fromarray(compressed_image)
    


# In[8]:


compress_image("Lion", 15)


# In[9]:
compressed_image.save("compressed_animal1.jpg")




