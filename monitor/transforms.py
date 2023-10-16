
import sys
import numpy as np
import scipy.stats as st
from scipy.signal import convolve2d
from PIL import Image, ImageEnhance



# ----------------------------------------------------------------------------------------------
# Test Verification Transform Functions
# ----------------------------------------------------------------------------------------------
# Apply transform to an image; Used by test-based verification technique.
#
# Args:
#     image   (np.array)       - the image to transform
#     epsilon (float)          - amount of transform to apply to image
#     ...                      - optional keyword args passed by 'transform_args'
#
# Returns:
#     (np.array) - the transformed image
# ----------------------------------------------------------------------------------------------

def haze(image:np.array, epsilon:float) -> np.array:
    '''Applies haze transform to an image

    Args:
        image (np.array): The input image
        epsilon (float): amount of transform

    Returns:
        [np.array]: image with haze
    '''    
    fog = np.ones_like(image)
    fog[:, :, 0] *= 1.0  # red
    fog[:, :, 1] *= 1.0  # green
    fog[:, :, 2] *= 1.0  # blue
    return (1-epsilon) * image[:, :, :] + epsilon * fog[:, :, :]

def increase_contrast(image:np.array, epsilon:float, tg_min:float=0.0, tg_max:float=1.0) -> np.array:
    '''Increases the contrast of the input image

    Args:
        image (np.array): The input image
        epsilon (float): Amount of transform
        tg_min (float, optional): Min value of image scaling. Defaults to 0.0.
        tg_max (float, optional): Max value of image scaling. Defaults to 1.0.

    Returns:
        np.array: The transformed image
    '''
    # this is a hack to prevent div by zero
    if epsilon >= 1.0:
        epsilon = 0.99999
    # This is the max and minimum value in the picture originally
    sc_min = 0.5*epsilon
    sc_max = 1 - sc_min
    output = (image - sc_min) * (tg_max - tg_min) / (sc_max - sc_min) + tg_min
    return np.clip(output, 0, 1)

def gaussianblureps(image:np.array, epsilon:float, kernelSize:int=17, scaling:int=20) -> np.array:
    '''Applies gaussian blur transform to input image

    Args:
        image (np.array): The input image
        epsilon (float): Amount of transform
        kernelSize (int, optional): The kernel size. Defaults to 17.
        scaling (int, optional): Scaling of the blur transform. Defaults to 20.

    Returns:
        np.array: The transformed image
    '''    
    image = image.copy()
    nsig = (0.01-scaling)*epsilon + scaling
    x = np.linspace(-nsig, nsig, kernelSize+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    kernel =  kern2d/kern2d.sum()

    for i in range(3):
        image[:, :, i] = convolve2d(image[:, :, i], kernel, mode='same', boundary='symm')
    return image

'''def changelight(image:np.array, epsilon:float)->np.array:
    Applies change of light input image

       Args:
           image (np.array): The input image
           epsilon (float): Amount of transform

       Returns:
           np.array: The transformed image
    
    image = image.copy()
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    img_enhancer = ImageEnhance.Brightness(image)
    image = img_enhancer.enhance(epsilon)
    image = np.array(image)
    return image
'''

test_transforms = {
    'haze': {'fn': haze, 'args': dict()},
    'contrast': {'fn': increase_contrast, 'args': dict()},
    'blur': {'fn': gaussianblureps, 'args': dict(kernelSize=17, scaling=20)}
    }
