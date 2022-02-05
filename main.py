import random
import numpy as np
import cv2
from matplotlib import pyplot as plt

def myImFilter(image, param):
    # Get the horizontal and vertical size of the image
    img_size_x = image.shape[1]
    img_size_y = image.shape[0]
    # Create a 3x3 mean kernel
    kernel = np.ones([3, 3], dtype=int)
    kernel = kernel / 11
    # Get the horizontal and vertical size of the kernel
    kernel_size_x = kernel.shape[1]
    kernel_size_y = kernel.shape[0]
    # Create the filtered image
    filtered_img_size_x = img_size_x + kernel_size_x - 1
    filtered_img_size_y = img_size_y + kernel_size_y - 1
    filtered_img = np.zeros([img_size_y, img_size_x])
    # Helping variables for resizing the final image, in order to exclude the distorted borders
    tmp1 = filtered_img_size_y - img_size_y
    tmp2 = filtered_img_size_x - img_size_x
    # Calculate the convolution of the image and the 3x3 kernel
    if param == 'mean':
        temp1 = 0
        for i1 in range(1, img_size_y-1):
            for j1 in range(1, img_size_x-1):
                filtered_img[i1, j1] = temp1
                temp1 = 0
                for i2 in range(0, 3):
                    q = 1
                    if i2 == 1:
                        p = 0
                    elif i2 == 2:
                        p = -1
                    else:
                        p = 1
                    for j2 in range(0, 3):
                        if j2 == 2:
                            q = -1
                        temp1 = temp1 + kernel[i2, j2] * image[i1 - p, j1 - q]
                        q = 0
        filtered_img = filtered_img[tmp1:-tmp1, tmp2:-tmp2]
        filtered_img = filtered_img.astype(np.uint8)
        cv2.imwrite('blurred.png', filtered_img)
    # Finding the median of the pixels in the contextually 3x3 area
    # Replacing the center pixel by the median
    elif param == 'median':
        temp2 = []
        for i1 in range(1, img_size_y - 1):
            for j1 in range(1, img_size_x - 1):
                if not j1 == 1:
                    temp2 = sorted(temp2)
                    filtered_img[i1, j1] = temp2[4]
                    temp2.clear()
                for i2 in range(0, 3):
                    q = 1
                    if i2 == 1:
                        p = 0
                    elif i2 == 2:
                        p = -1
                    else:
                        p = 1
                    for j2 in range(0, 3):
                        if j2 == 2:
                            q = -1
                        temp2.append(image[i1 - p, j1 - q])
                        q = 0
        filtered_img = filtered_img.astype(np.uint8)
        cv2.imwrite('median.png', filtered_img)

def myImNoise(image, param):
    if param == 'gaussian':
        # Define the values of mean, var and sigma, based on our preference
        mean = 0
        var = 0.05
        sigma = var ** 0.5
        # Constract the gaussian array based on the values of mean and sigma as well as the image shape
        gaussian = np.random.normal(mean, sigma, image.shape)
        # Create the resulting image by using the clip function
        noisy_image = np.clip((image + gaussian), 0, 1)
        plt.imshow(noisy_image)
        plt.title('Gaussian Noise')
        plt.show()
        return 1
    elif param == 'saltandpepper':
        row = image.shape[0]
        col = image.shape[1]
        # Pick randomly some image pixels from the range 0 - 1000
        # Color them black
        num_of_pixels = random.randint(5000, 10000)
        for i in range(num_of_pixels):
            y_axis = random.randint(0, row - 1)
            x_axis = random.randint(0, col - 1)
            image[y_axis][x_axis] = 0
        # Pick randomly some image pixels from the range 0 - 1000
        # Color them white
        num_of_pixels = random.randint(5000, 10000)
        for i in range(num_of_pixels):
            y_axis = random.randint(0, row - 1)
            x_axis = random.randint(0, col - 1)
            image[y_axis][x_axis] = 255
        cv2.imshow("sp.jpeg", img)
        cv2.waitKey(0);
        return 1
def myConv2(A, B):
    # Get the horizontal and vertical size of A and B
    A_size_x = A.shape[1]
    A_size_y = A.shape[0]
    B_size_x = B.shape[1]
    B_size_y = B.shape[0]
    # Below we will set the size of an array named "C"
    # The size is defined by the sizes of the array "A" and the kernel "B"
    # The size of the array "A" should be the same after the convolution
    # In order to achieve this, we should have an array C large enough
    # The reason behind this practise is that the convolution decreases the array's size
    # For example, assuming the size of the array "A" is 7x6 and the size of the kernel "B" is 5x5
    # The size of the array "C" should be 11x10 (which will decrease later on, to match the size of array "A")
    # The size of the output array is given by the formula: C_size = A_size + B_size -1
    C_size_x = A_size_x + B_size_x - 1
    C_size_y = A_size_y + B_size_y - 1
    # Create the C array full of zeros
    C = np.zeros((C_size_y,C_size_x))
    # Some helping variables for the iteration of the arrays A, B and C
    tmp1 = C_size_y - A_size_y
    tmp1= tmp1 / 2
    tmp1 = int(tmp1);
    tmp2 = C_size_x - A_size_x
    tmp2 = tmp2 / 2
    tmp2 = int(tmp2);
    # Filling the center of the array "C", with the values of the array "A"
    for i in range(tmp1, C_size_y - tmp1, 1):
        for j in range(tmp2, C_size_x - tmp2, 1):
            C[i, j] = A[i-tmp1, j-tmp2]
    print(C)
    # Iterating the array "C"
    for m in range(C_size_y - (C_size_y - A_size_y)):
        tmp3 = m
        for n in range(C_size_x - (C_size_x - A_size_x)):
            tmp4 = n
        # Iterating the kernel "B"
            for i in range(B_size_y):
                for j in range(B_size_x):
                    # Calculating the convolution
                    C[tmp3, tmp4] = C[tmp3, tmp4] + B[i, j] * C[tmp3+i, tmp4+j]
    print("\n", C)
    # After the convolution the output array should have the same size of the input array
    C = C[tmp1:-tmp1, tmp2:-tmp2]
    return C

# This is a 2D matrix
A = np.array([[25, 100, 75, 49, 130, 1],
              [25, 100, 75, 49, 130, 1],
              [50, 80, 0, 70, 100, 2],
              [5, 10, 20, 30, 0, 3],
              [60, 50, 12, 24, 32, 4],
              [37, 53, 55, 21, 90, 5],
              [140, 17, 0, 23, 22, 6]])
# This is a 2D kernel
B = np.array([[0, 0, 1, 1, 1],
              [0, 1, 1, 1, 0],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 0, 0],
              [0, 1, 1, 0, 0]])
# Calling the "myCon2(A, B)" function, to calculate the convolution
C = myConv2(A, B)
print("A: \n", A)
print("B: \n", B)
print("C: \n", C)
# Calling the "myImNoise(A,'param')" function after opening the image using OpenCV
# [...,::-1] is used for converting the "BGR" to the "RGB" format
# We divide by 255, for the pixel values to range between 0 and 1
img = cv2.imread('image.jpeg')[..., ::-1]/255.0
myImNoise(img, 'gaussian')
# "Salt-and-pepper" applies only to grayscale images
img = cv2.imread('image.jpeg', cv2.IMREAD_GRAYSCALE)
myImNoise(img, 'saltandpepper')
# Applying the median filter to the image that we had applied the salt-and-pepper filter previously
# In order to eliminate the previous noise
img = cv2.imread('sp.png', 0)
myImFilter(img, 'median')
# Applying the mean filter to the image that we had applied the median filter previously
img = cv2.imread('median.png', 0)
myImFilter(img, 'mean')




