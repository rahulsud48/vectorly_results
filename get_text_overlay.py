### Libraries #########################################
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


class getTextOverlay(object):

    def __init__(self, input_filename, output_filename): 

        self.input_filename = input_filename
        self.output_filename = output_filename

        self.img = self.load_input_file()
        return None


    def load_input_file(self, gray_scale = True):
        img_ = cv2.imread(self.input_filename)
        if gray_scale:
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        return img_


    def view_image(self, text, img):
        cv2.imshow(text, img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return None


    def save_file(self, img):
        cv2.imwrite(self.output_filename, img)
        return True


    def __call__(self, view_image = False):
        # Binary
        ret, self.img = cv2.threshold(self.img, 10,255, cv2.THRESH_BINARY)

        # Dilate and Erode      
        kernel_3x3 = np.ones((3,3), np.uint8)      
        self.img = cv2.dilate(self.img, kernel_3x3, iterations = 4)     
        self.img = cv2.erode(self.img, kernel_3x3, iterations = 4) 

        #Sharpen
        sharpen_kernel = np.array([[-1,-1,-1], [-1,20,-1], [-1,-1,-1]])
        self.img = cv2.filter2D(self.img, -1, sharpen_kernel)
        
        #View
        if view_image:
            self.view_image("Final Output Image with Text Overlay", self.img)
        self.save_file(self.img)
    
        return True

if __name__ == '__main__':
    getTextOverlay(input_filename = 'simpsons_frame0.png', output_filename = 'simpons_text.png')(view_image=True)