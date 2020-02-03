'''
Created on Feb 20, 2019

@author: deckyal
'''


import cv2 
import torch
import torchvision.transforms.functional as F
from PIL import Image,ImageFilter
import utils
import random
import numpy as np 
import numbers
import math
from operator import truediv




class RandomResizedCrop_WL(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img,ldmrk,heatmap=None):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        #Now reposition the landmark 
        #print(i,j,h,w)
        img2 = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        ldmrk2 = ldmrk.copy()
        
        if heatmap is not None : 
            htmp = F.resized_crop(heatmap, i, j, h, w, self.size, self.interpolation)
        
        ldmrk2[:68]-=j
        ldmrk2[68:]-=i
        
        #now rescale
        imHeight,imWidth = img.size
        
        sH = truediv(imHeight,h)
        sW = truediv(imWidth,w)
        
        ldmrk2[:68]*=sW
        ldmrk2[68:]*=sH        
        
        '''if heatmap is None : 
            return img2,ldmrk2
        else :'''
        return img2,ldmrk2,heatmap


class RandomHorizontalFlip_WL_DBL(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    
    def __init__(self, p=0.5):
        
        self.mapping =np.asarray([ 
        [0,16], 
        [1,15], 
        [2,14], 
        [3,13], 
        [4,12], 
        [5,11], 
        [6,10], 
        [7,9], 
        [8,8], 
        [9,7], 
        [10,6], 
        [11,5], 
        [12,4], 
        [13,3], 
        [14,2], 
        [15,1], 
        [16,0],
         
        [17,26], 
        [18,25], 
        [19,24], 
        [20,23], 
        [21,22], 
        [22,21], 
        [23,20], 
        [24,19], 
        [25,18], 
        [26,17], 
        
        [27,27], 
        [28,28], 
        [29,29], 
        [30,30], 
        
        [31,35], 
        [32,34], 
        [33,33], 
        [34,32], 
        [35,31],
         
        [36,45], 
        [37,44], 
        [38,43], 
        [39,42], 
        [40,47], 
        [41,46],
         
        [42,39], 
        [43,38], 
        [44,37], 
        [45,36], 
        [46,41], 
        [47,40],
        
         
        [48,54], 
        [49,53], 
        [50,52], 
        [51,51], 
        [52,50], 
        [53,49], 
        [54,48],
         
        [55,59], 
        [56,58], 
        [57,57], 
        [58,56], 
        [59,55],
         
        [60,64], 
        [61,63], 
        [62,62], 
        [63,61], 
        [64,60],
         
        [65,67], 
        [66,66], 
        [67,65],
        ])
        
        
        self.p = p
        
    def __call__(self, img,ldmrk,heatmap=None,secondImg = None):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            
            img = F.hflip(img)
            secondImg = F.hflip(secondImg)
            
            if heatmap is not None : 
                heatmap = F.hflip(heatmap)
            
            gt_o = ldmrk.copy()
            height, width= img.size
            length = 68
            t_map = self.mapping[:,1]
            
            for i in range(length) : 
                    
                '''if gt_o[i+length] > (height/2) : #y 
                    gt_o[i+length] = height/2 -  (ldmrk[i+length] -(height/2))
                if gt_o[i+length] < (height/2) : #y 
                    gt_o[i+length] = height/2 + ((height/2)-ldmrk[i+length])'''
                      
                if ldmrk[i] > (width/2) :
                    gt_o[t_map[i]] = (width/2) - (ldmrk[i] - (width/2))
                if ldmrk[i] < (width/2) :
                    gt_o[t_map[i]] = (width/2) + ((width/2) - ldmrk[i])    
                gt_o[t_map[i]+length] = ldmrk[i+length]
            
            '''if heatmap is None : 
                return img,gt_o
            else :''' 
            
            return img,gt_o,heatmap,secondImg
            
        '''if heatmap is None : 
            return img,ldmrk
        else :''' 
        return img,ldmrk,heatmap,secondImg

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    
class RandomRotation_WL_DBL(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img,gt,heatmap=None,secondImg = None):
        angle = self.get_params(self.degrees)
        
        img = F.rotate(img, angle, self.resample, self.expand, self.center)
        secondImg = F.rotate(secondImg, angle, self.resample, self.expand, self.center)
        
        if heatmap is not None : 
            heatmap  =  F.rotate(heatmap, angle, self.resample, self.expand, self.center)
        length = 68
        
        info = angle 
        rows,cols = img.size
    
        gt_o = np.array([gt[:length]-(cols/2),gt[length:]-(rows/2)])
        
        theta = np.radians(-info)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s), (s, c)))
        gt_o = np.dot(R,gt_o)
        
        gt_o =  np.concatenate((gt_o[0]+(cols/2),gt_o[1]+(rows/2)),axis = 0)
        '''if heatmap is None : 
            return img,gt_o
        else : '''
        return img,gt_o,heatmap,secondImg
            

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class Occlusion_WL_DBL(object):
    
    def __init__(self, p = 0.5):
        self.p = p 

    def __call__(self, img,gt,heatmap=None,secondImg = None):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            
            output = img.copy()
            gt_o = gt.copy()
            
            #pixels = output.load() # create the pixel map

            lengthW = 0.6
            lengthH = 0.6
            
            s_row = 7
            s_col = 12
            
            imHeight,imWidth = img.size
            
            #Now filling the occluder 
            l_w = imHeight//s_row 
            l_h = imWidth//s_col 
            
            #print(l_w,l_h)
            #print(l_w*lengthH, l_h*lengthW)
            for ix in range(s_row+1):
                for jx in range(s_col+1):
                    #print(ix*l_w,ix*l_w+int(l_w*lengthH) ,jx*l_h,jx*l_h+int(l_h*lengthW))
                    output.paste((255,255,255),[ix*l_w ,jx*l_h,ix*l_w+int(l_w*lengthH) ,jx*l_h+int(l_h*lengthW)])
                    
                    '''inter =  np.full([int(l_w*lengthH),int(l_h*lengthW),3],255)
                    pixels[ix*l_w:ix*l_w+int(l_w*lengthH) ,jx*l_h:jx*l_h+int(l_h*lengthW) ] =inter'''
                    #output.show()
            return output,gt_o,heatmap,secondImg
        return img,gt_o,heatmap,secondImg

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

######################################################################################################

class RandomResizedCrop_WL_DBL(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img,ldmrk,heatmap=None,secondImg = None):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        #Now reposition the landmark 
        #print(i,j,h,w)
        img2 = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        secondImg2 = F.resized_crop(secondImg, i, j, h, w, self.size, self.interpolation)
        
        ldmrk2 = ldmrk.copy()
        
        if heatmap is not None : 
            htmp = F.resized_crop(heatmap, i, j, h, w, self.size, self.interpolation)
        
        ldmrk2[:68]-=j
        ldmrk2[68:]-=i
        
        #now rescale
        imHeight,imWidth = img.size
        
        sH = truediv(imHeight,h)
        sW = truediv(imWidth,w)
        
        ldmrk2[:68]*=sW
        ldmrk2[68:]*=sH        
        
        '''if heatmap is None : 
            return img2,ldmrk2
        else :'''
        return img2,ldmrk2,heatmap,secondImg2


class RandomHorizontalFlip_WL(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    
    def __init__(self, p=0.5):
        
        self.mapping =np.asarray([ 
        [0,16], 
        [1,15], 
        [2,14], 
        [3,13], 
        [4,12], 
        [5,11], 
        [6,10], 
        [7,9], 
        [8,8], 
        [9,7], 
        [10,6], 
        [11,5], 
        [12,4], 
        [13,3], 
        [14,2], 
        [15,1], 
        [16,0],
         
        [17,26], 
        [18,25], 
        [19,24], 
        [20,23], 
        [21,22], 
        [22,21], 
        [23,20], 
        [24,19], 
        [25,18], 
        [26,17], 
        
        [27,27], 
        [28,28], 
        [29,29], 
        [30,30], 
        
        [31,35], 
        [32,34], 
        [33,33], 
        [34,32], 
        [35,31],
         
        [36,45], 
        [37,44], 
        [38,43], 
        [39,42], 
        [40,47], 
        [41,46],
         
        [42,39], 
        [43,38], 
        [44,37], 
        [45,36], 
        [46,41], 
        [47,40],
        
         
        [48,54], 
        [49,53], 
        [50,52], 
        [51,51], 
        [52,50], 
        [53,49], 
        [54,48],
         
        [55,59], 
        [56,58], 
        [57,57], 
        [58,56], 
        [59,55],
         
        [60,64], 
        [61,63], 
        [62,62], 
        [63,61], 
        [64,60],
         
        [65,67], 
        [66,66], 
        [67,65],
        ])
        
        
        self.p = p
        
    def __call__(self, img,ldmrk,heatmap=None):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img = F.hflip(img)
            if heatmap is not None : 
                heatmap = F.hflip(heatmap)
            
            gt_o = ldmrk.copy()
            height, width= img.size
            length = 68
            t_map = self.mapping[:,1]
            
            for i in range(length) : 
                    
                '''if gt_o[i+length] > (height/2) : #y 
                    gt_o[i+length] = height/2 -  (ldmrk[i+length] -(height/2))
                if gt_o[i+length] < (height/2) : #y 
                    gt_o[i+length] = height/2 + ((height/2)-ldmrk[i+length])'''
                      
                if ldmrk[i] > (width/2) :
                    gt_o[t_map[i]] = (width/2) - (ldmrk[i] - (width/2))
                if ldmrk[i] < (width/2) :
                    gt_o[t_map[i]] = (width/2) + ((width/2) - ldmrk[i])    
                gt_o[t_map[i]+length] = ldmrk[i+length]
            
            '''if heatmap is None : 
                return img,gt_o
            else :''' 
            
            return img,gt_o,heatmap
            
        '''if heatmap is None : 
            return img,ldmrk
        else :''' 
        return img,ldmrk,heatmap

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
    
class RandomRotation_WL(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img,gt,heatmap=None):
        angle = self.get_params(self.degrees)
        img = F.rotate(img, angle, self.resample, self.expand, self.center)
        if heatmap is not None : 
            heatmap  =  F.rotate(heatmap, angle, self.resample, self.expand, self.center)
        length = 68
        
        info = angle 
        rows,cols = img.size
    
        gt_o = np.array([gt[:length]-(cols/2),gt[length:]-(rows/2)])
        
        theta = np.radians(-info)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s), (s, c)))
        gt_o = np.dot(R,gt_o)
        
        gt_o =  np.concatenate((gt_o[0]+(cols/2),gt_o[1]+(rows/2)),axis = 0)
        '''if heatmap is None : 
            return img,gt_o
        else : '''
        return img,gt_o,heatmap
            

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class Occlusion_WL(object):
    
    def __init__(self, p = 0.5):
        self.p = p 

    def __call__(self, img,gt,heatmap=None):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            
            output = img.copy()
            gt_o = gt.copy()
            
            #pixels = output.load() # create the pixel map

            lengthW = 0.6
            lengthH = 0.6
            
            s_row = 7
            s_col = 12
            
            imHeight,imWidth = img.size
            
            #Now filling the occluder 
            l_w = imHeight//s_row 
            l_h = imWidth//s_col 
            
            #print(l_w,l_h)
            #print(l_w*lengthH, l_h*lengthW)
            for ix in range(s_row+1):
                for jx in range(s_col+1):
                    #print(ix*l_w,ix*l_w+int(l_w*lengthH) ,jx*l_h,jx*l_h+int(l_h*lengthW))
                    output.paste((255,255,255),[ix*l_w ,jx*l_h,ix*l_w+int(l_w*lengthH) ,jx*l_h+int(l_h*lengthW)])
                    
                    '''inter =  np.full([int(l_w*lengthH),int(l_h*lengthW),3],255)
                    pixels[ix*l_w:ix*l_w+int(l_w*lengthH) ,jx*l_h:jx*l_h+int(l_h*lengthW) ] =inter'''
                    #output.show()
            return output,gt_o,heatmap
        return img,gt_o,heatmap

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class GeneralNoise_WL(object):
    
    def __init__(self, p = 0.5):
        self.p = p
        self.noiseParamList = np.asarray([[0,0,0],[1,2,3],[1,3,5],[.001,.005,.01],[0.75,1.25,1.5],[0,0,0]])
        #self.noiseParamList = np.asarray([[0,0,0],[1,2,3],[1,3,5],[.001,.005,.01],[.8,.5,.2],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]
        #self.noiseParamList =np.asarray([[0,0,0],[2,3,4],[2,4,6],[.005,.01,.05],[.5,.2,.1],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]
        
    def __call__(self, img,gt,nt,nl):
        if random.random() < self.p:
            gt_o = gt.copy()
            output = img.copy()
            
            #print('oy')
            #noiseType = np.random.randint(0,5)
            #noiseLevel = np.random.randint(0,self.noiseParam)
            #noiseParam=noiseParamList[noiseType,noiseLevel]
            
            output = utils.generalNoise(img,nt,self.noiseParamList[nt,nl])
            
            return output,gt_o
        return img,gt

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


    
class GeneralNoise(object):
    
    def __init__(self, p = 0.5):
        self.p = p
        self.noiseParamList = np.asarray([[0,0,0],[1,2,3],[1,3,5],[.001,.005,.01],[0.75,1.25,1.5],[0,0,0]])
        #self.noiseParamList = np.asarray([[0,0,0],[1,2,3],[1,3,5],[.001,.005,.01],[.8,.5,.2],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]
        #self.noiseParamList =np.asarray([[0,0,0],[2,3,4],[2,4,6],[.005,.01,.05],[.5,.2,.1],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]
        
    def __call__(self, img,nt,nl):
        if random.random() < self.p:
            output = img.copy()
            
            #print('oy')
            #noiseType = np.random.randint(0,5)
            #noiseLevel = np.random.randint(0,self.noiseParam)
            #noiseParam=noiseParamList[noiseType,noiseLevel]
            
            output = utils.generalNoise(img,nt,self.noiseParamList[nt,nl])
            
            return output
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


##################################################################################################

    
##########################3

class RandomResizedCrop_m(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img,htmp):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation),F.resized_crop(htmp, i, j, h, w, self.size, self.interpolation)

class RandomHorizontalFlip_m(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img,htmp):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img),F.hflip(htmp)
        return img,htmp

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
     
class RandomRotation_m(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img,htmp):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center),F.rotate(htmp, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string



class Occlusion(object):
    
    def __init__(self, p = 0.5):
        self.p = p 

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            
            output = img.copy()
            
            #pixels = output.load() # create the pixel map

            lengthW = 0.5#0.6
            lengthH = 0.5#0.6
            
            s_row = 7
            s_col = 12
            
            imHeight,imWidth = img.size
            
            #Now filling the occluder 
            l_w = imHeight//s_row 
            l_h = imWidth//s_col 
            
            #print(l_w,l_h)
            #print(l_w*lengthH, l_h*lengthW)
            for ix in range(s_row+1):
                for jx in range(s_col+1):
                    #print(ix*l_w,ix*l_w+int(l_w*lengthH) ,jx*l_h,jx*l_h+int(l_h*lengthW))
                    output.paste((255,255,255),[ix*l_w ,jx*l_h,ix*l_w+int(l_w*lengthH) ,jx*l_h+int(l_h*lengthW)])
                    
                    '''inter =  np.full([int(l_w*lengthH),int(l_h*lengthW),3],255)
                    pixels[ix*l_w:ix*l_w+int(l_w*lengthH) ,jx*l_h:jx*l_h+int(l_h*lengthW) ] =inter'''
                    #output.show()
            return output
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
