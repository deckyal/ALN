'''
Created on Oct 29, 2019

@author: deckyal
'''
import numpy as np
import re
import cv2
from operator import truediv
import matplotlib 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
#import tensorflow as tf
import random
#from config import *
from scipy.integrate.quadrature import simps
import math
from scipy.stats import multivariate_normal
import os
from random import randint
import glob
from scipy.integrate import simps
from PIL import Image,ImageFilter,ImageEnhance
from math import isnan
import torch 

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
        
    if classname.find('Conv2d') != -1:
        #print('applying mother fucker')
        n = m.in_channels
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y,y)

def update_lr_ind(opt, lr):                     
    """Decay learning rates of the generator and discriminator."""
    for param_group in opt.param_groups:
        param_group['lr'] = lr

def update_lr( lr, optimizer):                     
    """Decay learning rates of the generator and discriminator."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def reset_grad(optimizer):
    """Reset the gradient buffers."""
    optimizer.zero_grad()
   
def denorm( x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def gradient_penalty( y, x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def label2onehot( labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def print_network( model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))
    
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    

def OpenCVtoPIL(opencv_image = None) : 
    cv2_im = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    return pil_im

def PILtoOpenCV(pil_image = None):
    open_cv_image = np.array(pil_image) 
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

def checkDirMake(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

#print (convertToOneHot(np.array([7]),num_classes = 8))


def generalNoise(tImageB = None,noiseType = 0,noiseParam = 0):
    if noiseType == 0 : 
        tImageB = tImageB
    elif noiseType == 1: #downsample 
        oWidth, oHeight = tImageB.size
        for i in range(int(noiseParam)) :#Scale down (/2) blurLevel times 
            width, height = tImageB.size
            tImageB = tImageB.resize((width//2,height//2))
            #print(tImageB.size)
        tImageB = tImageB.resize((oWidth,oHeight))
    elif noiseType == 2 : #Gaussian blur
        tImageB = tImageB.filter(ImageFilter.GaussianBlur(noiseParam))
    elif noiseType == 3 : #Gaussian noise 
        #tImageB = addNoise(tImageB)
        #convert to opencv 
        opencvImage = cv2.cvtColor(np.array(tImageB), cv2.COLOR_RGB2BGR)
        
        #print(opencvImage)
        opencvImage = addNoise(opencvImage,var=noiseParam)
        pilImage =  cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)
        #tImageB = Image.fromarray(random_noise(opencvImage))
        tImageB = Image.fromarray(pilImage)
    elif noiseType == 4 : #Brightness :
        #tImageB.show()
        
        e = ImageEnhance.Brightness(tImageB)
        tImageB = e.enhance(noiseParam)
         
        #tImageB.show()
        
        #opencvImage = cv2.cvtColor(np.array(tImageB), cv2.COLOR_RGB2BGR)
        
        #print('before',opencvImage)
        #opencvImage = np.asarray(opencvImage*noiseParam,dtype=np.int32)
        #print(opencvImage.shape)
        #test = opencvImage.astype(np.float64)*noiseParam
        
        '''print(opencvImage)
        print(test)'''
        #opencvImage = test.astype(np.uint8)
        
        '''for i in range(0,opencvImage.shape[0]):
            for j in range(0,opencvImage.shape[1]):
                #print(opencvImage[i,j],noiseParam)
                opencvImage[i,j,0] = round(opencvImage[i,j,0] * noiseParam)
                opencvImage[i,j,1] = round(opencvImage[i,j,1] * noiseParam)
                opencvImage[i,j,2] = round(opencvImage[i,j,2] * noiseParam)
                #print(opencvImage[i,j])
        #print('after',opencvImage)
        '''
        #pilImage =  cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)
        #tImageB = Image.fromarray(pilImage)
    elif noiseType == 5 : 
        tImageB = tImageB.convert('L')
        np_img = np.array(tImageB, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        tImageB = Image.fromarray(np_img, 'RGB')
        
    return tImageB

def addNoise (image,noise_type="gauss",var = .01):
    """
    Generate noise to a given Image based on required noise type
    
    Input parameters:
        image: ndarray (input image data. It will be converted to float)
        
        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row,col,ch= image.shape
    if noise_type == "gauss":       
        mean = 0.0
        #var = 0.001
        sigma = var**0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        #print(gauss)
        noisy = image + gauss*255
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.09
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
    else:
        return image


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def imageLandmarking(img, ldmrk, isPIL = True,inputGT = None):
    if isPIL : 
        #convert the image to the opencv format
        print(img)
        theImage = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    else : 
        theImage = img.copy()
    
    for y in range(68) : 
        cv2.circle(theImage,(int(ldmrk[y]),int(ldmrk[y+68])),2,(0,255,0) )
        if inputGT is not None : 
            cv2.circle(theImage,(int(inputGT[y]),int(inputGT[y+68])),2,(0,0,255) )
    
    return theImage
    

def unnormalizedToCV(input = [],customNormalize = None):
    output = []
    #unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    for i in range(input.shape[0]) : 
        #Unnormalized it, convert to numpy and multiple by 255.
        if customNormalize is None :  
            theImage = (unorm(input[i]).numpy()*255).transpose((1,2,0))
        else : 
            theImage = (input[i].numpy().transpose(1,2,0) + customNormalize)
        
        #Then transpose to be height,width,channel, to Int and BGR formate 
        theImage = cv2.cvtColor(theImage.astype(np.uint8 ),cv2.COLOR_RGB2BGR)
        output.append(theImage)
    
    return output


def unnormalizedAndLandmark(input = [], inputPred = [],inputGT = None,customNormalize = None,ldmarkNumber=68):
    #input is unnormalized [batch_size, channel, height, width] tensor from pytorch
    #inputGT is [batch_size, 136] tensor landmarks 
    #Output is [batch_size, height,width,channel] BGR, 0-255 Intensities opencv list of landmarked image
    output = []
    inputPred = inputPred.numpy()
    if inputGT is not None : 
        inputGT = inputGT.numpy()
    
    #unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    for i in range(inputPred.shape[0]) : 
        #Unnormalized it, convert to numpy and multiple by 255.
        if customNormalize is None :  
            theImage = (unorm(input[i]).numpy()*255).transpose((1,2,0))
        else : 
            theImage = (input[i].numpy().transpose(1,2,0) + customNormalize)
        
        #Then transpose to be height,width,channel, to Int and BGR formate 
        theImage = cv2.cvtColor(theImage.astype(np.uint8 ),cv2.COLOR_RGB2BGR)
        
        
        #Now landmark it. 
        for y in range(ldmarkNumber) :
            cv2.circle(theImage,(int(scale(inputPred[i,y])),int(scale(inputPred[i,y+ldmarkNumber]))),2,(0,255,0) )
            if inputGT is not None : 
                cv2.circle(theImage,(int(scale(inputGT[i,y])),int(scale(inputGT[i,y+ldmarkNumber]))),2,(0,0,255) )
        
        output.append(theImage)
    
    return output

def scale(input):
    if input > 99999 :
        input = 99999
    elif input < -99999 : 
        input = -99999
    elif isnan(input): 
        input = 0
    return input 

def plotImages(input = [], title = None, n_row = 4, n_col = 4, fromOpenCV = True,fileName = None,show=False):
    #Function to plot row,col image. 
    #Given [n_row*n_col,image_width, image_height, channel] input
    #tittle [n_row*n_col]
    
    fig = plt.figure()
     
    for i in range(n_row * n_col) :
        #print('the i ',i)
        ax = fig.add_subplot(n_row,n_col,i+1)
        if title is not None : 
            ax.set_title(title[i])
        
        if fromOpenCV :  
            plt.imshow(cv2.cvtColor(input[i],cv2.COLOR_BGR2RGB))
        else :     
            plt.imshow(input[i])
    
    if fileName : 
        plt.savefig(fileName)
    
    if show : 
        plt.show()
    

def calc_bb_IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou



def read_kp_file(filename,flatten = False):
     
    x = []
    
    if ('pts' in filename) :
        with open(filename) as file:
            data2 = [re.split(r'\t+',l.strip()) for l in file]
        for i in range(len(data2)) :
            if(i not in [0,1,2,len(data2)-1]):
                x.append([ float(j) for j in data2[i][0].split()] )
    if flatten : 
        return np.asarray(x).flatten('F')
    else : 
        return np.asarray(x)

def read_kp_file_text(filename):
     
    x = []
    y = []
    
    if ('txt' in filename) :
        with open(filename) as file:
            data2 = [re.split(r'\t+',l.strip()) for l in file]
        
        #print(data2)
        tmp = 0
        for i in range(len(data2)) :
            if(i not in [0,1,len(data2)-1]):
                for j in data2[i][0].split() :
                    if tmp % 2 == 0 : 
                        x.append(float(j))
                    else : 
                        y.append(float(j))
                    tmp+=1
                    
    return np.concatenate((x,y))


def read_bb_file(filename):
     
    x = []
    
    if ('pts' in filename) :
        with open(filename) as file:
            data2 = [re.split(r'\t+',l.strip()) for l in file]
        for i in range(len(data2)) :
            if(i not in [0,1,2,len(data2)-1]):
                x.append([ float(j) for j in data2[i][0].split()] )
    return np.asarray(x)


def errCalcNoisy(catTesting = 1, localize = False, t_dir = "300W-Test/01_Indoor/",name='Re3A',is3D=False,ext = ".txt",makeFlag = False):
    catTesting = catTesting;
    l_error = []
    
    if localize is False :#image dir 
        if is3D: 
            dir = curDir + 'images/300VW-Test_M/cat'+str(catTesting)+'/'
        else : 
            dir = curDir + 'images/300VW-Test/cat'+str(catTesting)+'/'
    else :  
        dir = curDir + t_dir +'/'
        
    list_txt = glob.glob1(dir,"*"+ext)
    
    for x in list_txt: 
        print(("Opening " +dir+x))
        
        file = open(dir+x)
        for line in file : 
            #print float(line)
            l_error.append(float(line))
            
        file.close()
    all_err = np.array(l_error)
    
    if makeFlag : 
        list_txt = glob.glob1(dir,"*"+ext)
        l_tr = []
        l_d = []
        
        for x in list_txt: 
            print(("Opening " +dir+x))
            
            file = open(dir+x)
            for line in file : 
                data = [ float(j) for j in line.split()] 
                #print(data)
                l_tr.append(float(data[1]))
                l_d.append(float(data[0]))
                
            file.close()
        
    
    if localize is False :    
        fileName = "src/result_compared/cat"+str(catTesting)+"/"
        aboveT = makeErrTxt(all_err,fileName= fileName+name+".txt",threshold = .08,lim = 1.1005)
        
        if makeFlag : 
            l_tr = np.asarray(l_tr);
            l_d = np.asarray(l_d);
            
            f = open(curDir+fileName+"flag.txt",'w')
            
            am_r = truediv(len(l_tr[np.where(l_tr > 0 )]),len(l_tr));
            am_d = truediv(len(l_d[np.where(l_d == 0 )]),len(l_d));
            
            f.write("%.4f %.4f\n" % (am_r,am_d));    
            f.close()
            
        print(("Above T ",name," : "+str(aboveT)))
        plot_results(catTesting,resFolder= 'src/result_compared/cat'+str(catTesting),addition=[name],is3D=is3D)
    else : #error dir 
        arrName = ['src/result_compared/300W/Indoor','src/result_compared/300W/Outdoor','src/result_compared/300W/InOut']
        aboveT = makeErrTxt(all_err,fileName= arrName[catTesting]+"/"+name+".txt",threshold = .08,lim =.35005, step = .0005)
        print(("Above T ",name," : "+str(aboveT)))
        plot_results(catTesting+4,resFolder= arrName[catTesting],addition=[name],is3D=is3D)
    
    return all_err
    #print(("All error : "+str(all_err)))



def errCalc(catTesting = 1, localize = False, t_dir = "300W-Test/01_Indoor/",name='Re3A',is3D=False,ext = ".txt",makeFlag = False):
    catTesting = catTesting;
    l_error = []
    
    if localize is False :#image dir 
        if is3D: 
            dir = curDir + 'images/300VW-Test_M/cat'+str(catTesting)+'/'
        else : 
            dir = curDir + 'images/300VW-Test/cat'+str(catTesting)+'/'
    else :  
        dir = curDir + t_dir +'/'
        
    list_txt = glob.glob1(dir,"*"+ext)
    
    for x in list_txt: 
        print(("Opening " +dir+x))
        
        file = open(dir+x)
        for line in file : 
            #print float(line)
            l_error.append(float(line))
            
        file.close()
    all_err = np.array(l_error)
    
    if makeFlag : 
        list_txt = glob.glob1(dir,"*"+ext)
        l_tr = []
        l_d = []
        
        for x in list_txt: 
            print(("Opening " +dir+x))
            
            file = open(dir+x)
            for line in file : 
                data = [ float(j) for j in line.split()] 
                #print(data)
                l_tr.append(float(data[1]))
                l_d.append(float(data[0]))
                
            file.close()
        
    
    if localize is False :    
        fileName = "src/result_compared/cat"+str(catTesting)+"/"
        aboveT = makeErrTxt(all_err,fileName= fileName+name+".txt",threshold = .08,lim = 1.1005)
        
        if makeFlag : 
            l_tr = np.asarray(l_tr);
            l_d = np.asarray(l_d);
            
            f = open(curDir+fileName+"flag.txt",'w')
            
            am_r = truediv(len(l_tr[np.where(l_tr > 0 )]),len(l_tr));
            am_d = truediv(len(l_d[np.where(l_d == 0 )]),len(l_d));
            
            f.write("%.4f %.4f\n" % (am_r,am_d));    
            f.close()
            
        print(("Above T ",name," : "+str(aboveT)))
        plot_results(catTesting,resFolder= 'src/result_compared/cat'+str(catTesting),addition=[name],is3D=is3D)
    else : #error dir 
        arrName = ['src/result_compared/300W/Indoor','src/result_compared/300W/Outdoor','src/result_compared/300W/InOut']
        aboveT = makeErrTxt(all_err,fileName= arrName[catTesting]+"/"+name+".txt",threshold = .08,lim =.35005, step = .0005)
        print(("Above T ",name," : "+str(aboveT)))
        plot_results(catTesting+4,resFolder= arrName[catTesting],addition=[name],is3D=is3D)
    
    return all_err
    #print(("All error : "+str(all_err)))

def makeErrTxt(error,fileName = 'result_compared/Decky.txt',threshold = .08,lim = .35005, step = .0001):
    print("Making errr")
    bin = np.arange(0,lim,step)#0.35005,0.0005), 300vw 1.1005
    
    #res = np.array([len(bin)])
    
    #creating the file 
    f = open(curDir+fileName,'w')    
    f.write('300W Challenge 2013 Result\n');
    f.write('Participant: Decky.\n');
    f.write('-----------------------------------------------------------\n');
    f.write('Bin 68_all 68_indoor 68_outdoor 51_all 51_indoor 51_outdoor\n');

    for i in range(len(bin)) : 
        err = truediv(len(error[np.where(error < bin[i])]),len(error))
        f.write("%.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" % (bin[i],err, err,err, err, err, err));    
    f.close()
    err_above = truediv(len(error[np.where(error > threshold )]),len(error));
    print((error[np.where(error > threshold )]))
    return err_above 


    
def plot_results(version, resFolder = 'result_compared',x_limit=0.08, colors=None, markers=None, linewidth=3,
                 fontsize=12, figure_size=(11, 6),addition = None,is3D = False,All = False):
    """
    Method that generates the 300W Faces In-The-Wild Challenge (300-W) results
    in the form of Cumulative Error Distributions (CED) curves. The function
    renders the indoor, outdoor and indoor + outdoor results based on both 68
    and 51 landmark points in 6 different figures.

    Please cite:
    C. Sagonas, E. Antonakos, G. Tzimiropoulos, S. Zafeiriou, M. Pantic. "300
    Faces In-The-Wild Challenge: Database and Results", Image and Vision
    Computing, 2015.
    
    Parameters
    ----------
    version : 1 or 2
        The version of the 300W challenge to use. If 1, then the reported
        results are the ones of the first conduct of the competition in the
        ICCV workshop 2013. If 2, then the reported results are the ones of
        the second conduct of the competition in the IMAVIS Special Issue 2015.
    x_limit : float, optional
        The maximum value of the horizontal axis with the errors.
    colors : list of colors or None, optional
        The colors of the lines. If a list is provided, a value must be
        specified for each curve, thus it must have the same length as the
        number of plotted curves. If None, then the colours are linearly sampled
        from the jet colormap. Some example colour values are:

                'r', 'g', 'b', 'c', 'm', 'k', 'w', 'orange', 'pink', etc.
                or
                (3, ) ndarray with RGB values

    linewidth : float, optional
        The width of the rendered lines.
    fontsize : int, optional
        The font size that is applied on the axes and the legend.
    figure_size : (float, float) or None, optional
        The size of the figure in inches.
    """
    
    if not is3D :
        title = "300VW 2D "
    else : 
        title = "300VW 3DA-2D "  
    # Check version
    if version == 1:
        participants = ['Dlssvm_Cfss', 'MD_CFSS', 'Mdnet_DlibERT', 'Meem_Cfss', 'Spot_Cfss']
        if not All : 
            title += 'category 1' 
    elif version == 2:
        participants = ['ccot_cfss', 'MD_CFSS', 'spot_cfss', 'srdcf_cfss']
        if not All : 
            title += 'category 2'
    elif version == 3:
        participants = ['ccot_cfss', 'MD_CFSS', 'meem_cfss', 'srdcf_cfss','staple_cfss']
        if not All : 
            title += 'category 3'
    elif version in [4,5,6]:
        if is3D : 
            print("in if")
            participants=[]
            l_participants = ['Re3A_3D','Re3A_C_3D','FA_3D']
            for z in l_participants : 
                if z not in participants : 
                    participants.append(z)
        else: 
            print("in else")
            participants = ['Baltrusaitis', 'Hasan',  'Jaiswal','Milborrow','Yan','Zhou']
            #participants = []
            participants.append('Re3A')
            participants.append('Re3A_C')
            participants.append('FA')
            
        arrName = ['Indoor','Outdoor','Indoor + Outdoor']
        if not All : 
            title = arrName[version - 4]
    else:
        raise ValueError('version must be either 1 or 2')
    
    if All : 
        title += " All Category "
    #participants = []
    if version in [1,2,3]:
        participants = []
        mapName = []
        if is3D :  
            #participants = []
            #participants.append('Re3A_3D')
            #participants.append('Re3A_C_3D')
            participants.append('RT_MT_3D')
            participants.append('RT_2_3D')
            participants.append('RT_4_3D')
            participants.append('RT_8_3D')
            participants.append('RT_16_3D')
            participants.append('RT_32_3D')
            
            participants.append('FA_MD_3D')
            participants.append('FA_MT_3D')
            participants.append('3DFFA_MD_3D')
            participants.append('3DFFA_MT_3D')
            
            mapName.append('FLL_MT_3D')
            mapName.append('CRCN_2_3D')
            mapName.append('CRCN_4_3D')
            mapName.append('CRCN_8_3D')
            mapName.append('CRCN_16_3D')
            mapName.append('CRCN_32_3D')
            
            mapName.append('FA_MD_3D')
            mapName.append('FA_MT_3D')
            mapName.append('3DFFA_MD_3D')
            mapName.append('3DFFA_MT_3D')
            
            colors = ['b','red','orange','yellow','yellow','yellow','green','brown','k','purple']
        else: 
            participants.append('RT_MT')
            participants.append('RT_2')
            participants.append('RT_4')
            participants.append('RT_8')
            participants.append('RT_16')
            participants.append('RT_32')
            participants.append('YANG')
            participants.append('MD_CFSS')
            participants.append('ME_CFSS')
            
            mapName.append('FLL_MT')
            mapName.append('CRCN_2')
            mapName.append('CRCN_4')
            mapName.append('CRCN_8')
            mapName.append('CRCN_16')
            mapName.append('CRCN_32')
            mapName.append('YANG')
            mapName.append('MD_CFSS')
            mapName.append('ME_CFSS')
            #participants.append('FA_MD')
            #participants.append('FA_MT')
            colors = ['b','red','orange','yellow','yellow','yellow','g','brown','k']
            
            #participants.append('Re3A')
            #participants.append('Re3A_C')
            #participants.append('FA_MD')
        #participants = []
    if addition is not None :
        for i in addition :  
            if i not in participants : 
                participants.append(i)
        
    # Initialize lists
    ced68 = []
    ced68_indoor = []
    ced68_outdoor = []
    ced51 = []
    ced51_indoor = []
    ced51_outdoor = []
    legend_entries = []

    # Load results
    results_folder = curDir+resFolder
    
    i = 0
    for f in participants:
        # Read file
        if 'Re3A' in f  or version in  [1,2,3,6]:
            index = 1
        elif version == 4 :#indoor 
            index = 2;
        elif version == 5 :#outdoor
            index = 3;
            
        filename = f + '.txt'
        tmp = np.loadtxt(str(Path(results_folder) / filename), skiprows=4)
        print(str(Path(results_folder) / filename))
        # Get CED values
        bins = tmp[:, 0]
        ced68.append(tmp[:, index])
        
        '''ced68_indoor.append(tmp[:, 2])
        ced68_outdoor.append(tmp[:, 3])
        ced51.append(tmp[:, 4])
        ced51_indoor.append(tmp[:, 5])
        ced51_outdoor.append(tmp[:, 6])'''
        # Update legend entries
        legend_entries.append(mapName[i])# + ' et al.')
        i+=1
    
    print(bins,x_limit)
    
    if version < 4 : 
        idx = [x[0] for x in np.where(bins==x_limit+.0001)] #.0810
    else : 
        idx = [x[0] for x in np.where(bins==x_limit+.005)] #.0810
        
    real_bins =  bins[:idx[0]]
    
    
    print(idx,real_bins)
    
    for i in range(len(ced68)) : 
        
        real_ced =  ced68[i][:idx[0]]
        #print(real_ced)
        #AUC = str(round(simps(real_ced,real_bins) * (1/x_limit),3))
        AUC = str(round(simps(real_ced,real_bins) * (1/x_limit),5))
        FR = str(round(1. - real_ced[-1],5)) #[-3]
        
        #print(real_bins[-1])
        
        #print(legend_entries[i] + " : "+str(simps(real_ced,real_bins) * (1/x_limit)))
        
        print(legend_entries[i] + " : " +AUC+" FR : "+FR) 
        #legend_entries[i]+=" [AUC : "+AUC+"]"#+"] [FR : "+FR+"]"
        
        #plt.plot(real_bins,real_ced)
        #plt.show()
    # 68 points, indoor + outdoor    
    _plot_curves(bins, ced68, legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size)
    
    '''# 68 points, indoor
    title = 'Indoor, 68 points'
    _plot_curves(bins, ced68_indoor, legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size)
    # 68 points, outdoor
    title = 'Outdoor, 68 points'
    _plot_curves(bins, ced68_outdoor, legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size)
    # 51 points, indoor + outdoor
    title = 'Indoor + Outdoor, 51 points'
    _plot_curves(bins, ced51, legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size)
    # 51 points, indoor
    title = 'Indoor, 51 points'
    _plot_curves(bins, ced51_indoor, legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size)
    # 51 points, outdoor
    title = 'Outdoor, 51 points'
    _plot_curves(bins, ced51_outdoor, legend_entries, title, x_limit=x_limit,
                 colors=colors, linewidth=linewidth, fontsize=fontsize,
                 figure_size=figure_size)'''
    
    
def _plot_curves(bins, ced_values, legend_entries, title, x_limit=0.08,
                 colors=None, linewidth=3, fontsize=12, figure_size=None):
    # number of curves
    n_curves = len(ced_values)
    
    # if no colors are provided, sample them from the jet colormap
    if colors is None:
        cm = plt.get_cmap('jet')
        colors = [cm(1.*i/n_curves)[:3] for i in range(n_curves)]
        
    # plot all curves
    fig = plt.figure()
    ax = plt.gca()
    for i, y in enumerate(ced_values):
        plt.plot(bins, y, color=colors[i],
                 linestyle='-',
                 linewidth=linewidth, 
                 label=legend_entries[i])
        #print bins.shape, y.shape
    # legend
    ax.legend(prop={'size': fontsize}, loc=4)
    
    # axes
    for l in (ax.get_xticklabels() + ax.get_yticklabels()):
        l.set_fontsize(fontsize)
    ax.set_xlabel('Normalized Point-to-Point Error', fontsize=fontsize)
    ax.set_ylabel('Images Proportion', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    # set axes limits
    ax.set_xlim([0., x_limit])
    ax.set_ylim([0., 1.])
    ax.set_yticks(np.arange(0., 1.1, 0.1))
    
    # grid
    plt.grid('on', linestyle='--', linewidth=0.5)
    
    # figure size
    if figure_size is not None:
        fig.set_size_inches(np.asarray(figure_size))
    
    plt.show()

def make_heatmap(image_name,t_image,add,y_batch,isRandom = True,percent_heatmap = .1,percent_heatmap_e = .05):
    
    tBase = os.path.basename(image_name)
    tName,tExt = os.path.splitext(tBase)
    theDir =  os.path.dirname(image_name)+"/../heatmap-"+add+"/"
    
    if not os.path.exists(theDir):
        os.makedirs(theDir)
        
    fName =theDir+tName+".npy"
    
    #print(fName)
    try : 
        b_channel,g_channel,r_channel = t_image[:,:,0],t_image[:,:,1],t_image[:,:,2]
    except : 
        print(image_name)
    
    if os.path.isfile(fName) and isRandom: 
        newChannel = np.load(fName)
        print("using saved npy")
    else :    
        print("make npy "+add)
        newChannel = b_channel.copy(); newChannel[:] = 0
        y_t = y_batch
        
        if isRandom : 
            t0,t1,t2,t3 = get_bb(y_t[0:int(n_o//2)], y_t[int(n_o//2):],68,False,
                                 random.uniform( -.25, .25 ),
                                 random.uniform( -.25, .25 ),
                                 random.uniform( -.25, .25 ),
                                 random.uniform( -.25, .25 ),
                                 random.uniform( -.25, .25 ))
        else : 
            t0,t1,t2,t3 = get_bb(y_t[0:int(n_o//2)], y_t[int(n_o//2):],68,False)
        #print(t0,t1,t2,t3)
        
        l_cd,rv = get_list_heatmap(0,None,t2-t0,t3-t1,percent_heatmap)
        l_cd_e,rv_e = get_list_heatmap(0,None,t2-t0,t3-t1,percent_heatmap_e)
        
        height, width,_ = t_image.shape
        
        scaler = 255/np.max(rv)
        #addOne = randint(0,2),addTwo = randint(0,2)
        for iter in range(68) :
            #print(height,width)
            if random: 
                ix,iy = int(y_t[iter]),int(y_t[iter+68])
            else : 
                ix,iy = int(y_t[iter])+randint(0,2),int(y_t[iter+68])+randint(0,2)
            #Now drawing given the center
            if iter in range(36,48): 
                l_cd_t = l_cd_e
                rv_t = rv_e
            else : 
                l_cd_t = l_cd
                rv_t = rv
            
            for iter2 in range(len(l_cd_t)) : 
                value = int(rv_t[iter2]*scaler)
                if newChannel[inBound(iy+l_cd_t[iter2][0],0,height-1), inBound(ix + l_cd_t[iter2][1],0,width-1)] < value : 
                    newChannel[inBound(iy+l_cd_t[iter2][0],0,height-1), inBound(ix + l_cd_t[iter2][1],0,width-1)] = int(rv_t[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)
        
        #np.save(fName,newChannel)
    
    return newChannel

def get_enlarged_bb(the_kp = None, div_x = 2, div_y = 2, images = None,is_bb = False, displacement = 0, 
                    displacementxy = None,n_points = 68):
    
    if not is_bb : 
        if displacementxy is not None : 
            t = get_bb(x_list = the_kp[:n_points],y_list = the_kp[n_points:],
                       adding_xmin=displacementxy,adding_xmax=displacementxy,
                       adding_ymin=displacementxy,adding_ymax=displacementxy)
        else : 
            t = get_bb(x_list = the_kp[:n_points],y_list = the_kp[n_points:],length = n_points,adding  = displacement)
    else : 
        t = the_kp
                
    l_x = (t[2]-t[0])/div_x
    l_y = (t[3]-t[1])/div_y
    
    x1 = int(max(t[0] - l_x,0))
    y1 = int(max(t[1] - l_y,0))
    
    x_min = x1; y_min = y1;
    
    #print tImage.shape
    x2 = int(min(t[2] + l_x,images.shape[1]))
    y2 = int(min(t[3] + l_y,images.shape[0]))
    
    return t,l_x,l_y,x1,y1,x_min,y_min,x2,y2

def inBoundN(input,min,max):
    if input < min : 
        return min 
    elif input > max : 
        return max 
    return input

def inBound(input,min,max):
    if input < min : 
        return int(min) 
    elif input > max : 
        return int(max) 
    return int(input)

def inBound_tf(input,min,max):
    if input < min : 
        return int(min) 
    elif input > max : 
        return int(max) 
    return int(input)

def eval(input):
    if input < 0 : 
        return 0
    else :
        return input
    
def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

def initialize_uninitialized_global_variables(sess):
    """
    Only initializes the variables of a TensorFlow session that were not
    already initialized.
    :param sess: the TensorFlow session
    :return:
    """
    # List all global variables
    global_vars = tf.global_variables()

    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]

    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars)) 


def addPadd(im) : 
    #im = cv2.imread("./test-frontal.png")
    height, width, channels =im.shape
    desired_size = np.max(np.array([height,width]))
    
    add_x,add_y = 0,0
    
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    if height > width : #so shift x 
        add_x = left
    else:
        add_y = top 
        
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    #print top,bottom,left,right
    '''cv2.imshow("image", new_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    return new_im,add_x,add_y

def transformation(input, gt, type, info,length = 68 ):
    mapping =[ 
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
        ]
    
    mapping84 =[ 
        [0,32], 
        [1,31], 
        [2,30], 
        [3,29], 
        [4,28], 
        [5,27], 
        [6,26], 
        [7,25], 
        [8,24], 
        [9,23], 
        [10,22], 
        [11,21], 
        [12,20], 
        [13,19], 
        [14,18], 
        [15,17], 
        [16,16],
        [17,15], 
        [18,14], 
        [19,13], 
        [20,12], 
        [21,11], 
        [22,10], 
        [23,9], 
        [24,8], 
        [25,7], 
        [26,6], 
        [27,5], 
        [28,4], 
        [29,3], 
        [30,2], 
        [31,1], 
        [32,0], 
         
        [33,42], 
        [34,41], 
        [35,40], 
        [36,39], 
        [37,38], 
        [38,37], 
        [39,36], 
        [40,35], 
        [41,34], 
        [42,33], 
        
        [43,46], 
        [44,45], 
        [45,44], 
        [46,43], 
        
        [47,51], 
        [48,50], 
        [49,49], 
        [50,48], 
        [51,47],
         
        [52,57], 
        [53,56], 
        [54,55], 
        [55,54], 
        [56,53], 
        [57,52],
         
        [58,63], 
        [59,62], 
        [60,61], 
        [61,60], 
        [62,59], 
        [63,58],
        
         
        [64,70], 
        [65,69], 
        [66,68], 
        [67,67], 
        [68,66], 
        [69,65], 
        [70,64],
         
        [71,75], 
        [72,74], 
        [73,73], 
        [74,72], 
        [75,71],
         
        [76,80], 
        [77,79], 
        [78,78], 
        [79,77], 
        [80,76],
         
        [81,83], 
        [82,82], 
        [83,81],
        ]
    if length > 68: 
        mapping = np.asarray(mapping84)
    else :
        mapping = np.asarray(mapping)
        
    if type == 1 : 
        #print("Flippping") #info is 0,1
        
        gt_o = gt.copy()
        height, width,_ = input.shape
        
        if info == 0 : #vertical 
            #print("Flipping vertically ^v")
            output = cv2.flip(input,0)
            
            for i in range(length) : 
                    
                if gt_o[i+length] > (height/2) : #y 
                    gt_o[i+length] = height/2 -  (gt[i+length] -(height/2))
                if gt_o[i+length] < (height/2) : #y 
                    gt_o[i+length] = height/2 + ((height/2)-gt[i+length])

        elif info == 1 : #horizontal 
            t_map = mapping[:,1]
            
            #gt_o_t = gt.copy()
            
            #print("Flipping Horizontally <- ->  ")
            #return np.fliplr(input)
            output = cv2.flip(input,1)
            
            for i in range(length) : 
                    
                if gt[i] > (width/2) : #x 
                    #gt_o_t[i] = (width/2) - (gt[i] - (width/2))
                    gt_o[t_map[i]] = (width/2) - (gt[i] - (width/2))
                if gt[i] < (width/2) : #x 
                    #gt_o_t[i] = (width/2) + ((width/2) - gt[i])
                    gt_o[t_map[i]] = (width/2) + ((width/2) - gt[i])
                #get the new index 
                #gt_o[t_map[i]] = gt_o_t[i]
                
                gt_o[t_map[i]+length] = gt[i+length]
                    
            #needs to be transformed. 
        
        return [output,gt_o]
    elif type == 2 : 
        #print("Rotate") # info is 1,2,3
        #output = np.rot90(input,info)
        rows,cols,_ = input.shape
    
        M = cv2.getRotationMatrix2D((cols/2,rows/2),info,1)
        output = cv2.warpAffine(input,M,(cols,rows))
        
        gt_o = np.array([gt[:length]-(cols/2),gt[length:]-(rows/2)])
        
        theta = np.radians(-info)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s), (s, c)))
        gt_o = np.dot(R,gt_o)
        
        gt_o =  np.concatenate((gt_o[0]+(cols/2),gt_o[1]+(rows/2)),axis = 0)
        
        '''
        print R.shape, gt_o.shape
        print gt_o.shape'''
        
        
        return [output,gt_o]
        
    elif type == 3 : #info is 0 to 1
        #print("Occlusion")
        
        output = input.copy()
        gt_o = gt.copy()
        
        lengthW = 0.5
        lengthH = 0.4
        
        s_row = 15
        s_col = 7
        
        imHeight,imWidth,_ = input.shape
        
        #Now filling the occluder 
        l_w = imHeight//s_row 
        l_h = imWidth//s_col 
         
        for ix in range(s_row):
            for jx in range(s_col):
                #print ix,jx,l_w,l_h
                #y1:y2, x1:x2
                
                #print(ix*b_size,outerH ,jx*b_size,outerW,'--',outerImgH,',',outerImgW )
                #print(ix*l_w,ix*l_w+l_w ,jx*l_h,jx*l_h+l_h )
                output[ix*l_w:ix*l_w+int(l_w*lengthH) ,jx*l_h:jx*l_h+int(l_h*lengthW) ] = np.full([int(l_w*lengthH),int(l_h*lengthW),3],255)
        
        return [output,gt_o]
    


def calcListNormalizedDistance(pred,gt):
    '''
    input : 
        pred : num_images,num points
        gt : num_images, num points 
    '''
    err = np.zeros(len(pred))
    
    print((pred.shape))
    
    num_points = pred.shape[2]
    
    for i in range(len(pred)) : 
        
        if num_points == 68 : 
            i_d = np.sqrt(np.square(pred[i,36] - gt[i,45]))
        else : 
            i_d = np.sqrt(np.square(pred[i,19] - gt[i,28]))
        
        sum = 0
        for j in range(num_points) : 
            sum += np.sqrt(np.square(pred[i,j]-gt[i,j]))
        
        err[i] = sum/(num_points * i_d)
        
    return err

def calcNormalizedDistance(pred,gt):
    '''
    input : 
        pred : 1,num points
        gt : 1, num points 
    '''
    
    num_points = pred.shape[0]
    #print(num_points)
    
    '''if num_points == 68*2 : 
        i_d = np.sqrt(np.square(pred[36] - gt[45]) + np.square(pred[36+68] - gt[45+68]))
    else : 
        i_d = np.sqrt(np.square(pred[19] - gt[28]) + np.square(pred[19+68] - gt[28+68]))
    '''
    if num_points == 68*2 : 
        i_d = np.sqrt(np.square(gt[36] - gt[45]) + np.square(gt[36+68] - gt[45+68]))
    else : 
        i_d = np.sqrt(np.square(gt[19] - gt[28]) + np.square(gt[19+68] - gt[28+68]))
    
    t_sum = 0
    num_points_norm = num_points//2
    
    for j in range(num_points_norm) : 
        t_sum += np.sqrt(np.square(pred[j]-gt[j])+np.square(pred[j+num_points_norm]-gt[j+num_points_norm]))
    
    err = t_sum/(num_points_norm * i_d)
        
    return err

#assumes p_a and p_b are both positive numbers that sum to 100
def myRand(a, p_a, b, p_b):
    return a if random.uniform(0,100) < p_a else b 


def calcLandmarkErrorListTF(pred,gt):
    
    all_err = []
    batch = pred.get_shape()[0]
    seq = pred.get_shape()[1]
    
    for i in range(batch) :
        for z in range(seq):  
            bb = get_bb_tf(gt[i,z,0:68],gt[i,z,68:])
            
            width = tf.abs(bb[2] - bb[0])
            height = tf.abs(bb[3] - bb[1])
            
            gt_bb = tf.sqrt(tf.square(width) + tf.square(height)) 
            
            num_points = pred.get_shape()[2]
            num_points_norm = num_points//2
            
            sum = []
            for j in range(num_points_norm) : 
                sum.append( tf.sqrt(tf.square(pred[i,z,j]-gt[i,z,j])+tf.square(pred[i,z,j+num_points_norm]-gt[i,z,j+num_points_norm])))
            
            err = tf.divide(tf.stack(sum),gt_bb*num_points_norm)
            
            all_err.append(err)
        
    return tf.reduce_mean(tf.stack(all_err))

def calcLandmarkError(pred,gt): #for 300VW
    '''
    input : 
        pred : 1,num points
        gt : 1, num points 
        
        according to IJCV
        Normalized by bounding boxes
    '''
    
    #print pred,gt
    
    
    num_points = pred.shape[0]
    
    num_points_norm = num_points//2
    
    bb = get_bb(gt[:68],gt[68:])
    
    #print(gt)
    #print(bb)
    
    '''width = np.abs(bb[2] - bb[0])
    height = np.abs(bb[3] - bb[1])
    
    gt_bb = np.sqrt(np.square(width) + np.square(height))
    
    
    print("1 : ",width,height,gt_bb)'''
    
    width = np.abs(bb[2] - bb[0])
    height = np.abs(bb[3] - bb[1])
    
    gt_bb = math.sqrt((width*width) +(height*height))
    
    #print("2 : ",width,height,(width^2) +(height^2),gt_bb)
    '''print(bb) 
    print(gt_bb)
    print("BB : ",gt)
    print("pred : ",pred)'''
    
    '''print(num_points_norm)
    print("BB : ",bb)
    print("GT : ",gt)
    print("PR : ",pred)'''
    #print(num_points)
    
    '''error = np.mean(np.sqrt(np.square(pred-gt)))/gt_bb
    return error''' 
    
    summ = 0
    for j in range(num_points_norm) : 
        #summ += np.sqrt(np.square(pred[j]-gt[j])+np.square(pred[j+num_points_norm]-gt[j+num_points_norm]))
        summ += math.sqrt(((pred[j]-gt[j])*(pred[j]-gt[j])) + ((pred[j+num_points_norm]-gt[j+num_points_norm])*(pred[j+num_points_norm]-gt[j+num_points_norm])))
    #err = summ/(num_points_norm * (gt_bb))
    err = summ/(num_points_norm*gt_bb)
    
        
    return err

def showGates(tg = None, batch_index_to_see = 0, n_to_see = 64, n_neurons = 1024,toShow = False, toSave = False, fileName = "gates.jpg"):
    #Total figure : 1024/64 data per image : 16 row per gate then *6 gate : 96
    
    
    t_f_row = n_neurons/n_to_see
    n_column = 6

    fig = plt.figure()
    
    for p_i in range(t_f_row) :
        
        inputGate =     tg[:,0,batch_index_to_see,p_i * n_to_see:p_i*n_to_see+n_to_see] #all sequence, gate 1, batch 0, 200 neurons
        newInputGate=   tg[:,1,batch_index_to_see,p_i * n_to_see:p_i*n_to_see+n_to_see] #all sequence, gate 2, batch 0, 200 neurons
        forgetGate =    tg[:,2,batch_index_to_see,p_i * n_to_see:p_i*n_to_see+n_to_see] #all sequence, gate 3, batch 0, 200 neurons
        outputGate =    tg[:,3,batch_index_to_see,p_i * n_to_see:p_i*n_to_see+n_to_see] #all sequence, gate 4, batch 0, 200 neurons
        cellState =     tg[:,4,batch_index_to_see,p_i * n_to_see:p_i*n_to_see+n_to_see]
        outputState =   tg[:,5,batch_index_to_see,p_i * n_to_see:p_i*n_to_see+n_to_see]
        
        #print p_i
        ax = fig.add_subplot(t_f_row,n_column,p_i*(n_column) + 1)
        if p_i == 0 : 
            ax.set_title('Input Gate')
        plt.imshow(inputGate,vmin=0,vmax=1)
        '''
        for temp in inputGate : 
            for temp2 in temp : 
                if temp2 < 0 : 
                    print temp2'''
        
        ax = fig.add_subplot(t_f_row,n_column,p_i*(n_column) + 2)
        if p_i == 0 : 
            ax.set_title('New Input Gate')
        plt.imshow(newInputGate,vmin=0,vmax=1)
        
        ax = fig.add_subplot(t_f_row,n_column,p_i*(n_column) + 3)
        if p_i == 0 : 
            ax.set_title('Forget Gate')
        plt.imshow(forgetGate,vmin=0,vmax=1)
        
        ax = fig.add_subplot(t_f_row,n_column,p_i*(n_column) + 4)
        if p_i == 0 : 
            ax.set_title('Output Gate')
        plt.imshow(outputGate,vmin=0,vmax=1)
        
        ax = fig.add_subplot(t_f_row,n_column,p_i*(n_column) + 5)
        if p_i == 0 : 
            ax.set_title('Cell State')
        plt.imshow(cellState,vmin=0,vmax=1)
        
        ax = fig.add_subplot(t_f_row,n_column,p_i*(n_column) + 6)
        if p_i == 0 : 
            ax.set_title('Output')
        plt.imshow(outputState,vmin=0,vmax=1)
        
        #plt.colorbar(orientation='vertical')
    if toShow : 
        plt.show()
    if toSave : 
        fig.savefig(fileName)

def get_list_heatmap(center,cov,image_size_x,image_size_y,percent_radius,exact_radius = None) :
    
    radius_x = int(image_size_x * percent_radius)
    radius_y = int(image_size_y * percent_radius)
    
    #print(radius_x,radius_y)
    
    l_cd = []
    
    t_radius_x = radius_x
    t_radius_y = radius_y
    
    if t_radius_x <= 0 : 
        t_radius_x = 1
    if t_radius_y <= 0 : 
        t_radius_y = 1 
        
    
    if exact_radius is not None : 
        t_radius_x = cov
        t_radius_y = cov
        
    #print(t_radius_x,t_radius_y,"radius")
    
    for x in range(center-t_radius_x,center+t_radius_x) :
        '''print((center-x)/t_radius_y)
        print(math.acos((center-x)/t_radius_y))    
        print(math.sin(math.acos((center-x)/t_radius_y)))'''
        
        yspan = t_radius_y*math.sin(math.acos(inBoundN((center-x)/t_radius_y,-1,1)));
        for y in range (int(center-yspan),int(center+yspan))  : 
            l_cd.append([x,y])
            
    l_cd = np.asarray(l_cd)
    
    mean = [center,center]
    
    if cov is None : 
        rv = multivariate_normal.pdf(l_cd,mean = mean, cov = [t_radius_x,t_radius_y])
    else :
        rv = multivariate_normal.pdf(l_cd,mean = mean, cov = [cov,cov])
        
    return l_cd,rv
    
def get_bb(x_list, y_list, length = 68,swap = False,adding = 0,adding_xmin=None, adding_xmax = None,adding_ymin = None, adding_ymax = None,show=False):
    #print x_list,y_list
    xMin = 999999;xMax = -9999999;yMin = 9999999;yMax = -99999999;
    
    if show : 
        print(x_list, y_list)
        
    for i in range(length): #x
        if xMin > x_list[i]: 
            xMin = int(x_list[i])
        if xMax < x_list[i]: 
            xMax = int(x_list[i])
    
        if yMin > y_list[i]:
            yMin = int(y_list[i])
        if yMax < y_list[i]: 
            yMax = int(y_list[i])
 
        #if show : 
        #    print("ymin : ",yMin,'ymax : ',yMax)
    l_x = xMax - xMin
    l_y = yMax - yMin
    #print(xMin,xMax,yMin,yMax)
    if swap : 
        return [xMin,xMax,yMin,yMax]
    else : 
        if adding_xmin is None: 
            if show : 
                print("return ",[xMin-adding*l_x,yMin-adding*l_y,xMax+adding*l_x,yMax+adding*l_y])
            return [xMin-adding*l_x,yMin-adding*l_y,xMax+adding*l_x,yMax+adding*l_y]
        else :
            return [xMin+adding_xmin*l_x,yMin+adding_ymin*l_y,xMax+adding_xmax*l_x,yMax+adding_ymax*l_y] 

def get_bb_tf(x_list, y_list, length = 68,adding = 0, axMin = None, axMax = None, ayMin = None, ayMax = None):
    #print x_list,y_list
    xMin = tf.constant(999999.0);xMax = tf.constant(-9999999.0);yMin = tf.constant(9999999.0);yMax = tf.constant(-99999999.0);
    
    for i in range(length): #x
        xMin = tf.minimum(x_list[i],xMin)
        xMax = tf.maximum(x_list[i],xMax)
        yMin = tf.minimum(y_list[i],yMin)
        yMax = tf.maximum(y_list[i],yMax)
    
    l_x = xMax - xMin
    l_y = yMax - yMin
    
    #adding ranging from 0 to 1
    if axMin is None : 
        return xMin-adding*l_x,yMin-adding*l_y,xMax+adding*l_x,yMax+adding*l_y
    else :  
        return xMin+axMin*l_x,yMin+ayMin*l_y,xMax+axMax*l_x,yMax+ayMax*l_y

def padding(image):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return constant

def get_bb_face(seq_size=2,synthetic = False,path= "images/bb/"):
    
    list_gt = []
    list_labels = []
    list_labels_t = []
    
    for f in file_walker.walk(curDir +path):
        #print(f.name, f.full_path) # Name is without extension
        if f.isDirectory: # Check if object is directory
            for sub_f in f.walk():
                if sub_f.isFile:
                    if('txt' in sub_f.full_path): 
                        #print(sub_f.name, sub_f.full_path) #this is the groundtruth
                        list_labels_t.append(sub_f.full_path)
                if sub_f.isDirectory: # Check if object is directory
                    list_img = []
                    for sub_sub_f in sub_f.walk(): #this is the image
                        list_img.append(sub_sub_f.full_path)
                    list_gt.append(sorted(list_img))
    
    list_gt = sorted(list_gt)
    list_labels_t = sorted(list_labels_t)
    
    
    for lbl in list_labels_t : 
        
        with open(lbl) as file:
            x = [re.split(r',+',l.strip()) for l in file]
        y = [ list(map(int, i)) for i in x]
        list_labels.append(y)
    
    
    if seq_size is not None : 
        list_images = []
        list_ground_truth = []
        for i in range(0,len(list_gt)): 
            counter = 0
            for j in range(0,int(len(list_gt[i])/seq_size)):
                
                temp = []
                temp2 = []
                for z in range(counter,counter+seq_size):
                    temp.append(list_gt[i][z])
                    #temp2.append([list_labels[i][z][2],list_labels[i][z][3],list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][0]+list_labels[i][z][2],list_labels[i][z][1]+list_labels[i][z][3]])
                    if not synthetic : 
                        temp2.append([list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][0]+list_labels[i][z][2],list_labels[i][z][1]+list_labels[i][z][3]])
                    else : 
                        #temp2.append([list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][2],list_labels[i][z][3]])
                        temp2.append([list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][2],list_labels[i][z][3]])
                
                counter+=seq_size
                #print counter
                    
                list_images.append(temp) 
                list_ground_truth.append(temp2) 
    else : 
        list_images = []
        list_ground_truth = []
        for i in range(0,len(list_gt)): #per folder 
            temp = []
            temp2 = []
            for j in range(0,len(list_gt[i])):#per number of seq * number of data/seq_siz 
                
                temp.append(list_gt[i][j])
                #temp2.append([list_labels[i][z][2],list_labels[i][z][3],list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][0]+list_labels[i][z][2],list_labels[i][z][1]+list_labels[i][z][3]])
                if not synthetic : 
                    temp2.append([list_labels[i][j][0],list_labels[i][j][1],list_labels[i][j][0]+list_labels[i][j][2],list_labels[i][j][1]+list_labels[i][j][3]])
                else : 
                    #temp2.append([list_labels[i][z][0],list_labels[i][z][1],list_labels[i][z][2],list_labels[i][z][3]])
                    temp2.append([list_labels[i][j][0],list_labels[i][j][1],list_labels[i][j][2],list_labels[i][j][3]])
            
                    
            list_images.append(temp) 
            list_ground_truth.append(temp2)
        
    '''
    print len(list_images)
    print len(list_ground_truth)
            
    print (list_images[0])
    print (list_ground_truth[0])

    img = cv2.imread(list_images[0][0])
    
    cv2.rectangle(img,(list_ground_truth[0][0][2],list_ground_truth[0][0][3]),(list_ground_truth[0][0][4],list_ground_truth[0][0][5]),(255,0,255),1)
    cv2.imshow('jim',img)
    cv2.waitKey(0)
    '''
    return[list_images,list_ground_truth]#2d list of allsize, seqlength, (1 for image,6 for bb)


def makeGIF(files,filename):
    import imageio
    image = []
    for i in files :
        cv2_im = cv2.cvtColor(i,cv2.COLOR_BGR2RGB) 
        image.append(cv2_im)
        #pil_im = Image.fromarray(cv2_im)   
    #print np.asarray(image).shape
    imageio.mimsave(filename,image,'GIF')    


def get_kp_face_temp(seq_size=None,data_list = ["300VW-Train"],per_folder = False,n_skip = 1,is3D = False,is84 = False, dir_name = None,theCurDir = None):
        
    list_gt = []
    list_labels_t = []
    list_labels = []
    
    if theCurDir is not None : 
        theDir = theCurDir
    else : 
        theDir = curDir + "images/"
    
    counter_image = 0
    
    i = 0
    
    if dir_name is not None : 
        annot_name = dir_name 
    else :
        if is84 :
            annot_name = 'annot84'
        elif is3D : 
            annot_name = 'annot2'
        else : 
            annot_name = 'annot' 
            
    for data in data_list : 
        print(("Opening "+data))
        
        for f in file_walker.walk(theDir):
            if f.isDirectory: # Check if object is directory
                
                print((f.name, f.full_path)) # Name is without extension
                for sub_f in f.walk():
                    
                    if sub_f.isDirectory: # Check if object is directory
                        
                        list_dta = []
                        
                        #print sub_f.name
                        
                        for sub_sub_f in sub_f.walk(): #this is the data
                            if(".npy" not in sub_sub_f.full_path):
                                list_dta.append(sub_sub_f.full_path)
                        
                    
                        if(sub_f.name == annot_name) : #If that's annot, add to labels_t 
                            list_labels_t.append(sorted(list_dta))
                        elif(sub_f.name == 'img'): #Else it is the image
                            list_gt.append(sorted(list_dta))
                            counter_image+=len(list_dta)
    '''
    print len(list_gt[2])
    print len(list_labels_t[2])
    '''               
                       
    #print list_gt 
    #print list_labels_t     
    
    print("Now opening keylabels")
                        
    for lbl in list_labels_t : 
        #print lbl
        lbl_68 = [] #Per folder
        for lbl_sub in lbl : 
            
            print(lbl_sub)
            
            if ('pts' in lbl_sub) : 
                x = []
                
                with open(lbl_sub) as file:
                    data2 = [re.split(r'\t+',l.strip()) for l in file]
                
                #print data
                
                for i in range(len(data2)) :
                    if(i not in [0,1,2,len(data2)-1]):
                        x.append([ float(j) for j in data2[i][0].split()] )
                #y = [ list(map(int, i)) for i in x]
                
                #print len(x)
                lbl_68.append(x) #1 record
                
        list_labels.append(lbl_68)
        
    #print len(list_gt[2])           #dim  : numfolder, num_data
    #print len(list_labels[2])  #dim  : num_folder, num_data, 68
    
    list_images = []
    
    max_width = max_height = -9999
    min_width = min_height = 9999
    mean_width = mean_height = 0
    
    print(("Total data : "+str(counter_image)))
    
    print("Now partitioning data if required")
    
    if seq_size is not None : 
        
        list_ground_truth = np.zeros([int(counter_image/(seq_size*n_skip)),seq_size,136])
        indexer = 0;
        
        for i in range(0,len(list_gt)): #For each dataset
            counter = 0
            for j in range(0,int(len(list_gt[i])/(seq_size*n_skip))): #for number of data/batchsize
                
                temp = []
                temp2 = np.zeros([seq_size,136])
                i_temp = 0
                
                for z in range(counter,counter+(seq_size*n_skip),n_skip):#1 to seq_size 
                    temp.append(list_gt[i][z])
                    temp2[i_temp] = np.array(list_labels[i][z]).flatten('F')
                    i_temp+=1
                    
                list_images.append(temp)
                list_ground_truth[indexer] = temp2
                    
                indexer += 1
                counter+=seq_size*n_skip
                #print counter
    else : 
        if per_folder : #divide per folder
            print("Per folder")
            list_ground_truth = []
            
            indexer = 0;
            for i in range(0,len(list_gt)): #For each dataset
                temp = []
                temp2 = []
                
                
                for j in range(0,len(list_gt[i]),n_skip): #for number of data/batchsize
                    #print len(list_gt[i])
                    #print len(list_labels[i])
                    #print(list_gt[i][j],list_labels[i][j])
                    temp.append(list_gt[i][j])
                    temp2.append(np.array(list_labels[i][j]).flatten('F'))
                    
                list_images.append(temp) 
                list_ground_truth.append(temp2)
            
        else : #make as one long list, for localisation
            if dir_name is not None : 
                list_ground_truth = np.zeros([counter_image,204])
            elif is84: 
                list_ground_truth = np.zeros([counter_image,168])
            else : 
                list_ground_truth = np.zeros([counter_image,136])
            indexer = 0;
            for i in range(0,len(list_gt)): #For each dataset
                for j in range(0,len(list_gt[i]),n_skip): #for number of data
                    
                    #print(("{}/{} {}/{}".format(i,len(list_gt),j,len(list_gt[i]))))
                    tmpImage = cv2.imread(list_gt[i][j])
                    list_images.append(list_gt[i][j])
                    #print(list_gt[i][j])
                    list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                        
                    indexer += 1
                    #print counter
            mean_width/= indexer
            mean_height/= indexer
        
    return list_images,list_ground_truth,[mean_width,mean_height, min_width,max_width, min_height, max_height]



def get_kp_face(seq_size=None,data_list = ["300VW-Train"],per_folder = False,n_skip = 1,is3D = False,is84 = False, dir_name = None,theCurDir = None):
        
    list_gt = []
    list_labels_t = []
    list_labels = []
    
    if theCurDir is not None : 
        theDir = theCurDir
    else : 
        theDir = curDir + "images/"
    
    counter_image = 0
    
    i = 0
    
    if dir_name is not None : 
        annot_name = dir_name 
    else :
        if is84 :
            annot_name = 'annot84'
        elif is3D : 
            annot_name = 'annot2'
        else : 
            annot_name = 'annot' 
            
    for data in data_list : 
        print(("Opening "+data))
        
        for f in file_walker.walk(theDir++data+"/"):
            if f.isDirectory: # Check if object is directory
                
                print((f.name, f.full_path)) # Name is without extension
                for sub_f in f.walk():
                    
                    if sub_f.isDirectory: # Check if object is directory
                        
                        list_dta = []
                        
                        #print sub_f.name
                        
                        for sub_sub_f in sub_f.walk(): #this is the data
                            if(".npy" not in sub_sub_f.full_path):
                                list_dta.append(sub_sub_f.full_path)
                        
                    
                        if(sub_f.name == annot_name) : #If that's annot, add to labels_t 
                            list_labels_t.append(sorted(list_dta))
                        elif(sub_f.name == 'img'): #Else it is the image
                            list_gt.append(sorted(list_dta))
                            counter_image+=len(list_dta)
    '''
    print len(list_gt[2])
    print len(list_labels_t[2])
    '''               
                       
    #print list_gt 
    #print list_labels_t     
    
    print("Now opening keylabels")
                        
    for lbl in list_labels_t : 
        #print lbl
        lbl_68 = [] #Per folder
        for lbl_sub in lbl : 
            
            print(lbl_sub)
            
            if ('pts' in lbl_sub) : 
                x = []
                
                with open(lbl_sub) as file:
                    data2 = [re.split(r'\t+',l.strip()) for l in file]
                
                #print data
                
                for i in range(len(data2)) :
                    if(i not in [0,1,2,len(data2)-1]):
                        x.append([ float(j) for j in data2[i][0].split()] )
                #y = [ list(map(int, i)) for i in x]
                
                #print len(x)
                lbl_68.append(x) #1 record
                
        list_labels.append(lbl_68)
        
    #print len(list_gt[2])           #dim  : numfolder, num_data
    #print len(list_labels[2])  #dim  : num_folder, num_data, 68
    
    list_images = []
    
    max_width = max_height = -9999
    min_width = min_height = 9999
    mean_width = mean_height = 0
    
    print(("Total data : "+str(counter_image)))
    
    print("Now partitioning data if required")
    
    if seq_size is not None : 
        
        list_ground_truth = np.zeros([int(counter_image/(seq_size*n_skip)),seq_size,136])
        indexer = 0;
        
        for i in range(0,len(list_gt)): #For each dataset
            counter = 0
            for j in range(0,int(len(list_gt[i])/(seq_size*n_skip))): #for number of data/batchsize
                
                temp = []
                temp2 = np.zeros([seq_size,136])
                i_temp = 0
                
                for z in range(counter,counter+(seq_size*n_skip),n_skip):#1 to seq_size 
                    temp.append(list_gt[i][z])
                    temp2[i_temp] = np.array(list_labels[i][z]).flatten('F')
                    i_temp+=1
                    
                list_images.append(temp)
                list_ground_truth[indexer] = temp2
                    
                indexer += 1
                counter+=seq_size*n_skip
                #print counter
    else : 
        if per_folder : #divide per folder
            print("Per folder")
            list_ground_truth = []
            
            indexer = 0;
            for i in range(0,len(list_gt)): #For each dataset
                temp = []
                temp2 = []
                
                '''print(len(list_gt[i]))
                print(list_gt[i][0])
                print(len(list_labels[i]))'''
                
                for j in range(0,len(list_gt[i]),n_skip): #for number of data/batchsize
                    #print len(list_gt[i])
                    #print len(list_labels[i])
                    #print(list_gt[i][j],list_labels[i][j])
                    temp.append(list_gt[i][j])
                    temp2.append(np.array(list_labels[i][j]).flatten('F'))
                    
                list_images.append(temp) 
                list_ground_truth.append(temp2)
            
        else : #make as one long list, for localisation
            if dir_name is not None : 
                list_ground_truth = np.zeros([counter_image,204])
            elif is84: 
                list_ground_truth = np.zeros([counter_image,168])
            else : 
                list_ground_truth = np.zeros([counter_image,136])
            indexer = 0;
            for i in range(0,len(list_gt)): #For each dataset
                for j in range(0,len(list_gt[i]),n_skip): #for number of data
                    
                    #print(("{}/{} {}/{}".format(i,len(list_gt),j,len(list_gt[i]))))
                    tmpImage = cv2.imread(list_gt[i][j])
                    '''height, width, channels = tmpImage.shape
                    
                    mean_width+=width;
                    mean_height+=height;
                    
                    if max_width<width : 
                        max_width = width
                    if max_height<height : 
                        max_height = height
                        
                    if min_width>width : 
                        min_width = width
                    if min_height>height : 
                        min_height = height'''
                        
                    list_images.append(list_gt[i][j])
                    #print(list_gt[i][j])
                    list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                        
                    indexer += 1
                    #print counter
            mean_width/= indexer
            mean_height/= indexer
        
        '''
        im_width = 240
        im_height = 180
        
        img = cv2.imread(list_images[500])
        height, width, channels = img.shape
        img = cv2.resize(img,(im_width,im_height))
        
        ratioWidth = truediv(im_width,width)
        ratioHeight = truediv(im_height,height)
        
        print ratioWidth,im_width,width
        print ratioHeight,im_height,height
        
        x_list = list_ground_truth[500,0:68] * ratioWidth
        y_list = list_ground_truth[500,68:136] * ratioHeight
        #getting the bounding box of x and y
        
        
        bb = get_bb(x_list,y_list)
        
        cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,255),1)
        for i in range(68) :
            cv2.circle(img,(int(x_list[i]),int(y_list[i])),3,(0,0,255))
        
        cv2.imshow('jim',img)
        cv2.waitKey(0)'''
        
    return list_images,list_ground_truth,[mean_width,mean_height, min_width,max_width, min_height, max_height]


def get_kp_face_localize(seq_size=None,data = "300W/01_Indoor"):
        
    list_gt = []
    list_labels_t = []
    list_labels = []
    
    counter_image = 0
    
    i = 0
    
    print(("Opening "+data))
    
    for f in file_walker.walk(curDir + "images/"+data+"/"):
        print((f.name, f.full_path)) # Name is without extension
        if f.isDirectory: # Check if object is directory
            
            for sub_f in f.walk():
                
                if sub_f.isDirectory: # Check if object is directory
                    
                    list_dta = []
                    
                    #print sub_f.name
                    
                    for sub_sub_f in sub_f.walk(): #this is the data
                        list_dta.append(sub_sub_f.full_path)
                    
                    if(sub_f.name == 'annot') : #If that's annot, add to labels_t 
                        list_labels_t.append(sorted(list_dta))
                    elif(sub_f.name == 'img'): #Else it is the image
                        list_gt.append(sorted(list_dta))
                        counter_image+=len(list_dta)
    '''
    print len(list_gt[2])
    print len(list_labels_t[2])
    '''
    print("Now opening keylabels")
                        
    for lbl in list_labels_t : 
        #print lbl
        lbl_68 = [] #Per folder
        for lbl_sub in lbl : 
            
            #print lbl_sub
            
            if ('pts' in lbl_sub) : 
                x = []
                
                with open(lbl_sub) as file:
                    data = [re.split(r'\t+',l.strip()) for l in file]
                
                #print data
                
                for i in range(len(data)) :
                    if(i not in [0,1,2,len(data)-1]):
                        x.append([ float(j) for j in data[i][0].split()] )
                #y = [ list(map(int, i)) for i in x]
                
                #print len(x)
                lbl_68.append(x) #1 record
                
        list_labels.append(lbl_68)
        
    #print len(list_gt[2])           #dim  : numfolder, num_data
    #print len(list_labels[2])  #dim  : num_folder, num_data, 68
    
    list_images = []
    list_ground_truth = []
    
    max_width = max_height = -9999
    min_width = min_height = 9999
    mean_width = mean_height = 0
    
    print(("Total data : "+str(counter_image)))
    
    print("Now partitioning data if required")
    
        
    indexer = 0;
    if seq_size is None : 
        for i in range(0,len(list_gt)): #For each dataset
            
            temp = []
            temp2 = []
            
            for j in range(0,len(list_gt[i])): #for number of data/batchsize
                t_temp = []
                t_temp2 = []
                
                for k in range (2) : 
                    t_temp.append(list_gt[i][j])
                    t_temp2.append(np.array(list_labels[i][j]).flatten('F'))
                    
                temp.append(t_temp)
                temp2.append(t_temp2)
                
            list_images.append(temp) 
            list_ground_truth.append(temp2)
    else : 
        for i in range(0,len(list_gt)): #For each dataset
            
            for j in range(0,len(list_gt[i])): #for number of data/batchsize
                t_temp = []
                t_temp2 = []
                
                for k in range (2) : 
                    t_temp.append(list_gt[i][j])
                    t_temp2.append(np.array(list_labels[i][j]).flatten('F'))
                    
                list_images.append(t_temp)
                list_ground_truth.append(t_temp2)
    #print list_images
    return list_images,list_ground_truth,[mean_width,mean_height, min_width,max_width, min_height, max_height]
    



def write_kp_file(finalTargetL,arr,length = 68):
    
    file = open(finalTargetL,'w')
    file.write('version: 1\n')
    file.write('n_points: '+str(length)+'\n')
    file.write('{\n')
    for j in range(length) :
        file.write(str(arr[j])+' '+str(arr[j+length])+'\n')
    file.write('}')
    file.close()


def writeLdmarkFile(fileName, ldmark):
    file = open(fileName,'w')
    file.write('version: 1\n')
    file.write('n_points: 68\n')
    file.write('{\n')
    for i in range(68) :
        file.write(str(ldmark[i])+' '+str(ldmark[i+68])+'\n')
    file.write('}')
    file.close()
    return


def test():
    import numpy as np 
    #z = np.array(((1,1),(-1,1),(-1,-1),(1,-1),(1,1)))
    
    v = z[:,0]
    a = z[:,1]
    
    vc = (v>0).astype(int)
    ac = (a>0).astype(int)
    q0 = (vc+ac>1).astype(int)*1
    
    vc = (v<0).astype(int)
    ac = (a>0).astype(int)
    q1 = (vc+ac>1).astype(int)*2
    
    vc = (v<0).astype(int)
    ac = (a<0).astype(int)
    q2 = (vc+ac>1).astype(int)*3
    
    vc = (v>0).astype(int)
    ac = (a<0).astype(int)
    q3 = (vc+ac>1).astype(int)*4
    
    qtotal = (q0+q1+q2+q3)-1
    print(q0,q1,q2,q3)
    print(qtotal)
    
    
def toQuadrant(z):
    v = z[:,0]
    a = z[:,1]
    
    vc = (v>0).astype(int)
    ac = (a>0).astype(int)
    q0 = (vc+ac>1).astype(int)*1
    
    vc = (v<0).astype(int)
    ac = (a>0).astype(int)
    q1 = (vc+ac>1).astype(int)*2
    
    vc = (v<0).astype(int)
    ac = (a<0).astype(int)
    q2 = (vc+ac>1).astype(int)*3
    
    vc = (v>0).astype(int)
    ac = (a<0).astype(int)
    q3 = (vc+ac>1).astype(int)*4
    
    qtotal = (q0+q1+q2+q3)-1
    #print(q0,q1,q2,q3)
    #print(qtotal)
    return(qtotal)
    

#test()