'''
Created on Oct 17, 2018

@author: deckyal
'''

from math import sqrt
import re

from PIL import Image,ImageFilter

import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import csv
import torchvision.transforms.functional as F
import numbers
from torchvision.transforms import RandomRotation,RandomResizedCrop,RandomHorizontalFlip



from utils import *
from config import *
from ImageAugment import *
import utils
from os.path import isfile
import os
#import nudged
import shutil
import file_walker
import copy
from NetworkModels import GeneralDAEX, LogisticRegression, DAEE

#noiseParamList = np.asarray([[0,0,0],[1,2,3],[1,3,5],[.001,.005,.01],[.8,.5,.2],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]
noiseParamList =np.asarray([[0,0,0],[2,3,4],[2,4,6],[.005,.01,.05],[.5,.2,.1],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]

#noiseParamListTrain = np.asarray([[0,0,0],[2,3,4],[2,4,6],[.005,.01,.05],[.5,.2,.1],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]
noiseParamListTrain = np.asarray([[0,0,0],[2,3,4],[2,4,6],[.005,.01,.05],[.5,.2,.1],[0,0,0]])#0 [], 1[1/2,2/4,3/8], 2 [1,3,5], 3 [.01,.1,1], [.001,.005,.01]

rootDir = "/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data"
#rootDir = "/homedtic/daspandilatif/workspace/MMTVA/data"

rootDirLdmrk = "/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/"

def addGaussianNoise(img,noiseLevel = 1):
    noise = torch.randn(img.size()) * noiseLevel
    noisy_img = img + noise
    return noisy_img



def toQuadrant(inputData = None, min = -10, max = 10,  toOneHot = False):
    threshold = truediv(min+max,2)
    vLow = False
    aLow = False
    q = 0
    
    #print(min,max)
    
    #print('the threshold : ',threshold)
    
    if inputData[0] < threshold : 
        vLow = True
    
    if inputData[1] < threshold : 
        aLow = True
    
    if vLow and aLow : 
        q = 2
    elif vLow and not aLow : 
        q = 1 
    elif not vLow and not aLow : 
        q = 0 
    else : 
        q = 3 
    
    if toOneHot : 
        rest = np.zeros(4)
        rest[q]+=1
        return rest 
    else : 
        return q 
    
    
class AFFChallenge(data.Dataset): #return affect on Valence[0], Arousal[1] order
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["AffectChallenge"],listMode = 'Train',onlyFace = True, image_size =112, 
                 transform = None,useIT = False,augment = False, step = 1,isVideo = False, seqLength = None, dbType = 0,
                 returnQuadrant = False, returnNoisy = False, returnWeight = False ,fill = 1,returnSound = False):#dbtype 0 is AFEW, 1 is SEWA
        #fill = 1, will attempt to duplicate the data to the length of ground turth 
        #fill = 0, will just use the existing image to synchronize it with the groundturht value 
        
        self.dbType = dbType
        
        self.seq_length = seqLength 
        self.isVideo = isVideo
        self.returnSound = returnSound
        
        self.transform = transform
        self.augment = augment 
        
        self.imageSize = image_size
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.useIT = useIT
        self.curDir = rootDir +"/"#/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/"
        
        self.returnNoisy = returnNoisy
        self.returnWeight = returnWeight
        
        self.returnQ = returnQuadrant
        
        if self.augment : 
            self.flip = RandomHorizontalFlip(1)
            self.rot = RandomRotation(45)
            self.occ = Occlusion(1)
            self.rc = RandomResizedCrop(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
            
        if self.returnNoisy : 
            self.gn = GeneralNoise(1)
            self.occ = Occlusion(1)
        
        
        list_gt = []
        list_labels_tE = []
        list_labels_sound = []
        
        counter_image = 0
        weight = None
        
        for data in data_list : 
            print(("Opening "+data))
            
            for mode in listMode :
                #self.isTest = 1 if mode =='Test'else 0
            
                if self.returnWeight :
                    name = 'AC-'+mode+'.npy'
                    if weight is None : 
                        weight = np.load(rootDir+"/DST-SE-AF/"+name).astype('float')+1
                    else : 
                        weight += np.load(rootDir+"/DST-SE-AF/"+name).astype('float')+1
                
                
                fullDir = self.curDir +data+"/images/VA/"+mode
                fullDirLbl = self.curDir +data+"/labels/VA/"+mode
                fullDirLblSnd = self.curDir +data+"/sounds/VA/"+mode
                
                listFolder = os.listdir(fullDir)
                listFolder.sort()
                
                for tempx in range(len(listFolder)):
                    f = listFolder[tempx]
                    fullPath = os.path.join(fullDir,f)
                    #print('opening fullpath',fullPath)
                    if os.path.isdir(fullPath): # Check if object is directory
                        #print((f.name, f.full_path)) # Name is without extension
                        #c_image,c_ldmark = 0,0
                        vaFile = fullDirLbl+"/"+f+".txt"
                        list_labels_tE.append(vaFile)
                        
                        sndFile = fullDirLblSnd+"/"+f+"_audioFeatures.csv"
                        list_labels_sound.append(sndFile)
                            
                        list_dta = []
                        for sub_f in file_walker.walk(fullPath):
                            list_dta.append(sub_f.full_path)
                        list_gt.append(sorted(list_dta))
                        
                        #counter_image+=len(list_dta)
        
        if self.returnWeight : 
            sum = weight.sum(0)
            
            weight = (weight/sum)
            #print('1',weight)
            
            weight = 1/weight
            #print('2',weight)
            
            sum = weight.sum(0)
            weight = weight/sum
            #print('3',weight)
            
            self.weight =  weight
        
        print("Now opening keylabels")
         
        list_labelsEN = []
        list_labelsE = []
        list_snds = []
        
        for ix in range(len(list_labels_tE)) : #lbl,lble in (list_labels_t,list_labels_tE) :
            
            dataFile = list_labels_tE[ix]    
            x = []
            #print(lbl_sub)
            with open(dataFile) as file:
                data2 = [re.split(r'\t+',l.strip()) for l in file]
            for i in range(len(data2)) :
                if i == 0 : 
                    pass 
                else : 
                    temp = [ float(j) for j in data2[i][0].split(',')]
                    #temp.reverse() #to give the valence first. then arousal
                    x.append(temp)
                    #print(temp)
                    
            list_sndsx = []
            print(list_labels_sound[ix])
            with open(list_labels_sound[ix], 'r') as csvFile:
                
                dialect = csv.Sniffer().sniff(csvFile.read(1024), delimiters=";,")
                csvFile.seek(0)
                reader = csv.reader(csvFile, dialect)
                #reader = csv.reader(csvFile)
                for row in reader:
                    #print(row)
                    #print(row[0])
                    ''' if ';' in row : 
                        data = row[0].split(';')
                    else : 
                        data = row[0].split(',')
                    '''
                    #print(data[1:])
                    #print(data,row[0])
                    #print(data)
                    #print('d',data)
                    list_sndsx.append(np.array(row[1:]).astype(np.float))
                    #print(np.array(row[1:]).astype(np.float))
            #print(list_sndsx)
            list_snds.append(list_sndsx)
               
            
            list_labelsE.append(np.array(x))
            list_labelsEN.append(dataFile)
            #break
            
        t_l_imgs = []
        t_l_gtE = []
        t_l_snd = []
        t_list_gtE_names = []
        
        '''print(list_snds[0])
        exit(0)'''
        '''doCheck = True
        
        if doCheck : 
            for i in range(0,len(list_gt)):
                fileName = list_labelsEN[i]
                gtCount = len(list_labelsE[i])
                imgCount = len(list_gt[i])
                
                if imgCount != gtCount : 
                    print('filling ')
                    print('targetFolder = ')'''
            
        
        countVideo=0
        countGT = 0
        #print(list_labelsE)
        if not self.isVideo :
            #Flatten it to one list
            
            for i in range(0,len(list_gt)): #For each dataset
                
                list_images = []
                list_gtE_names = []
                
                indexer = 0
                
                if fill : 
                    length = len(list_labelsE[i])
                else : 
                    length = len(list_gt[i])
                
                counter_image+=length
                
                list_ground_truthE = np.zeros([length,2])
                list_sounds = np.zeros([length,46])
                
                print(list_labelsEN[i])
                
                
                countVideo+=len(list_gt[i])
                countGT+=len(list_labelsE[i])
                '''print('length images : ',len(list_gt[i]))
                print('length gt : ',len(list_labelsE[i]))'''
                
                #if (len(list_gt[i]!= len))
                
                realIndex = 0  
                
                #print('lenght : ',length)
                #print(list_snds[i])
                for j in range(0,length,step): #for number of data #n_skip is usefull for video data
                    
                    if fill : 
                        #additional check
                        fnameIndex = int(os.path.basename(list_gt[i][realIndex]).split('.')[0])-1
                        if fnameIndex != j :
                            list_images.append(list_gt[i][realIndex])
                            #print('mismatch',j,'-',realIndex,'--',fnameIndex)
                        else : 
                            list_images.append(list_gt[i][realIndex])
                            
                            if realIndex < len(list_gt[i])-1:
                                realIndex+=1 
                            
                        list_gtE_names.append(list_labelsEN[i])
                        list_ground_truthE[indexer] = np.array(list_labelsE[i][j]).flatten('F')
                        #print('bitch',list_snds[i][j])
                        #print(list_snds[i][j].shape)
                        #print(list_snds[i][j])
                        if j >= len(list_snds[i]): 
                            list_sounds[indexer] = list_snds[i][-1]
                        else : 
                            #print(list_snds[i][j])
                            list_sounds[indexer] = list_snds[i][j]
                        
                    else :
                        fnameIndex = int(os.path.basename(list_gt[i][j]).split('.')[0])-1
                        list_images.append(list_gt[i][j])
                        #print(list_labelsEN)
                        list_gtE_names.append(list_labelsEN[i])
                        #print(list_labelsEN[i])
                        list_ground_truthE[indexer] = np.array(list_labelsE[i][fnameIndex]).flatten('F')
                        list_sounds[indexer] = list_snd[i][fnameIndex]
                        
                    
                    '''if len(list_labels[i][j] < 1): 
                        print(list_labels[i][j])'''
                    #print(len(list_labels[i][j]))
                    #print('f',list_labelsE[i][j])
                    #list_ground_truthE[indexer] = np.array(list_labelsE[i][j]).flatten('F')
                    indexer += 1
                
                t_l_snd.append(list_sounds)
                t_l_imgs.append(list_images)
                t_l_gtE.append(list_ground_truthE)
                t_list_gtE_names.append(list_gtE_names)
                
        else : 
            if self.seq_length is None :
                list_ground_truth = np.zeros([int(counter_image/(self.seq_length*step)),self.seq_length,136])
                indexer = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                    counter = 0
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize
                        
                        temp = []
                        temp2 = np.zeros([self.seq_length,136])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z])
                            temp2[i_temp] = list_labels[i][z]
                            i_temp+=1
                            
                        list_images.append(temp)
                        list_ground_truth[indexer] = temp2
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                self.l_imgs = list_images
                self.l_gt = list_ground_truth
            else : 
                counter_seq = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                
                    indexer = 0;
                    list_gtE_names = []
                    list_ground_truthE = np.zeros([int(len(list_gt[i])/(self.seq_length*step)),self.seq_length,2])#np.zeros([counter_image,2])
                    
                    counter = 0
                    list_images = []
                    
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize   
                        
                        
                        
                        temp = []
                        tmpn2 = []
                        temp3 = np.zeros([self.seq_length,2])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z]) 
                            temp3[i_temp] = list_labelsE[i][z].flatten('F')
                            
                            tmpn2.append(list_labelsEN[i])
                            
                            i_temp+=1
                            counter_seq+=1
                            
                        list_images.append(temp)
                        list_ground_truthE[indexer] = temp3
                        list_gtE_names.append(tmpn2)
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                        
                    t_l_imgs.append(list_images)
                    t_l_gtE.append(list_ground_truthE)
                    
                    t_list_gtE_names.append(list_gtE_names)
                
        '''print('length : ',len(t_l_imgs)) #Folder
        print('lengt2 : ',len(t_l_imgs[0])) #all/seq
        print('lengt3 : ',len(t_l_imgs[0][0])) #seq
        print('length4 : ',len(t_l_imgs[0][0][0]))'''
        
        
        #print(list_gt) 
        self.length = counter_image
        
        #[folder, all/seq,seq]
        
        print(countVideo,'-',countGT)
        #exit(0)
        
        self.l_imgs = []
        self.l_gtE = []
        self.l_snd = []
        self.list_gtE_names = []
        
        #print('cimage : ',counter_image)
        
        
        if not self.isVideo :
            self.l_gtE = np.zeros([counter_image,2])
            self.l_snd = np.zeros([counter_image,46])
            indexer = 0
            
            
            for i in range(len(t_l_imgs)): 
                for j in range(len(t_l_imgs[i])): 
                    self.l_imgs.append(t_l_imgs[i][j])
                    #print(i,j,'-',len(t_l_imgs[i]))
                    #self.l_gtE[indexer] = t_l_gtE[i][j]
                    
                    #print(t_l_gtE[i][j])
                    va = t_l_gtE[i][j]
                    v = va[0]
                    a = va[1]
                    if (v<-1 or v>1): 
                        print('more',v)
                        v/=10
                        print('more2',v)
                    if (a<-1 or v>1): 
                        print('more',a) 
                        a/=10
                        print('more2',a)
                        
                    self.l_gtE[indexer] = np.array((v,a))#t_l_gtE[i][j]
                    
                    self.list_gtE_names.append(t_list_gtE_names[i][j])
                    self.l_snd[indexer] = t_l_snd[i][j]
                    indexer+=1
                
        else : 
            self.l_gtE = np.zeros([counter_seq,self.seq_length,2])
            
            indexer = 0
            
            for i in range(len(t_l_imgs)): #dataset
                for j in range(len(t_l_imgs[i])): #seq counter
                    
                    t_img = []
                    
                    t_gte = np.zeros([self.seq_length,2])
                    
                    t_gt_n = []
                    t_gt_en = []
                    i_t = 0
                    
                    for k in range(len(t_l_imgs[i][j])): #seq size
                        
                        t_img.append(t_l_imgs[i][j][k])
                        t_gte[i_t] = t_l_gtE[i][j][k]
                        
                        t_gt_en.append(t_list_gtE_names[i][j][k])
                        
                        i_t+=1
                        
                    self.l_imgs.append(t_img)
                    self.l_gtE[indexer] = t_gte
                    self.list_gtE_names.append(t_gt_en)
                    
                    indexer+=1
                        
        print('limgs : ',len(self.l_imgs))

    def __getitem__(self,index):
        #Read all data, transform etc.
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
        
        l_imgs = []; l_ldmrk = []; l_VA = []; l_nc = []; l_qdrnt = []; l_SD = []#,torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]
        if self.returnNoisy : 
            l_nimgs = []
        
        if self.returnWeight : 
            l_weights = []
        
        if not self.isVideo : 
            x_l = [self.l_imgs[index]];labelE_l =[self.l_gtE[index].copy()];label_n =[self.list_gtE_names[index]]; l_snd = [self.l_snd[index]]
        else : 
            x_l = self.l_imgs[index];labelE_l =self.l_gtE[index].copy();label_n =self.list_gtE_names[index];  l_snd = self.l_snd[index]
        
        
        #print('label n ',label_n)
        for x,labelE,ln,ls in zip(x_l,labelE_l,label_n, l_snd) : 
            #print(x,labelE,label,ln)
            #print(x)
            tImage = Image.open(x).convert("RGB")
            tImageB = None
            
            newChannel = None
            
            if self.augment : 
                if self.returnNoisy :
                    sel = np.random.randint(0,3) #Skip occlusion as noise
                else : 
                    sel = np.random.randint(0,4)
                #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
                if sel == 0 : 
                    pass
                elif sel == 1 : 
                    tImage = self.flip(tImage)
                elif sel == 2 : 
                    tImage = self.rot(tImage)
                elif sel == 3 : 
                    tImage = self.occ(tImage)
                    
                #random crop
                if (np.random.randint(1,3)%2==0) :
                    tImage= self.rc(tImage)
                
            if self.returnNoisy :
                nImage = tImage.copy()
            
                #additional blurring
                if (np.random.randint(1,3)%2==0): 
                    #sel_n = np.random.randint(1,6)
                    sel_n = np.random.randint(1,7)
                    
                    #sel_n = 4
                    #gn = GeneralNoise_WL(1)
                    #tImage,label= gn(tImage,label,sel_n,np.random.randint(0,3))
                    
                    if sel_n > 5 : 
                        #occ = Occlusion(1)
                        nImage = self.occ(nImage)
                    else :
                        #rc = GeneralNoise(1)
                        #tImage = rc(tImage,sel_n,np.random.randint(0,3))
                        nImage = self.gn(nImage,sel_n,np.random.randint(0,3))
                    
            label = torch.zeros(1)
            Mt = torch.zeros(1)
            
            
            if self.useIT : 
                tImage = self.transformInternal(tImage)
                if self.returnNoisy : 
                    nImage = self.transformInternal(nImage)
            else : 
                tImage = self.transform(tImage)
                if self.returnNoisy : 
                    nImage = self.transform(nImage)
            
            
            
            l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label))#,x,self.list_gt_names[index]
            l_SD.append(torch.FloatTensor(ls))
            
            if self.returnNoisy : 
                l_nimgs.append(nImage)
            
            if self.returnQ : 
                min = 0; max = 1;    
                l_qdrnt.append(toQuadrant(labelE, min, max, toOneHot=False))
            
            #print(self.weight)
            if self.returnWeight :
                v = labelE[0] 
                a = labelE[0]
                
                v = v*10+10
                a = a*10+10
                
                v,a = int(v),int(a)
                '''print('the v :{} a : {} db : {}'.format(v,a,self.dbType))
                print(self.weight)
                print(self.weight.shape)'''
                l_weights.append([self.weight[v,0],self.weight[a,1]])
                
            l_nc.append(ln)
                
        if not self.isVideo : 
            if self.returnQ : 
                if self.returnNoisy :
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,ln[0],l_qdrnt[0],l_nimgs[0]]
                else : 
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,ln[0],l_qdrnt[0]]
            else :
                if self.returnNoisy : 
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,ln[0],l_nimgs[0]]
                else : 
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,ln[0]]
                    
            if self.returnWeight :
                res.append(torch.tensor(l_weights[0]))
                
                
            if self.returnSound :
                res.append(torch.tensor(l_SD[0]))
                
            return res 
        else : 
            #lImgs = torch.Tensor(len(l_imgs),3,self.imageHeight,self.imageWidth)
            #lVA = torch.Tensor(len(l_VA),2)
            #lLD = torch.Tensor(len(l_ldmrk),136)
            lImgs = torch.stack(l_imgs)
            lVA = torch.stack(l_VA)
            lLD = torch.stack(l_ldmrk)
            
            #print(l_VA)
            l_qdrnt = torch.tensor((l_qdrnt))
            
            #print(lImgs.shape, l_imgs[0].shape, l_VA[0].shape,len(lImgs))
            
            #torch.cat(l_imgs, out=lImgs)
            #torch.cat(l_VA, out=lVA)
            #torch.cat(l_ldmrk, out=lLD)
            if self.returnQ : 
                if self.returnNoisy : 
                    res = [lImgs, lVA, lLD, Mt,l_nc,l_qdrnt,l_nimgs]
                else : 
                    res = [lImgs, lVA, lLD, Mt,l_nc,l_qdrnt]
            else : 
                if self.returnNoisy : 
                    res = [lImgs, lVA, lLD, Mt,l_nc,l_nimgs]
                else : 
                    res = [lImgs, lVA, lLD, Mt,l_nc]
                    
            if self.returnWeight : 
                l_weights = torch.tensor(l_weights)
                res.append(l_weights)
                
            return res 
    
    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    
    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    
    def __len__(self):
        return len(self.l_imgs)

    
    

class FacialLandmarkDataset(data.Dataset):
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["300VW-Train"],dir_gt = None,onlyFace = True, image_size =224, 
                 transform = None,useIT = False,augment = False, step = 1):
        
        self.transform = transform
        self.onlyFace = onlyFace
        self.augment = augment 
        
        self.imageSize = image_size
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.useIT = useIT
        self.curDir = rootDirLdmrk# "/home/deckyal/eclipse-workspace/FaceTracking/"
        
        list_gt = []
        list_labels_t = []
        
        counter_image = 0
        annot_name = 'annot'
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        for data in data_list : 
            print(("Opening "+data))
            for f in file_walker.walk(self.curDir + "images/"+data+"/"):
                if f.isDirectory: # Check if object is directory
                    #print((f.name, f.full_path)) # Name is without extension
                    for sub_f in f.walk():
                        if sub_f.isDirectory: # Check if object is directory
                            list_dta = []
                            if(sub_f.name == annot_name) : #If that's annot, add to labels_t
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_labels_t.append(sorted(list_dta))
                            elif(sub_f.name == 'img'): #Else it is the image
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_gt.append(sorted(list_dta))
                                counter_image+=len(list_dta)
        
        self.length = counter_image 
        print("Now opening keylabels")
        
        list_labels = []     
        for lbl in list_labels_t :
            lbl_68 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('pts' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
            list_labels.append(lbl_68)
            
        list_images = []
        list_ground_truth = np.zeros([counter_image,136])
        
        #Flatten it to one list
        indexer = 0
        for i in range(0,len(list_gt)): #For each dataset
            for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                list_images.append(list_gt[i][j])
                list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                indexer += 1
                
        self.l_imgs = list_images
        self.l_gt = list_ground_truth

    def __getitem__(self,index):
        #Read all data, transform etc.
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
         
        x,label  = self.l_imgs[index],self.l_gt[index].copy()
        
        tImage = Image.open(x).convert("RGB")
        tImageB = None
        
        if self.onlyFace :    
            #crop the face region
            #t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 2,div_y = 2,images = cv2.imread(x),displacementxy = random.uniform(-.5,.5))
            t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 8,div_y = 8,images = cv2.imread(x))#,displacementxy = random.uniform(-.5,.5))
            
            area = (x1,y1, x2,y2)
            tImage =  tImage.crop(area)
            
            label[:68] -= x_min
            label[68:] -= y_min
            
            tImage = tImage.resize((self.imageWidth,self.imageWidth))
            
            #print(self.imageWidth/(x2 - x1))
            #print(self.imageHeight/(y2 - y1))
            
            label[:68] *= truediv(self.imageWidth,(x2 - x1))
            label[68:] *= truediv(self.imageHeight,(y2 - y1))
        
        '''print(label)
        image = utils.imageLandmarking(tImage,label)
        cv2.imshow('tt',image)
        cv2.waitKey(0)'''
        '''if self.noiseType is not None :
            tImageB = tImage.copy()
            
            #print(tImageB.size)
            
            if self.injectedNoise is None : 
                
                noiseType = np.random.randint(0,5)
                noiseLevel = np.random.randint(0,self.noiseParam)
                noiseParam=noiseParamList[noiseType,noiseLevel]
                
            else : 
                noiseType =self.injectedNoise[0]
                if self.injectedNoise[1] < 0 : 
                    noiseParam = noiseParamListTrain[self.injectedNoise[0],np.random.randint(0,self.noiseParam)]
                else : 
                    noiseParam=noiseParamList[self.injectedNoise[0],self.injectedNoise[1]]
            
            tImageB = generalNoise(tImageB,noiseType,noiseParam)
            
            if self.transform is not None:
                tImageB = self.transform(tImageB)
        '''
        
        if self.augment : 
            sel = np.random.randint(0,4)
            #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
            if sel == 0 : 
                pass
            elif sel == 1 : 
                flip = RandomHorizontalFlip_WL(1)
                tImage,label = flip(tImage,label)
            elif sel == 2 : 
                rot = RandomRotation_WL(45)
                tImage,label = rot(tImage,label)
            elif sel == 3 : 
                occ = Occlusion_WL(1)
                tImage,label = occ(tImage,label)
                
            #random crop
            if True : 
                rc = RandomResizedCrop_WL(size = self.imageSize,scale = (0.75,1), ratio = (0.75, 1.33))
                tImage,label= rc(tImage,label)
            
            #additional blurring
            if (np.random.randint(1,3)%2==0) and True: 
                sel_n = np.random.randint(1,3)
                rc = GeneralNoise_WL(1)
                tImage,label= rc(tImage,label,sel_n,2)
            
        if self.useIT : 
            tImage = self.transformInternal(tImage)
        else : 
            tImage = self.transform(tImage)
        
        return tImage,torch.FloatTensor(label)
    
    
    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    
    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    
    def __len__(self):
        return len(self.l_imgs)




class ImageDataset(data.Dataset):
    
    def __init__(self, data_list = ["300VW-Train"],dir_gt = None,onlyFace = True,step = 1,
                  transform = None, image_size =224, noiseType = None, noiseParam = None,
                  injectedNoise = None,return_noise_label = False,injectedLink = None,
                  isVideo = False,giveCroppedFace = False,annotName  = 'annot',lndmarkNumber  = 68
                  ,isSewa = True):
        
        self.return_noise_label = return_noise_label
        self.giveCroppedFace = giveCroppedFace
        
        self.injectedNoise = injectedNoise
        self.noiseType  = noiseType
        self.noiseParam =noiseParam
        
        self.transform = transform
        self.onlyFace = onlyFace
        
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.injectedLink = injectedLink
        
        self.lndmarkNumber = lndmarkNumber
        self.isSewa = isSewa
        
        
        list_gt = []
        list_labels_t_sewa = []
        list_labels_t = []
        
        counter_image = 0
        annot_name = annotName
        
        
        is_video = isVideo
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        for data in data_list : 
            print(("Opening "+data))
            
            if self.injectedLink is not None : 
                the_direction = self.injectedLink
            else :
                the_direction = curDir + "images/"+data+"/"
            
            for f in file_walker.walk(the_direction):
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
                            elif (sub_f.name == 'annot'): #Else it is the image
                                list_labels_t_sewa.append(sorted(list_dta))
            
        self.length = counter_image 
        print("Now opening keylabels ",counter_image)
        
        #print(list_labels_t)
        
        list_labels = []     
        for lbl in list_labels_t :
            lbl_68 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('pts' in lbl_sub) : 
                    x = []
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
            list_labels.append(lbl_68)
        
        
        if self.isSewa : 
            list_labels_sewa = []     
            for lbl in list_labels_t_sewa :
                lbl_68 = [] #Per folder
                for lbl_sub in lbl :
                    #print(lbl_sub)
                    if ('pts' in lbl_sub) : 
                        x = []
                        with open(lbl_sub) as file:
                            data2 = [re.split(r'\t+',l.strip()) for l in file]
                        for i in range(len(data2)) :
                            if(i not in [0,1,2,len(data2)-1]):
                                x.append([ float(j) for j in data2[i][0].split()] )
                        lbl_68.append(np.array(x).flatten('F')) #1 record
                list_labels_sewa.append(lbl_68)
        
        
            
        list_images = []
        
        if self.isSewa: 
            list_ground_truth_68 = np.zeros([counter_image,136])
        
        list_ground_truth = np.zeros([counter_image,self.lndmarkNumber*2])
        
        #Flatten it to one list
        indexer = 0
        for i in range(0,len(list_gt)): #For each dataset
            for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                list_images.append(list_gt[i][j])
                
                list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                if self.isSewa : 
                    list_ground_truth_68[indexer] = np.array(list_labels_sewa[i][j]).flatten('F')
                    
                indexer += 1
                
        self.l_imgs = list_images
        self.l_gt = list_ground_truth
        self.l_gt68 = list_ground_truth_68
        
        #print(self.l_imgs,self.l_gt)

    def __getitem__(self,index):
        
        #Read all data, transform etc.
        
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
         
        x,label  = self.l_imgs[index],self.l_gt[index].copy()
        
        if self.isSewa : 
            labelSewa = self.l_gt68[index].copy()
        #print(x,label)
        
        #print(label)
        
        tImage = Image.open(x).convert("RGB")
        tImageB = None
        
        if self.onlyFace :    
            #crop the face region
            t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 2,div_y = 2,images = cv2.imread(x),n_points=self.lndmarkNumber)
            area = (x1,y1, x2,y2)
            tImage =  tImage.crop(area)
            tImage = tImage.resize((self.imageWidth,self.imageWidth))
            
            label[:self.lndmarkNumber] -= x_min
            label[self.lndmarkNumber:] -= y_min
            
            label[:self.lndmarkNumber] *= truediv(self.imageWidth,(x2 - x1))
            label[self.lndmarkNumber:] *= truediv(self.imageHeight,(y2 - y1))
        
        if self.giveCroppedFace : 
            label_cr =  self.l_gt[index].copy()
            
            tImageCr = Image.open(x).convert("RGB")
            if self.lndmarkNumber < 68 : 
                t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label_cr,div_x = 2.5,div_y = 2.5,images = cv2.imread(x),n_points=self.lndmarkNumber)
            else : 
                t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label_cr,div_x = 8,div_y = 8,images = cv2.imread(x),n_points=self.lndmarkNumber)
                
            area = (x1,y1, x2,y2)
            #print('the arae',area)
            tImageCr =  tImageCr.crop(area)
            tImageCr = tImageCr.resize((self.imageWidth,self.imageWidth))
            
            #print('bef',label_cr)
            label_cr[:self.lndmarkNumber] -= x_min
            label_cr[self.lndmarkNumber:] -= y_min
            
            #print('a',label_cr,(x2 - x1),(y2 - y1))
            
            label_cr[:self.lndmarkNumber] *= truediv(self.imageWidth,(x2 - x1))
            label_cr[self.lndmarkNumber:] *= truediv(self.imageHeight,(y2 - y1))
            
            
            if self.isSewa : 
                #print('bef',label_cr)
                labelSewa[:68] -= x_min
                labelSewa[68:] -= y_min
                
                #print('a',label_cr,(x2 - x1),(y2 - y1))
                
                labelSewa[:68] *= truediv(self.imageWidth,(x2 - x1))
                labelSewa[68:] *= truediv(self.imageHeight,(y2 - y1))
            
            #print('asfter',label_cr)
            
            print('a',label_cr,label_cr.shape)
            print('b',labelSewa,labelSewa.shape)
            
            if self.transform is not None : 
                tImageCr = self.transform(tImageCr)
            
        if self.noiseType is not None :
            tImageB = tImage.copy()
            
            #print(tImageB.size)
            
            if self.injectedNoise is None : 
                
                noiseType = np.random.randint(0,5)
                noiseLevel = np.random.randint(0,self.noiseParam)
                noiseParam=noiseParamList[noiseType,noiseLevel]
                
            else : 
                noiseType =self.injectedNoise[0]
                if self.injectedNoise[1] < 0 : 
                    noiseParam = noiseParamListTrain[self.injectedNoise[0],np.random.randint(0,self.noiseParam)]
                else : 
                    noiseParam=noiseParamList[self.injectedNoise[0],self.injectedNoise[1]]
            
            #print(noiseType,noiseParam)
            '''if self.noiseType == 1: #downsample
                for i in range(int(self.noiseParam/2)) :#Scale down (/2) blurLevel times 
                    width, height = tImageB.size
                    tImageB = tImageB.resize((width//2,height//2))
                    #print(tImageB.size)
            elif self.noiseType == 2 : #Gaussian blur
                tImageB = tImageB.filter(ImageFilter.GaussianBlur(self.noiseParam))
            elif self.noiseType == 3 : #Gaussian noise 
                #tImageB = addNoise(tImageB)
                #convert to opencv 
                opencvImage = cv2.cvtColor(np.array(tImageB), cv2.COLOR_RGB2BGR)
                
                #print(opencvImage)
                opencvImage = addNoise(opencvImage)
                pilImage =  cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)
                #tImageB = Image.fromarray(random_noise(opencvImage))
                tImageB = Image.fromarray(pilImage)'''
            
            tImageB = generalNoise(tImageB,noiseType,noiseParam)
            
            if self.transform is not None:
                tImageB = self.transform(tImageB)
        
        if self.transform is not None:
            tImage = self.transform(tImage)
        
        if self.noiseType : 
            if self.return_noise_label : 
                return tImage,tImageB,torch.FloatTensor(label),noiseType
            else  : 
                return tImage,tImageB,torch.FloatTensor(label),x
        else : 
            if not self.giveCroppedFace : 
                return tImage,torch.FloatTensor(label),x
            else : 
                if self.isSewa : 
                    return tImage,torch.FloatTensor(label),tImageCr,x,label_cr,labelSewa
                else : 
                    return tImage,torch.FloatTensor(label),tImageCr,x,label_cr
    
    def __len__(self):
        
        return len(self.l_imgs)





class Recola(data.Dataset):
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["r-temp"],onlyFace = True, image_size =224, 
                 transform = None,useIT = False,augment = False, step = 1,multimodal = True):
        
        self.transform = transform
        self.onlyFace = onlyFace
        self.augment = augment 
        
        self.imageSize = image_size
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.useIT = useIT
        self.data_list = ['r-temp']
        self.sr = 16000
    
        curDir = rootDir +"/"#/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/"
            
        list_gtA = {} #[batch, sequences, [val,arousal]]
        list_gtV = {}
        list_m_images = {} #[batch, sequences, [b,g,r]]
        list_m_physio = {} #[batch, sequences, [ECG,EDA]]
        list_m_audios = {} #[batch, sequences, [audio]]
        list_m_ldmrks = {} #[batch, sequences, [audio]]
        list_syn = {} #[batch, sequences, [time]]
        
        counter_image = 0
        
        dirImageName = 'm-images'
        dirPhysioName = 'm-physio'
        dirAudioName = 'm-audios'
        dirSynName = 'syn-im'
        dirLdmrkNames = 'ldmrks'
        
        dir_gt = 'l-gs'
        dur_gt_ind = 'l-ind'
        
        listKey = []
         
        for data in data_list : 
            print(("Opening "+data))
            for f in file_walker.walk(curDir +data+"/"):
                if f.isDirectory: # Check if object is directory
                    print((f.name, f.full_path)) # Name is without extension
                    if(f.name == dirImageName) : #If thats image folder
                        for sub_f in f.walk(): #this is the data
                            if sub_f.isDirectory: 
                                list_dta = []
                                for sub_sub_f in sub_f.walk() : 
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                        counter_image+=len(list_dta)
                                list_m_images[sub_f.name] =  sorted(list_dta)
                                print('addding to list key',sub_f.name)
                                listKey.append(sub_f.name)
                    
                    
                    if(f.name == dirLdmrkNames) : #If thats image folder
                        for sub_f in f.walk(): #this is the data
                            if sub_f.isDirectory: 
                                list_dta = []
                                for sub_sub_f in sub_f.walk() : 
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                        counter_image+=len(list_dta)
                            list_m_ldmrks[sub_f.name] =  sorted(list_dta)
                    
                    elif(f.name == dirSynName) : #SynImage
                        for sub_f in f.walk(): #this is the data    
                            print(sub_f.full_path)  
                            if '.csv' in sub_f.full_path :     
                                list_dta = []
                                with open(sub_f.full_path, 'r') as csvFile:
                                    reader = csv.reader(csvFile,delimiter=';')
                                    for row in reader:
                                        if 'frame' in row : 
                                            pass 
                                        else : 
                                            list_dta.append([int(row[0]),float(row[1])])
                                list_syn[sub_f.name] =  sorted(list_dta)
                                
                            
                    elif(f.name == dirPhysioName) : #Physio
                        for sub_f in f.walk(): #this is the data    
                            print(sub_f.full_path)  
                            if '.csv' in sub_f.full_path : 
                                list_dta = []    
                                with open(sub_f.full_path, 'r') as csvFile:
                                    reader = csv.reader(csvFile,delimiter=';')
                                    for row in reader:
                                        if 'time' in row : 
                                            pass 
                                        else : 
                                            list_dta.append([float(row[0]),float(row[1]),float(row[4])])
                                list_m_physio[sub_f.name] =  sorted(list_dta)
                                
                    elif(f.name == dir_gt) : #gt
                        for sub_f in f.walk(): 
                            print(sub_f.full_path)
                            for sub_sub_f in sub_f.walk(): #this is the data    
                                if '.arff' in sub_sub_f.full_path :
                                    data = []
                                    with open(sub_sub_f.full_path) as f: 
                                        for line in f: 
                                            l = line.rstrip('\r\n').split(',')
                                            if ('v_' in l[0]) or ('n_' in l[0]) : #dev_x, train_x,
                                                #print('in',l[0])
                                                data.append(np.asarray([float(l[1]),float(l[2])]))
                                #print(len(sorted(data)),sub_sub_f.name)
                                #print('test',sub_sub_f.name,sub_f.name)
                                if sub_f.name == "valence" :
                                    #print('add val')
                                    list_gtV[sub_sub_f.name] = data
                                else :
                                    #print('add arou')
                                    list_gtA[sub_sub_f.name] = data
                                    
                                    
                    elif(f.name == dirAudioName) : #Physio
                        for sub_f in f.walk(): #this is the data    
                            print(sub_f.full_path)  
                            if '.wav' in sub_f.full_path : 
                                
                                signald,_ = librosa.load(sub_f.full_path,self.sr)  # File assumed to be in the same directory
                                
                                self.fps = 25
                                self.totalS = 300
                                sps = int(truediv(truediv(len(signald),self.totalS),self.fps))
                                ninstance = int(truediv(len(signald),sps))
                                
                                list_dta = []
                                for i in range(ninstance): 
                                    list_dta.append(signald[i*sps:(i+1)*sps]) 
                                
                                list_m_audios[sub_f.name] =  np.asarray(list_dta)
                                                
        print("Now opening keylabels")
        print(len(list_m_images))
        #print(list_m_images.keys())
        print(len(list_syn))
        #print(list_syn[0])
        #print(list_m_images[0])
        #print(list_m_physio[0])
        print(counter_image)
        #print(list_m_physio.keys())
        #print(len(list_gtA))
        #print(list_gtA.keys())
        
        #exit(0)
        t_listGT = []
        t_listMeta = []
        #1,2,3 = train 1,2,3
        #11,12,13 = dev 1,2,3
        #21,22,23 = test 1,2,3 
        
        t_listImg = []
        t_listLdmrk = []
        t_listPhy = []
        t_listAud = []
        
        #Now aligning them according to the groundtruth  to one long list 
        for key in listKey : 
            print(key)
            
            #first get the val and arousal with the cor time
            listA = list_gtA[key]
            listV = list_gtV[key]
            
            #First ensure the time are the same of V and A\
            #print('gt : ',len(listA),len(listV))
            assert(len(listA[0]) == len(listV[0])) 
            #print('ok')
            
            
            l_data=[]
            #Now merge the V and A as one long list
            for x,y in zip(listA,listV): 
                t_listGT.append([x[1],y[1]])
                t_listMeta.append([x[0],y[0],convertName(key)])
                l_data.append([x[0],y[0],convertName(key)])
            #print(t_listGT[0])
            #Now the actual data 
            #the image
            t_listIm = list_m_images[key]
            t_listSyn = list_syn[key]
            t_listLm = list_m_ldmrks[key]
            
            #print(listIm)
            
            #check the length is the same
            #print('img : ',len(t_listIm),len(t_listSyn))
            assert(len(t_listIm) == len(t_listSyn))
            
            #print('syn',listSyn[5][1])
            
            #Now get the data according the timeframe, if missing will attempt to replicate the data to fill in
            #print('l t',len(l_data))
            i = 0
            j = 0
            for timeFrame in l_data : 
                t = timeFrame[0]
                #print('d',i,len(t_listSyn),j,t,key,len(l_data))
                
                if ((j == len(l_data)-1) and i == len(t_listSyn)):  #anomaly
                    t_listImg.append(t_listIm[i-1])
                    t_listLdmrk.append(t_listLm[i-1])
                    
                elif t_listSyn[i][1] == t:
                    #print('equal',i,j,t,key,len(l_data)) 
                    t_listImg.append(t_listIm[i])
                    t_listLdmrk.append(t_listLm[i])
                    i += 1
                else : #missing, fill with the 
                    #print('missing',i,key)
                    t_listImg.append(t_listIm[i-1])
                    t_listLdmrk.append(t_listLm[i-1])
                j+=1
                
                
            #Now the physio, it has 10 data/second
            dps = 50
            listP = list_m_physio[key]
            #print(len(listP))
            
            meanECG = 90
            meanEDA = 0
            
            '''multiplier = 2;
            i = 0
            j = 0
            for timeFrame in l_data : 
                t = timeFrame[0]
                  
                if listP[i][0] == t:
                    #print('equal',i,j,t,key,len(listP),len(l_data))
                    temp = []
                    for k in range(-5,5) : #get 5 temporal windows
                        idx = k+i
                        if idx < 0 or (idx >= len(listP)): 
                            temp.append([meanECG, meanEDA])
                        else :   
                            temp.append([listP[idx][1],listP[idx][2]])
                        i += 1 
                    t_listPhy.append(temp)
                j+=1'''
            
            #ECG, EDA
            dps = 10
            multiplier = 5;
            llim = -int(truediv(dps,2))
            rlim = int(truediv(dps,2))
            i = 0
            j = 0
            for timeFrame in l_data : 
                t = timeFrame[0]
                  
                if listP[i][0] == t:
                    #print('equal',i,j,t,key,len(listP),len(l_data))
                    temp = []
                    for k in range(llim, rlim) : #get 5 temporal windows
                        idx = k+i
                        for l in range(multiplier):
                            idx2 = idx*multiplier
                            if idx2 < 0 or (idx2 >= len(listP)): 
                                temp.append([meanECG, meanEDA])
                            else :   
                                temp.append([listP[idx2][1],listP[idx2][2]])
                            
                        i += 1 
                    t_listPhy.append(temp)
                j+=1
    
            #Audio
            listAud = list_m_audios[key]
            l = -truediv(sps,2)
            r = truediv(sps,2)

            i = 0
            j = 0
            for z,timeFrame in enumerate(l_data,0) : #we don't synchronize the sound. assume they are synchorinzed
                
                #print(z,len(listAud))
                
                if z >= len(listAud): 
                    z = len(listAud)-1
                
                
                t = timeFrame[0]
                
                signald = listAud[z]
                il = l*z
                
                if il < 0 : 
                    il = 0
                
                ir = il+sps
                
                if ir > len(signald): 
                    ir = len(signald)
                
                il = int(il)
                ir = int(ir)
                
                z = np.zeros(sps)
                z[:len(signald[il:ir])] = signald[il:ir] 
                    
                t_listAud.append(z)
                j+=1
        
        
        self.t_listGT = t_listGT
        self.t_listImg = t_listImg
        self.t_listPhy = t_listPhy
        self.t_listMeta = t_listMeta
        self.t_listLdm = t_listLdmrk
        self.t_listAud = t_listAud
        
        '''print(len(t_listGT))
        print(len(t_listImg))
        print(len(t_listPhy))
        print(len(t_listMeta))
        print(len(t_listLdmrk))'''
        '''
        self.t_listGT = t_listGT
        self.t_listImg = t_listImg
        self.t_listPhy = t_listPhy
        self.t_listMeta = t_listMeta
        self.t_listLdm = []
        
        print(len(t_listGT))
        print(len(t_listImg))
        print(len(t_listPhy))
        print(len(t_listMeta))
        exit(0)'''

    def __getitem__(self,index):
        #output : 
        #input : [B,img(R,G,B),physioECG[10],physioEDA[10]]
        #label : [B,Val,Ar]
        #meta  : [B,time,name] 
        
        x = self.t_listImg[index]
        m_ldmrk = np.asarray(utils.read_kp_file(self.t_listLdm[index])).flatten('F')
        m_bioECG = self.t_listPhy[index]
        m_audio = self.t_listAud[index]
        o_gt = self.t_listGT[index]
        o_meta = self.t_listMeta[index]
                
        tImage = Image.open(x).convert("RGB")
        
        if self.onlyFace :    
            #crop the face region
            #t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 2,div_y = 2,images = cv2.imread(x),displacementxy = random.uniform(-.5,.5))
            t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = m_ldmrk.copy(),div_x = 8,div_y = 8,images = cv2.imread(x))#,displacementxy = random.uniform(-.5,.5))
            
            area = (x1,y1, x2,y2)
            tImage =  tImage.crop(area)
            
            m_ldmrk[:68] -= x_min
            m_ldmrk[68:] -= y_min
            
            tImage = tImage.resize((self.imageWidth,self.imageWidth))
            
            m_ldmrk[:68] *= truediv(self.imageWidth,(x2 - x1))
            m_ldmrk[68:] *= truediv(self.imageHeight,(y2 - y1))
        
        label = m_ldmrk
        if self.augment : 
            sel = np.random.randint(0,4)
            #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
            if sel == 0 : 
                pass
            elif sel == 1 : 
                flip = RandomHorizontalFlip_WL(1)
                tImage,label,_ = flip(tImage,label)
            elif sel == 2 : 
                rot = RandomRotation_WL(45)
                tImage,label,_ = rot(tImage,label)
            elif sel == 3 : 
                occ = Occlusion_WL(1)
                tImage,label,_ = occ(tImage,label)
                
            #random crop
            if True : 
                rc = RandomResizedCrop_WL(size = self.imageSize,scale = (0.75,1), ratio = (0.75, 1.33))
                tImage,label,_= rc(tImage,label)
            
            #additional blurring
            if (np.random.randint(1,3)%2==0) and False: 
                sel_n = np.random.randint(1,3)
                rc = GeneralNoise_WL(1)
                tImage,label= rc(tImage,label,sel_n,2)
            
        if self.useIT : 
            tImage = self.transformInternal(tImage)
        else : 
            tImage = self.transform(tImage)
        
        return tImage,torch.FloatTensor(m_bioECG),torch.FloatTensor(o_gt),o_meta,torch.FloatTensor(m_audio)
    
    
    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    
    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    
    def __len__(self):
        return len(self.t_listImg)




class AVChallenge(data.Dataset):
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["AVC-Short"],dir_gt = None,onlyFace = True, image_size =224, 
                 transform = None,useIT = False,augment = False, step = 1,isTest = False,wHeatmap= False,isVideo = False, seqLength = None, 
                 nSplit = 5, listSplit = [0,1,2,3,4],split = False):
        
        self.isTest = isTest
        self.seq_length = seqLength 
        self.isVideo = isVideo
        
        self.transform = transform
        self.onlyFace = onlyFace
        self.augment = augment 
        self.wHeatmap = wHeatmap
        
        self.imageSize = image_size
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.useIT = useIT
        
        if dir_gt is None : 
            self.curDir = rootDir#"/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data"
        else : 
            self.curDir = dir_gt
        
        
        list_gt = []
        list_labels_t = []
        list_labels_tE = []
        
        counter_image = 0
        annotL_name = 'annot'
        annotE_name = 'annot2'
        
        dirVideo = 'video'
        dirLandmark = 'landmarks'
        dirBBox = 'bbox'
        dirValence = 'valence'
        dirArousal = 'arousal'
        
        
        l_flatten_imgs = []
        l_flatten_bbs = []
        l_flatten_kps = []
        l_flatten_va = []
        
        l_record = {}
        listKey = []
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        for data in data_list :
            self.datasetName = data
            
            print(("Opening "+data+'/'+dirVideo))
            
            for f in file_walker.walk(self.curDir +'/'+data+'/'+dirVideo):
                
                #entity numbers is based on the video numbers
                        
                l_img = []
                l_lndmark = []
                l_bb = []
                l_va = []
                
                if f.isDirectory: # Check if object is directory
                    #print((f.name, f.full_path)) # Name is without extension
                    #get the name 
                    name = f.name 
                    listKey.append(name)
                    
                    print(name)
                    
                    #if the test, get the bb and va 
                    if not self.isTest :
                        v_files = self.curDir +'/'+data+'/'+dirValence+'/'+name+".txt"
                        a_files = self.curDir +'/'+data+'/'+dirArousal+'/'+name+".txt"
                        
                        print(v_files)
                        print(a_files)
                        
                        #now read the gt :    
                        with open(v_files) as file:
                            data_v = [re.split(r'\t+',l.strip()) for l in file]
                            
                        with open(a_files) as file:
                            data_a = [re.split(r'\t+',l.strip()) for l in file]
                            
                        for a,v in zip(data_a,data_v):
                            l_va.append([float(a[0]),float(v[0])])
                            l_flatten_va.append([float(a[0]),float(v[0])])
                        
                        #print(l_va)
                        '''for i in range(len(data2)) :
                            x.append([ float(j) for j in data2[i][0].split()] )
                            
                        lbl_2.append(np.array(x).flatten('F'))'''
                    
                         
                    ldmarkDir = self.curDir +'/'+data+'/'+dirLandmark+'/'+name
                    bbDir = self.curDir +'/'+data+'/'+dirBBox+'/'+name
                    
                    print(ldmarkDir, bbDir)
                    print(f.full_path)
                    
                    prevKP = prevBB = None
                    
                    #First get the file list
                    listImg = sorted(glob.iglob(os.path.join(f.full_path, '*.jpg')))
                    for imgFile in listImg : 
                        #print(imgFile)
                        
                        #now for each data, get corresponding ldmrk and bb
                        head, tail = os.path.split(imgFile)
                        tail = str(int(tail.split('.')[0]))
                        
                        ldmarkFile = ldmarkDir+'/'+tail+'.pts'
                        bbFile = bbDir+'/'+tail+'.pts'
                        
                        if prevKP is None : 
                            prevKP = ldmarkFile
                            prevBB = bbFile
                        
                        l_img.append(imgFile)
                        l_flatten_imgs.append(imgFile)
                        
                        if os.path.exists(bbFile):
                            d = utils.read_bb_file(bbFile).flatten('F')
                            l_bb.append(d)
                            l_flatten_bbs.append(d)
                            
                            prevBB = bbFile 
                        else : 
                            d = utils.read_bb_file(prevBB).flatten('F') 
                            l_bb.append(d)
                            l_flatten_bbs.append(d)
                        
                        if os.path.exists(ldmarkFile):
                            d = utils.read_kp_file(ldmarkFile).flatten('F') 
                            l_lndmark.append(d)
                            l_flatten_kps.append(d)
                            
                            prevKP = ldmarkFile
                        else : 
                            d = utils.read_kp_file(prevKP).flatten('F') 
                            l_lndmark.append(d) 
                            l_flatten_kps.append(d)
                    
                    if self.isTest: 
                        l_va = np.zeros([len(l_lndmark),2])
                    #if the test, get the bb and va 
                    l_record[name] = [l_img,l_lndmark,l_bb,l_va]
                    
        
        listKey = sorted(listKey)
        
        for key in listKey : 
            data = l_record[key]
            print(len(data[0]))
            print(len(data[1]))
            print(len(data[2]))
            print(len(data[3]))
        
        if not self.isVideo : #if is not video, flatten it 
            #this provides
            #l_imgs = [totalLength, imgs]
            #l_kp = [totalLength, 136]
            
            t_imgs = l_flatten_imgs 
            t_bbs = l_flatten_bbs
            t_kps = l_flatten_kps
            
            if self.isTest : 
                t_va = np.zeros([len(l_flatten_kps),2])
            else : 
                t_va = l_flatten_va
            
            if split :
                print('splitting')
                self.l_imgs = [] 
                self.l_bbs = []  
                self.l_kps = [] 
                self.l_va = []
                
                totalData = len(t_imgs)
                perSplit = int(truediv(totalData, nSplit))
                for x in listSplit :
                    print('split : ',x) 
                    begin = x*perSplit
                    if x == nSplit-1 : 
                        end = begin + (totalData - begin)
                    else : 
                        end = begin+perSplit
                    print(begin,end,totalData)
                    for x2 in range(begin,end) : 
                        #print(x2,totalData)
                        self.l_imgs.append(t_imgs[x2])
                        self.l_bbs.append(t_bbs[x2])
                        self.l_kps.append(t_kps[x2])
                        self.l_va.append(t_va[x2])
                
                self.l_bbs = np.asarray(self.l_bbs)
                self.l_kps = np.asarray(self.l_kps)
                self.l_va = np.asarray(self.l_va)
            else :
                self.l_imgs = t_imgs 
                self.l_bbs = np.asarray(t_bbs)  
                self.l_kps = np.asarray(t_kps) 
                self.l_va = np.asarray(t_va)             
            
            print(len(self.l_imgs),len(self.l_bbs),len(self.l_kps),len(self.l_va))
            '''print(self.l_bbs[0])
            print(self.l_kps[0])
            print(self.l_va[0])'''
            
        else :
            #this provides
            #l_imgs = [N, seq, imgs]
            #l_kp = [N, seq, 136]
            #need to be repaired further
            
            self.l_imgs = []
            self.l_bbs = np.zeros([int(counter_image/(self.seq_length*step)),self.seq_length,2])
            self.l_kps = np.zeros([int(counter_image/(self.seq_length*step)),self.seq_length,136])
            self.l_va = np.zeros([int(counter_image/(self.seq_length*step)),self.seq_length,2])
            
            indexer = 0
            for key in listKey :
                
                counter = 0
                data = l_record[key]
                
                for i in range(int(len(data[0])/(self.seq_length*step))):
                    t_imgs = []
                    t_bbs = [self.seq_length,8]
                    t_kps = [self.seq_length,136]
                    t_va = [self.seq_length,2]
                     
                    i_temp = 0
                    for j in range(counter,counter+(self.seq_length*step),step): 
                        t_imgs.append(data[0][j]) 
                        t_bbs[i_temp] = data[1][j]
                        t_kps[i_temp] = data[2][j]
                        t_va[i_temp] = data[3][j]
                        
                        i_temp+=1
                    self.l_imgs.append(t_imgs)
                    self.l_bbs[indexer] = t_bbs
                    self.l_kps[indexer] = t_kps
                    self.l_va[indexer] = t_va
                        
                    indexer += 1
                    counter+=self.seq_length*step
        
        print(len(self.l_imgs))
        print(len(self.l_bbs))
        print(len(self.l_kps))
        print(len(self.l_va))
        print('done')
        
    def __getitem__(self,index):
        #Read all data, transform etc.
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
        
        l_imgs = []; l_VA = []; l_ldmrk = []; l_nc = []#,torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]
        
        if not self.isVideo : 
            x_l = [self.l_imgs[index]];labelE_l =[self.l_va[index].copy()];label_l = [self.l_kps[index].copy()];label_n =[self.l_bbs[index]] 
        else : 
            x_l = self.l_imgs[index];labelE_l =self.l_va[index].copy();label_l = self.l_kps[index].copy();label_n =self.l_bbs[index]
        
        for x,labelE,label,ln in zip(x_l,labelE_l,label_l,label_n) : 
            #print(x,labelE,label,ln)
            tImage = Image.open(x).convert("RGB")
            tImageB = None
            
            if self.onlyFace :    
                #crop the face region
                #t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 2,div_y = 2,images = cv2.imread(x),displacementxy = random.uniform(-.5,.5))
                t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label.copy(),div_x = 8,div_y = 8,images = cv2.imread(x))#,displacementxy = random.uniform(-.5,.5))
                
                area = (x1,y1, x2,y2)
                tImage =  tImage.crop(area)
                
                label[:68] -= x_min
                label[68:] -= y_min
                
                tImage = tImage.resize((self.imageWidth,self.imageWidth))
                
                label[:68] *= truediv(self.imageWidth,(x2 - x1))
                label[68:] *= truediv(self.imageHeight,(y2 - y1))
            
            newChannel = None
            
            if self.wHeatmap : 
                theMiddleName = self.datasetName
                
                filePath = x.split(os.sep)
                ifolder = filePath.index(theMiddleName)
                
                print(ifolder)
                image_name = filePath[-1]
                
                annot_name_H = os.path.splitext(image_name)[0]+'.npy'
                
                sDirName = filePath[:ifolder]
                dHeatmaps = '/'.join(sDirName)+'/heatmaps'
                
                finalTargetH = dHeatmaps+'/'+annot_name_H
                print(finalTargetH)
                
                if isfile(finalTargetH) and False: 
                    newChannel  = np.load(finalTargetH)
                    newChannel = Image.fromarray(newChannel)
                else : 
                    checkDirMake(dHeatmaps)
                    
                    tImageTemp = cv2.cvtColor(np.array(tImage),cv2.COLOR_RGB2BGR)
                    #tImageTemp = cv2.imread(x)#tImage.copy()
                
                    b_channel,g_channel,r_channel = tImageTemp[:,:,0],tImageTemp[:,:,1],tImageTemp[:,:,2]
                    newChannel = b_channel.copy(); newChannel[:] = 0
                    
                    t0,t1,t2,t3 = utils.get_bb(label[0:68], label[68:])
                    
                    l_cd,rv = utils.get_list_heatmap(0,None,t2-t0,t3-t1,.05)
                    height, width,_ = tImageTemp.shape
                    
                    wx = t2-t0
                    wy = t3-t1
                
                    scaler = 255/np.max(rv)
                    
                    for iter in range(68) :
                        ix,iy = int(label[iter]),int(label[iter+68])
                        
                        #Now drawing given the center
                        for iter2 in range(len(l_cd)) : 
                            value = int(rv[iter2]*scaler)
                            if newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] < value : 
                                newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] = int(rv[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)
                    
                    '''tImage2 = cv2.merge((b_channel, newChannel,newChannel, newChannel))
                    cv2.imshow("combined",tImage2)
                    cv2.waitKey(0)'''
                    
                    np.save(finalTargetH,newChannel)
                    newChannel = Image.fromarray(newChannel)
            
            if self.augment : 
                sel = np.random.randint(0,4)
                #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
                if sel == 0 : 
                    pass
                elif sel == 1 : 
                    flip = RandomHorizontalFlip_WL(1)
                    tImage,label,newChannel = flip(tImage,label,newChannel)
                elif sel == 2 : 
                    rot = RandomRotation_WL(45)
                    tImage,label,newChannel = rot(tImage,label,newChannel)
                elif sel == 3 : 
                    occ = Occlusion_WL(1)
                    tImage,label,newChannel = occ(tImage,label,newChannel)
                    
                #random crop
                if True : 
                    rc = RandomResizedCrop_WL(size = self.imageSize,scale = (0.75,1), ratio = (0.75, 1.33))
                    tImage,label,newChannel= rc(tImage,label,newChannel)
                
                #additional blurring
                if (np.random.randint(1,3)%2==0) and False: 
                    sel_n = np.random.randint(1,3)
                    rc = GeneralNoise_WL(1)
                    tImage,label= rc(tImage,label,sel_n,2)
            
            if self.useIT : 
                tImage = self.transformInternal(tImage)
            else : 
                tImage = self.transform(tImage)
            
            if not self.wHeatmap : 
                l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label))#,x,self.list_gt_names[index]
            else : 
                newChannel = transforms.Resize(224)(newChannel)
                newChannel = transforms.ToTensor()(newChannel)
                newChannel = newChannel.sub(125)
                l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label)); l_nc.append(newChannel)
                #return tImage,torch.FloatTensor(labelE),torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]
                
        if not self.isVideo : 
            if self.wHeatmap : 
                return l_imgs[0], l_VA[0], l_ldmrk[0], l_nc[0]
            else : 
                return l_imgs[0], l_VA[0], l_ldmrk[0]
        else : 
            #lImgs = torch.Tensor(len(l_imgs),3,self.imageHeight,self.imageWidth)
            #lVA = torch.Tensor(len(l_VA),2)
            #lLD = torch.Tensor(len(l_ldmrk),136)
            lImgs = torch.stack(l_imgs)
            lVA = torch.stack(l_VA)
            lLD = torch.stack(l_ldmrk)
            
            #print(lImgs.shape, l_imgs[0].shape, l_VA[0].shape,len(lImgs))
            
            #torch.cat(l_imgs, out=lImgs)
            #torch.cat(l_VA, out=lVA)
            #torch.cat(l_ldmrk, out=lLD)
            
            if self.wHeatmap :
                #lnc = torch.Tensor(len(l_nc),1,self.imageHeight,self.imageWidth)
                #torch.cat(l_nc, out=lnc)
                lnc = torch.stack(l_nc)
                 
                return lImgs, lVA, lLD, lnc 
            else : 
                return lImgs, lVA, lLD
                
    
    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    
    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    
    def __len__(self):
        return len(self.l_imgs)



def readCSV(fileName):
    if '.csv' in fileName :     
        list_dta = []
        with open(fileName, 'r') as csvFile:
            reader = csv.reader(csvFile,delimiter=';')
            for row in reader:
                #print('frame' in row[0],row[0].split(',')[1])
                if not 'frame' in row[0] : 
                    list_dta.append([float(row[0].split(',')[1])])
        return list_dta
        #print(fileName,'data : ',list_dta)
        #return sorted(list_dta)
        #print(fileName,'data : ',list_dta)
    else : 
        return None


class SEWAFEW(data.Dataset):
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["AFEW"],dir_gt = None,onlyFace = True, image_size =224, 
                 transform = None,useIT = False,augment = False, step = 1,split = False, 
                 nSplit = 5, listSplit = [0,1,2,3,4],wHeatmap= False,isVideo = False, seqLength = None,
                 returnM = False, toAlign = False, dbType = 0):#dbtype 0 is AFEW, 1 is SEWA
        
        self.dbType = dbType
        
        self.seq_length = seqLength 
        self.isVideo = isVideo
        self.align = toAlign
        self.useNudget = False
        self.returnM = returnM
        
        self.transform = transform
        self.onlyFace = onlyFace
        self.augment = augment 
        self.wHeatmap = wHeatmap
        
        self.imageSize = image_size
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.useIT = useIT
        self.curDir = rootDir+"/"#/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/"
        
        
        if self.dbType ==1 : 
            annotL_name = "annotOri"
            self.ldmrkNumber = 49
            self.nose = 16
            self.leye = 24
            self.reye = 29
            #mean_shape49-pad3-224
            
            self.mean_shape = np.load(curDir+'mean_shape49-pad3-'+str(image_size)+'.npy')
        else :
            annotL_name = 'annot'
            self.ldmrkNumber = 68
            self.nose = 33
            self.leye = 41
            self.reye = 46
            
            self.mean_shape = np.load(curDir+'mean_shape-pad-'+str(image_size)+'.npy')
        
        self.swap = False 
        
        if self.swap : 
            self.ptsDst = np.asarray([
                [self.mean_shape[self.nose+self.ldmrkNumber],self.mean_shape[self.nose]],[self.mean_shape[self.leye+self.ldmrkNumber],self.mean_shape[self.leye]],[self.mean_shape[self.reye+self.ldmrkNumber],self.mean_shape[self.reye]]
                ],dtype= np.float32)
            
            self.ptsTn = [self.mean_shape[self.nose+self.ldmrkNumber],self.mean_shape[self.nose]],[self.mean_shape[self.leye+self.ldmrkNumber],self.mean_shape[self.leye]],[self.mean_shape[self.reye+self.ldmrkNumber],self.mean_shape[self.reye]]
        else : 
            self.ptsDst = np.asarray([
                [self.mean_shape[self.nose],self.mean_shape[self.nose+self.ldmrkNumber]],[self.mean_shape[self.leye],self.mean_shape[self.leye+self.ldmrkNumber]],[self.mean_shape[self.reye],self.mean_shape[self.reye+self.ldmrkNumber]]
                ],dtype= np.float32)
            self.ptsTn = [self.mean_shape[self.nose],self.mean_shape[self.nose+self.ldmrkNumber]],[self.mean_shape[self.leye],self.mean_shape[self.leye+self.ldmrkNumber]],[self.mean_shape[self.reye],self.mean_shape[self.reye+self.ldmrkNumber]]
            
            self.ptsTnFull = np.column_stack((self.mean_shape[:self.ldmrkNumber],self.mean_shape[self.ldmrkNumber:]))
        
        list_gt = []
        list_labels_t = []
        list_labels_tE = []
        
        counter_image = 0
        
        annotE_name = 'annot2'
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        list_missing = []
        
        for data in data_list : 
            print(("Opening "+data))
            for f in file_walker.walk(self.curDir +data+"/"):
                if f.isDirectory: # Check if object is directory
                    #print((f.name, f.full_path)) # Name is without extension
                    #c_image,c_ldmark = 0,0
                    
                    if self.dbType == 1 : #we directly get the VA file in case of sewa  
                        #first get the valence 
                        valFile = f.full_path+"/valence/"+f.name+"_Valence_A_Aligned.csv"
                        aroFile = f.full_path+"/arousal/"+f.name+"_Arousal_A_Aligned.csv"
                        
                        list_labels_tE.append([valFile,aroFile])
                        #print(valFile,aroFile)
                        
                    for sub_f in f.walk():
                        if sub_f.isDirectory: # Check if object is directory
                            list_dta = []
                            
                            #print(sub_f.name)
                            if(sub_f.name == annotL_name) : #If that's annot, add to labels_t
                                
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_labels_t.append(sorted(list_dta))
                                c_image = len(list_dta)
                            elif(sub_f.name == 'img'): #Else it is the image
                                
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_gt.append(sorted(list_dta))
                                counter_image+=len(list_dta)
                                c_ldmrk = len(list_dta)
        
                            elif (sub_f.name == annotE_name) :
                                if self.dbType == 0 : 
                                    #If that's annot, add to labels_t
                                    for sub_sub_f in sub_f.walk(): #this is the data
                                        if(".npy" not in sub_sub_f.full_path):
                                            list_dta.append(sub_sub_f.full_path)
                                    list_labels_tE.append(sorted(list_dta))
                                    
                    if(c_image!=c_ldmrk) and False: 
                        print(f.full_path," is incomplete ",'*'*10,c_image,'-',c_ldmrk)
                        ori = "/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/allVideo/"
                        target = '/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/allVideo/retrack/'
                        #shutil.copy(ori+f.name+".avi",target+f.name+".avi")
                        list_missing.append(f.name)
                                    
        self.length = counter_image
        print("Now opening keylabels")
        
        '''l_target = "/home/deckyal/missingSewa.txt"
        import pickle
        
        with open(l_target, 'wb') as fp:
            pickle.dump(list_missing, fp)
        
        with open (l_target, 'rb') as fp:
            itemlist = pickle.load(fp)
        print(itemlist)
        
        exit(0)'''
        
        list_labelsN = [] 
        list_labelsEN = []
        
        list_labels = [] 
        list_labelsE = []
        
        for ix in range(len(list_labels_t)) : #lbl,lble in (list_labels_t,list_labels_tE) :
            lbl_68 = [] #Per folder
            lbl_2 = [] #Per folder
            
            lbl_n68 = [] #Per folder
            lbl_n2 = [] #Per folder
            for jx in range(len (list_labels_t[ix])): #lbl_sub in lbl :
                
                #print(os.path.basename(list_gt[ix][jx]))
                #print(os.path.basename(list_labels_t[ix][jx]))
                #print(os.path.basename(list_labels_tE[ix][jx]))
                
                lbl_sub = list_labels_t[ix][jx]
                if ('pts' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    lbl_68.append(read_kp_file(lbl_sub,True))
                        
                    '''with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record'''
                    
                    lbl_n68.append(lbl_sub)
                
                if self.dbType == 0 : 
                    lbl_subE = list_labels_tE[ix][jx]
                    if ('aro' in lbl_subE) : 
                        x = []
                        #print(lbl_sub)
                        with open(lbl_subE) as file:
                            data2 = [re.split(r'\t+',l.strip()) for l in file]
                        for i in range(len(data2)) :
                            #x.append([ float(j) for j in data2[i][0].split()] )
                            temp = [ float(j) for j in data2[i][0].split()]
                            temp.reverse() #to give the valence first. then arousal
                            x.append(temp)
                        
                        #x.reverse()
                        
                        lbl_2.append(np.array(x).flatten('F'))
                        lbl_n2.append(lbl_sub)
                
            if self.dbType == 1 : #sewa  
                #print(list_labels_t[ix][0])
                valFile = np.asarray(readCSV(list_labels_tE[ix][0]))
                aroFile = np.asarray(readCSV(list_labels_tE[ix][1]))
                
                lbl_n2.append(list_labels_tE[ix][0])
                lbl_2 = np.column_stack((valFile,aroFile))
                
            
            list_labelsN.append(lbl_n68)
            list_labelsEN.append(lbl_n2)
            
            list_labels.append(lbl_68)
            list_labelsE.append(lbl_2)
        
            
        '''for lbl in list_labels_t :
            lbl_68 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('pts' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
            list_labels.append(lbl_68)
        
        list_labelsE = []
        for lbl in list_labels_tE :
            lbl_2 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('aro' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_2.append(np.array(x).flatten('F')) #1 record
            list_labelsE.append(lbl_2)'''
        
        
        
        '''print(len(list_labelsN[0])) 
        print(len(list_labelsEN[0])) 
        print(len(list_labels[0])) 
        print(len(list_labelsE[0]))'''
                
        
        t_l_imgs = []
        t_l_gt = []
        t_l_gtE = []
        
        t_list_gt_names = []
        t_list_gtE_names = []
        
        #print(list_labelsEN)
        
        if not self.isVideo :
            #Flatten it to one list
            for i in range(0,len(list_gt)): #For each dataset
                
                list_images = []
                list_gt_names = []
                list_gtE_names = []
                indexer = 0
                
                list_ground_truth = np.zeros([len(list_gt[i]),self.ldmrkNumber*2])
                list_ground_truthE = np.zeros([len(list_gt[i]),2])
                
                for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                    list_images.append(list_gt[i][j])
                    
                    list_gt_names.append(list_labelsN[i][j])
                    if self.dbType == 0 : 
                        list_gtE_names.append(list_labelsEN[i][j])
                    else : 
                        list_gtE_names.append(list_labelsEN[i][0])
                    #print(list_labelsEN[i])
                    
                    '''if len(list_labels[i][j] < 1): 
                        print(list_labels[i][j])'''
                    #print(len(list_labels[i][j]))
                    list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                    list_ground_truthE[indexer] = np.array(list_labelsE[i][j]).flatten('F')
                    indexer += 1
                
                t_l_imgs.append(list_images)
                t_l_gt.append(list_ground_truth)
                t_l_gtE.append(list_ground_truthE)
                
                t_list_gt_names.append(list_gt_names)
                t_list_gtE_names.append(list_gtE_names)
        
        else : 
            if self.seq_length is None :
                list_ground_truth = np.zeros([int(counter_image/(self.seq_length*step)),self.seq_length,136])
                indexer = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                    counter = 0
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize
                        
                        temp = []
                        temp2 = np.zeros([self.seq_length,136])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z])
                            temp2[i_temp] = list_labels[i][z]
                            i_temp+=1
                            
                        list_images.append(temp)
                        list_ground_truth[indexer] = temp2
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                self.l_imgs = list_images
                self.l_gt = list_ground_truth
            else : 
                counter_seq = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                
                    indexer = 0;    
                    list_gt_names = []
                    list_gtE_names = []
                    
                    list_ground_truth = np.zeros([int(len(list_gt[i])/(self.seq_length*step)),self.seq_length,136]) #np.zeros([counter_image,136])
                    list_ground_truthE = np.zeros([int(len(list_gt[i])/(self.seq_length*step)),self.seq_length,2])#np.zeros([counter_image,2])
                    
                    counter = 0
                    list_images = []
                    
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize   
                        tmpn68 = []
                        tmpn2 = []
                        
                        temp = []
                        temp2 = np.zeros([self.seq_length,136])
                        temp3 = np.zeros([self.seq_length,2])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z])
                            temp2[i_temp] = list_labels[i][z].flatten('F')
                            temp3[i_temp] = list_labelsE[i][z].flatten('F')
                             
                            tmpn68.append(list_labelsN[i][z])
                            tmpn2.append(list_labelsEN[i][z])
                            
                            i_temp+=1
                            counter_seq+=1
                            
                        list_images.append(temp)
                        list_ground_truth[indexer] = temp2
                        list_ground_truthE[indexer] = temp3
                        
                        list_gt_names.append(tmpn68)
                        list_gtE_names.append(tmpn2)
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                        
                    t_l_imgs.append(list_images)
                    t_l_gt.append(list_ground_truth)
                    t_l_gtE.append(list_ground_truthE)
                    
                    t_list_gt_names.append(list_gt_names)
                    t_list_gtE_names.append(list_gtE_names)
                
        '''print('length : ',len(t_l_imgs)) #Folder
        print('lengt2 : ',len(t_l_imgs[0])) #all/seq
        print('lengt3 : ',len(t_l_imgs[0][0])) #seq
        print('length4 : ',len(t_l_imgs[0][0][0]))'''
        
        #[folder, all/seq,seq]
        
        self.l_imgs = []
        self.l_gt = []
        self.l_gtE = []
        
        self.list_gt_names = []
        self.list_gtE_names = []
        
        #print('cimage : ',counter_image)
        
        
        if split :
            '''print('splitting')
            self.l_imgs = []
            self.l_gt = []
            self.l_gtE = []
            
            totalData = len(list_images)
            perSplit = int(truediv(totalData, nSplit))
            for x in listSplit :
                print('split : ',x) 
                begin = x*perSplit
                if x == nSplit-1 : 
                    end = begin + (totalData - begin)
                else : 
                    end = begin+perSplit
                print(begin,end,totalData)
                for x2 in range(begin,end) : 
                    #print(x2,totalData)
                    self.l_imgs.append(list_images[x2])
                    self.l_gt.append(list_ground_truth[x2])
                    self.l_gtE.append(list_ground_truthE[x2])'''
            indexer = 0 
            
            self.l_gt = []
            self.l_gtE = []
            '''else : 
                self.l_gt = np.zeros([counter_seq,self.seq_length,136])
                self.l_gtE = np.zeros([counter_seq,self.seq_length,2])'''
                
            totalData = len(t_l_imgs)
            perSplit = int(truediv(totalData, nSplit))
            for x in listSplit :
                print('split : ',x) 
                begin = x*perSplit
                if x == nSplit-1 : 
                    end = begin + (totalData - begin)
                else : 
                    end = begin+perSplit
                print(begin,end,totalData)
                
                if not self.isVideo :
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): 
                            #print('append ',t_l_imgs[i][j])
                            self.l_imgs.append(t_l_imgs[i][j])
                            self.l_gt.append(t_l_gt[i][j])
                            self.l_gtE.append(t_l_gtE[i][j])
                            
                            self.list_gt_names.append(t_list_gt_names[i][j])
                            self.list_gtE_names.append(t_list_gtE_names[i][j])
                            indexer+=1
                            
                else : 
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): #seq counter
                        
                            t_img = []
                            t_gt = []
                            t_gtE = []
                            t_gt_N = []
                            t_gt_EN = []
                            tmp = 0
                            
                            for k in range(len(t_l_imgs[i][j])): #seq size
                                t_img.append(t_l_imgs[i][j][k])
                                t_gt.append(t_l_gt[i][j][k])
                                t_gtE.append(t_l_gtE[i][j][k])
                                
                                t_gt_N.append(t_list_gt_names[i][j][k])
                                t_gt_EN.append(t_list_gtE_names[i][j][k])
                                tmp+=1
                                
                            #print('append ',t_img)
                            self.l_imgs.append(t_img)
                            self.l_gt.append(t_gt)
                            self.l_gtE.append(t_gtE)
                            
                            self.list_gt_names.append(t_gt_N)
                            self.list_gtE_names.append(t_gt_EN)
                            indexer+=1
                    
                print(len(self.l_imgs))
                    
            self.l_gt = np.asarray(self.l_gt)
            self.l_gtE = np.asarray(self.l_gtE)
        else :
            if not self.isVideo :
                self.l_gt = np.zeros([counter_image,136])
                self.l_gtE = np.zeros([counter_image,2])
                indexer = 0
                
                
                for i in range(len(t_l_imgs)): 
                    for j in range(len(t_l_imgs[i])): 
                        self.l_imgs.append(t_l_imgs[i][j])
                        print(i,j,'-',len(t_l_imgs[i]))
                        self.l_gt[indexer] = t_l_gt[i][j]
                        self.l_gtE[indexer] = t_l_gtE[i][j]
                        
                        self.list_gt_names.append(t_list_gt_names[i][j])
                        self.list_gtE_names.append(t_list_gtE_names[i][j])
                        indexer+=1
                    
            else : 
                self.l_gt= np.zeros([counter_seq,self.seq_length,136])
                self.l_gtE = np.zeros([counter_seq,self.seq_length,2])
                
                indexer = 0
                
                for i in range(len(t_l_imgs)): #dataset
                    for j in range(len(t_l_imgs[i])): #seq counter
                        
                        t_img = []
                        
                        t_gt = np.zeros([self.seq_length,136])
                        t_gte = np.zeros([self.seq_length,2])
                        
                        t_gt_n = []
                        t_gt_en = []
                        i_t = 0
                        
                        for k in range(len(t_l_imgs[i][j])): #seq size
                            
                            t_img.append(t_l_imgs[i][j][k])
                            t_gt[i_t] = t_l_gt[i][j][k]
                            t_gte[i_t] = t_l_gtE[i][j][k]
                            
                            t_gt_n.append(t_list_gt_names[i][j][k])
                            t_gt_en.append(t_list_gtE_names[i][j][k])
                            
                            i_t+=1
                            
                        self.l_imgs.append(t_img)
                        self.l_gt[indexer] = t_gt
                        self.l_gtE[indexer] = t_gte
                        
                        self.list_gt_names.append(t_gt_n)
                        self.list_gtE_names.append(t_gt_en)
                        
                        indexer+=1
                        
        print('limgs : ',len(self.l_imgs))

    def __getitem__(self,index):
        #Read all data, transform etc.
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
        
        l_imgs = []; l_VA = []; l_ldmrk = []; l_nc = []#,torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]
        
        if not self.isVideo : 
            x_l = [self.l_imgs[index]];labelE_l =[self.l_gtE[index].copy()];label_l = [self.l_gt[index].copy()];label_n =[self.list_gt_names[index]] 
        else : 
            x_l = self.l_imgs[index];labelE_l =self.l_gtE[index].copy();label_l = self.l_gt[index].copy();label_n =self.list_gt_names[index]
        
        for x,labelE,label,ln in zip(x_l,labelE_l,label_l,label_n) : 
            #print(x,labelE,label,ln)
            tImage = Image.open(x).convert("RGB")
            tImageB = None
            
            if self.onlyFace :    
                #crop the face region
                #t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 2,div_y = 2,images = cv2.imread(x),displacementxy = random.uniform(-.5,.5))
                if self.ldmrkNumber > 49 : 
                    t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label.copy(),div_x = 8,div_y = 8,images = cv2.imread(x))#,displacementxy = random.uniform(-.5,.5))
                else : 
                    t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  utils.get_enlarged_bb(the_kp = label.copy(),
                                                           div_x = 3,div_y = 3,images = cv2.imread(x), n_points = 49)#,displacementxy = random.uniform(-.5,.5))
                
                area = (x1,y1, x2,y2)
                tImage =  tImage.crop(area)
                
                label[:self.ldmrkNumber] -= x_min
                label[self.ldmrkNumber:] -= y_min
                
                tImage = tImage.resize((self.imageWidth,self.imageHeight))
                
                label[:self.ldmrkNumber] *= truediv(self.imageWidth,(x2 - x1))
                label[self.ldmrkNumber:] *= truediv(self.imageHeight,(y2 - y1))
                
                
                #now aliging 
                if self.align : 
                    tImageT = utils.PILtoOpenCV(tImage.copy())
                    if self.swap : 
                        ptsSource = torch.tensor([
                            [label[self.nose+self.ldmrkNumber],label[self.nose]],[label[self.leye+self.ldmrkNumber],label[self.leye]],[label[self.reye+self.ldmrkNumber],label[self.reye]]
                            ])
                        ptsSn = [
                            [label[self.nose+self.ldmrkNumber],label[self.nose]],[label[self.leye+self.ldmrkNumber],label[self.leye]],[label[self.reye+self.ldmrkNumber],label[self.reye]]
                            ]
                    else : 
                        ptsSource = torch.tensor([
                            [label[self.nose],label[self.nose+self.ldmrkNumber]],[label[self.leye],label[self.leye+self.ldmrkNumber]],[label[self.reye],label[self.reye+self.ldmrkNumber]]
                            ])
                        ptsSn =[
                            [label[self.nose],label[self.nose+self.ldmrkNumber]],[label[self.leye],label[self.leye+self.ldmrkNumber]],[label[self.reye],label[self.reye+self.ldmrkNumber]]
                            ]
                        
                        ptsSnFull = np.column_stack((label[:self.ldmrkNumber],label[self.ldmrkNumber:]))
                        ptsSnFull = np.asarray(ptsSnFull,np.float32)
                        
                    ptsSource = ptsSource.numpy()
                    ptsSource = np.asarray(ptsSource,np.float32)
                    
                    if self.useNudget : 
                        trans = nudged.estimate(ptsSn,self.ptsTn)
                        M = np.asarray(trans.get_matrix())[:2,:]
                        #print("Nudged : ",mN,trans.get_scale(),trans.get_rotation())
                    else :
                        #M = cv2.getAffineTransform(ptsSource,self.ptsDst)
                        #_,_,aff  = self.procrustes(ptsSource,self.ptsDst)
                         
                        #print(ptsSource.shape,'-', self.ptsDst.shape)
                        #print(ptsSnFull.shape,'-', self.ptsTnFull.shape)
                        
                        _,_,aff  = self.procrustes(self.ptsTnFull,ptsSnFull)
                        M = aff[:2,:]
                        
                    dst = cv2.warpAffine(tImageT,M,(self.imageWidth,self.imageHeight))
                    
                    #print(np.asarray(ptsSn).shape, np.asarray(self.ptsTn).shape,M.shape)
                    
                    
                    M_full = np.append(M,[[0,0,1]],axis = 0)
                    l_full = np.stack((label[:self.ldmrkNumber],label[self.ldmrkNumber:],np.ones(self.ldmrkNumber)))
                    
                    ldmark = np.matmul(M_full, l_full)
                    
                    if False : 
                        print(ldmark)
                        for i in range(self.ldmrkNumber) :
                            cv2.circle(dst,(int(scale(ldmark[0,i])),int(scale(ldmark[1,i]))),2,(0,255,0) )
                        
                        cv2.imshow('test align',dst)
                        cv2.waitKey(0)
                    
                    label = np.concatenate((ldmark[0],ldmark[1]))
                    tImage = utils.OpenCVtoPIL(dst)
                    
            newChannel = None
            
            if self.wHeatmap : 
                theMiddleName = 'img'
                filePath = x.split(os.sep)
                ifolder = filePath.index(theMiddleName)
                
                print(ifolder)
                image_name = filePath[-1]
                
                annot_name_H = os.path.splitext(image_name)[0]+'.npy'
                
                sDirName = filePath[:ifolder]
                dHeatmaps = '/'.join(sDirName)+'/heatmaps'
                
                finalTargetH = dHeatmaps+'/'+annot_name_H
                print(finalTargetH)
                
                if isfile(finalTargetH) and False: 
                    newChannel  = np.load(finalTargetH)
                    newChannel = Image.fromarray(newChannel)
                else : 
                    checkDirMake(dHeatmaps)
                    
                    tImageTemp = cv2.cvtColor(np.array(tImage),cv2.COLOR_RGB2BGR)
                    #tImageTemp = cv2.imread(x)#tImage.copy()
                    
                    print(len(label),label)
                    
                    b_channel,g_channel,r_channel = tImageTemp[:,:,0],tImageTemp[:,:,1],tImageTemp[:,:,2]
                    newChannel = b_channel.copy(); newChannel[:] = 0
                    
                    t0,t1,t2,t3 = utils.get_bb(label[0:self.ldmrkNumber], label[self.ldmrkNumber:],length=self.ldmrkNumber)
                    
                    l_cd,rv = utils.get_list_heatmap(0,None,t2-t0,t3-t1,.05)
                    height, width,_ = tImageTemp.shape
                    
                    wx = t2-t0
                    wy = t3-t1
                
                    scaler = 255/np.max(rv)
                    
                    for iter in range(self.ldmrkNumber) :
                        ix,iy = int(label[iter]),int(label[iter+self.ldmrkNumber])
                        
                        #Now drawing given the center
                        for iter2 in range(len(l_cd)) : 
                            value = int(rv[iter2]*scaler)
                            if newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] < value : 
                                newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] = int(rv[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)
                    
                    '''tImage2 = cv2.merge((b_channel, newChannel,newChannel, newChannel))
                    cv2.imshow("combined",tImage2)
                    cv2.waitKey(0)'''
                    
                    np.save(finalTargetH,newChannel)
                    newChannel = Image.fromarray(newChannel)
            
            if self.augment : 
                sel = np.random.randint(0,4)
                #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
                if sel == 0 : 
                    pass
                elif sel == 1 : 
                    flip = RandomHorizontalFlip_WL(1,self.ldmrkNumber)
                    tImage,label,newChannel = flip(tImage,label,newChannel,self.ldmrkNumber)
                elif sel == 2 and not self.align : 
                    rot = RandomRotation_WL(45)
                    tImage,label,newChannel = rot(tImage,label,newChannel,self.ldmrkNumber)
                elif sel == 3 : 
                    occ = Occlusion_WL(1)
                    tImage,label,newChannel = occ(tImage,label,newChannel)
                    
                #random crop
                if not self.align : 
                    rc = RandomResizedCrop_WL(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
                    tImage,label,newChannel= rc(tImage,label,newChannel)
                
                #additional blurring
                if (np.random.randint(1,3)%2==0) and True : 
                    sel_n = np.random.randint(1,6)
                    #sel_n = 4
                    rc = GeneralNoise_WL(1)
                    tImage,label= rc(tImage,label,sel_n,np.random.randint(0,3))
            
            if self.returnM : 
                if self.swap : 
                    ptsSource = torch.tensor([
                        [label[self.nose+self.ldmrkNumber],label[self.nose]],[label[self.leye+self.ldmrkNumber],label[self.leye]],[label[self.reye+self.ldmrkNumber],label[self.reye]]
                        ])
                    ptsSn = [
                        [label[self.nose+self.ldmrkNumber],label[self.nose]],[label[self.leye+self.ldmrkNumber],label[self.leye]],[label[self.reye+self.ldmrkNumber],label[self.reye]]
                        ]
                else : 
                    ptsSource = torch.tensor([
                        [label[self.nose],label[self.nose+self.ldmrkNumber]],[label[self.leye],label[self.leye+self.ldmrkNumber]],[label[self.reye],label[self.reye+self.ldmrkNumber]]
                        ])
                    ptsSn =[
                        [label[self.nose],label[self.nose+self.ldmrkNumber]],[label[self.leye],label[self.leye+self.ldmrkNumber]],[label[self.reye],label[self.reye+self.ldmrkNumber]]
                        ]
                    
                    ptsSnFull = np.column_stack((label[:self.ldmrkNumber],label[self.ldmrkNumber:]))
                    ptsSnFull = np.asarray(ptsSnFull,np.float32)
                    
                ptsSource = ptsSource.numpy()
                ptsSource = np.asarray(ptsSource,np.float32)
                
                if self.useNudget : 
                    trans = nudged.estimate(ptsSn,self.ptsTn)
                    M = np.asarray(trans.get_matrix())[:2,:]
                else :
                    #M = cv2.getAffineTransform(ptsSource,self.ptsDst)
                    _,_,aff  = self.procrustes(self.ptsTnFull,ptsSnFull)
                    M = aff[:2,:]    
                
                if False :
                    tImageT = utils.PILtoOpenCV(tImage.copy())
                    dst = cv2.warpAffine(tImageT,M,(self.imageWidth,self.imageHeight))
                    
                    print(np.asarray(ptsSn).shape, np.asarray(self.ptsTn).shape,M.shape)
                    
                    M_full = np.append(M,[[0,0,1]],axis = 0)
                    l_full = np.stack((label[:self.ldmrkNumber],label[self.ldmrkNumber:],np.ones(self.ldmrkNumber)))
                    
                    ldmark = np.matmul(M_full, l_full)
                    print(ldmark)
                    for i in range(self.ldmrkNumber) :
                        cv2.circle(dst,(int(scale(ldmark[0,i])),int(scale(ldmark[1,i]))),2,(0,0,255) )
                    
                    cv2.imshow('test recovered',dst)
                    cv2.waitKey(0)
                    
                    
                
                Minter = self.param2theta(np.append(M,[[0,0,1]],axis = 0), self.imageWidth,self.imageHeight)
                Mt = torch.from_numpy(Minter).float()
            else : 
                Mt = torch.zeros(1)
            
            
            if self.useIT : 
                tImage = self.transformInternal(tImage)
            else : 
                tImage = self.transform(tImage)
            
            
            
            
            if not self.wHeatmap : 
                l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label))#,x,self.list_gt_names[index]
            else : 
                newChannel = transforms.Resize(224)(newChannel)
                newChannel = transforms.ToTensor()(newChannel)
                newChannel = newChannel.sub(125)
                l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label)); l_nc.append(newChannel)
                #return tImage,torch.FloatTensor(labelE),torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]
                
                
                
        if not self.isVideo : 
            if self.wHeatmap : 
                return l_imgs[0], l_VA[0], l_ldmrk[0], l_nc[0], Mt
            else : 
                return l_imgs[0], l_VA[0], l_ldmrk[0], Mt
        else : 
            #lImgs = torch.Tensor(len(l_imgs),3,self.imageHeight,self.imageWidth)
            #lVA = torch.Tensor(len(l_VA),2)
            #lLD = torch.Tensor(len(l_ldmrk),136)
            lImgs = torch.stack(l_imgs)
            lVA = torch.stack(l_VA)
            lLD = torch.stack(l_ldmrk)
            
            #print(lImgs.shape, l_imgs[0].shape, l_VA[0].shape,len(lImgs))
            
            #torch.cat(l_imgs, out=lImgs)
            #torch.cat(l_VA, out=lVA)
            #torch.cat(l_ldmrk, out=lLD)
            
            if self.wHeatmap :
                #lnc = torch.Tensor(len(l_nc),1,self.imageHeight,self.imageWidth)
                #torch.cat(l_nc, out=lnc)
                lnc = torch.stack(l_nc)
                return lImgs, lVA, lLD, lnc, Mt
            else : 
                return lImgs, lVA, lLD, Mt
                
    
    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    
    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    
    
    def param2theta(self,param, w, h):
        param = np.linalg.inv(param)
        theta = np.zeros([2,3])
        theta[0,0] = param[0,0]
        theta[0,1] = param[0,1]*h/w
        theta[0,2] = param[0,2]*2/w + theta[0,0] + theta[0,1] - 1
        theta[1,0] = param[1,0]*w/h
        theta[1,1] = param[1,1]
        theta[1,2] = param[1,2]*2/h + theta[1,0] + theta[1,1] - 1
        return theta
    
    def procrustes(self, X, Y, scaling=True, reflection='best'):
        
        n,m = X.shape
        ny,my = Y.shape
    
        muX = X.mean(0)
        muY = Y.mean(0)
        
        X0 = X - muX
        Y0 = Y - muY
        
        ssX = (X0**2.).sum()
        ssY = (Y0**2.).sum()
        
        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)
        
        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY
        
        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)
        
        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U,s,Vt = np.linalg.svd(A,full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)
        
        if reflection is not 'best':
            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0
            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:,-1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)
        
        traceTA = s.sum()
        
        if scaling:
            # optimum scaling of Y
            b = traceTA * normX / normY
            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA**2
            # transformed coords
            Z = normX*traceTA*np.dot(Y0, T) + muX
        
        else:
            b = 1
            d = 1 + ssY/ssX - 2 * traceTA * normY / normX
            Z = normY*np.dot(Y0, T) + muX
        
        # transformation matrix
        if my < m:
            T = T[:my,:]
        
        c = muX - b*np.dot(muY, T)
        
        #transformation values 
        #tform = {'rotation':T, 'scale':b, 'translation':c}
        tform = np.append(b*T,[c],axis = 0).T
        tform = np.append(tform,[[0,0,1]],axis = 0)
        
        return d, Z, tform
    
    def __len__(self):
        return len(self.l_imgs)





class SEWAFEWReduced(data.Dataset): #return affect on Valence[0], Arousal[1] order
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["AFEW"],dir_gt = None,onlyFace = True, image_size =224, 
                 transform = None,useIT = False,augment = False, step = 1,split = False, 
                 nSplit = 5, listSplit = [0,1,2,3,4],isVideo = False, seqLength = None, dbType = 0,
                 returnQuadrant = False, returnNoisy = False, returnWeight = False):#dbtype 0 is AFEW, 1 is SEWA
        
        self.dbType = dbType
        
        self.seq_length = seqLength 
        self.isVideo = isVideo
        
        self.transform = transform
        self.onlyFace = onlyFace
        self.augment = augment 
        
        self.imageSize = image_size
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.useIT = useIT
        self.curDir = rootDir +"/"#/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/"
        
        self.returnNoisy = returnNoisy
        self.returnWeight = returnWeight
        
        if self.returnWeight :
            name = 'VA-Train-'+str(listSplit[0])+'.npy' 
            if self.dbType == 1 : 
                name='S-'+name
            weight = np.load(rootDir+"/DST-SE-AF/"+name).astype('float')+1
            sum = weight.sum(0)
            
            weight = (weight/sum)
            #print('1',weight)
            
            weight = 1/weight
            #print('2',weight)
            
            sum = weight.sum(0)
            weight = weight/sum
            #print('3',weight)
            "just tesing for the latencyh if its possible. "
            self.weight =  weight
        
        self.returnQ = returnQuadrant
        
        if self.augment : 
            self.flip = RandomHorizontalFlip(1)
            self.rot = RandomRotation(45)
            self.occ = Occlusion(1)
            self.rc = RandomResizedCrop(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
            
        if self.returnNoisy : 
            self.gn = GeneralNoise(1)
            self.occ = Occlusion(1)
            
        list_gt = []
        list_labels_tE = []
        
        counter_image = 0
        
        annotE_name = 'annot2'
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        list_missing = []
        
        
        for data in data_list : 
            print(("Opening "+data))
            
            fullDir = self.curDir +data+"/"
            listFolder = os.listdir(fullDir)
            listFolder.sort()
            
            for tempx in range(len(listFolder)):
                f = listFolder[tempx]
                fullPath = os.path.join(fullDir,f)
                #print('opening fullpath',fullPath)
                if os.path.isdir(fullPath): # Check if object is directory
                    #print((f.name, f.full_path)) # Name is without extension
                    #c_image,c_ldmark = 0,0
                    
                    if self.dbType == 1 : #we directly get the VA file in case of sewa  
                        #first get the valence 
                        valFile = fullPath+"/valence/"+f+"_Valence_A_Aligned.csv"
                        aroFile = fullPath+"/arousal/"+f+"_Arousal_A_Aligned.csv"
                        
                        list_labels_tE.append([valFile,aroFile])
                        #print(valFile,aroFile)
                        
                    for sub_f in file_walker.walk(fullPath):
                        if sub_f.isDirectory: # Check if object is directory
                            list_dta = []
                            #print(sub_f.name)
                            if(sub_f.name == 'img-128'): #Else it is the image
                                
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_gt.append(sorted(list_dta))
                                counter_image+=len(list_dta)
                                c_ldmrk = len(list_dta)
        
                            elif (sub_f.name == annotE_name) :
                                if self.dbType == 0 : 
                                    #If that's annot, add to labels_t
                                    for sub_sub_f in sub_f.walk(): #this is the data
                                        if(".npy" not in sub_sub_f.full_path):
                                            list_dta.append(sub_sub_f.full_path)
                                    list_labels_tE.append(sorted(list_dta))
                                    
                                    
        self.length = counter_image
        print("Now opening keylabels")
         
        list_labelsEN = []
        list_labelsE = []
        
        for ix in range(len(list_labels_tE)) : #lbl,lble in (list_labels_t,list_labels_tE) :
            
            lbl_2 = [] #Per folder
            lbl_n2 = [] #Per folder
            
            if self.dbType == 1 : #sewa  
                #print(list_labels_t[ix][0])
                valFile = np.asarray(readCSV(list_labels_tE[ix][0]))
                aroFile = np.asarray(readCSV(list_labels_tE[ix][1]))
                
                lbl_n2.append(list_labels_tE[ix][0])
                lbl_2 = np.column_stack((valFile,aroFile))
            else : 
                for jx in range(len (list_labels_tE[ix])): #lbl_sub in lbl :
                    
                    #print(os.path.basename(list_gt[ix][jx]))
                    #print(os.path.basename(list_labels_t[ix][jx]))
                    #print(os.path.basename(list_labels_tE[ix][jx]))
                    
                    if self.dbType == 0 : 
                        lbl_subE = list_labels_tE[ix][jx]
                        if ('aro' in lbl_subE) : 
                            x = []
                            #print(lbl_sub)
                            with open(lbl_subE) as file:
                                data2 = [re.split(r'\t+',l.strip()) for l in file]
                            for i in range(len(data2)) :
                                temp = [ float(j) for j in data2[i][0].split()]
                                temp.reverse() #to give the valence first. then arousal
                                x.append(temp)
                            
                            lbl_2.append(np.array(x).flatten('F'))
                            lbl_n2.append(lbl_subE)
                
            
            list_labelsEN.append(lbl_n2)
            list_labelsE.append(lbl_2)
        
            
        '''for lbl in list_labels_t :
            lbl_68 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('pts' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
            list_labels.append(lbl_68)
        
        list_labelsE = []
        for lbl in list_labels_tE :
            lbl_2 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('aro' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_2.append(np.array(x).flatten('F')) #1 record
            list_labelsE.append(lbl_2)'''
        
        
        
        '''print(len(list_labelsN[0])) 
        print(len(list_labelsEN[0])) 
        print(len(list_labels[0])) 
        print(len(list_labelsE[0]))'''
                
        
        t_l_imgs = []
        t_l_gtE = []
        
        t_list_gtE_names = []
        
        #print(list_labelsEN)
        
        if not self.isVideo :
            #Flatten it to one list
            for i in range(0,len(list_gt)): #For each dataset
                
                list_images = []
                list_gtE_names = []
                indexer = 0
                
                list_ground_truthE = np.zeros([len(list_gt[i]),2])
                
                for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                    list_images.append(list_gt[i][j])
                    #print(list_labelsEN)
                    if self.dbType == 0 : 
                        list_gtE_names.append(list_labelsEN[i][j])
                    else : 
                        list_gtE_names.append(list_labelsEN[i][0])
                    #print(list_labelsEN[i])
                    
                    '''if len(list_labels[i][j] < 1): 
                        print(list_labels[i][j])'''
                    #print(len(list_labels[i][j]))
                    list_ground_truthE[indexer] = np.array(list_labelsE[i][j]).flatten('F')
                    indexer += 1
                
                t_l_imgs.append(list_images)
                t_l_gtE.append(list_ground_truthE)
                t_list_gtE_names.append(list_gtE_names)
        else : 
            if self.seq_length is None :
                list_ground_truth = np.zeros([int(counter_image/(self.seq_length*step)),self.seq_length,136])
                indexer = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                    counter = 0
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize
                        
                        temp = []
                        temp2 = np.zeros([self.seq_length,136])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z])
                            temp2[i_temp] = list_labels[i][z]
                            i_temp+=1
                            
                        list_images.append(temp)
                        list_ground_truth[indexer] = temp2
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                self.l_imgs = list_images
                self.l_gt = list_ground_truth
            else : 
                counter_seq = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                
                    indexer = 0;
                    list_gtE_names = []
                    list_ground_truthE = np.zeros([int(len(list_gt[i])/(self.seq_length*step)),self.seq_length,2])#np.zeros([counter_image,2])
                    
                    counter = 0
                    list_images = []
                    
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize   
                        
                        
                        
                        temp = []
                        tmpn2 = []
                        temp3 = np.zeros([self.seq_length,2])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z]) 
                            temp3[i_temp] = list_labelsE[i][z].flatten('F')
                            
                            
                            if self.dbType == 0 : 
                                #list_gtE_names.append(list_labelsEN[i][j])
                                tmpn2.append(list_labelsEN[i][j])
                            else : 
                                #list_gtE_names.append(list_labelsEN[i][0])
                                tmpn2.append(list_labelsEN[i][0])
                            
                            i_temp+=1
                            counter_seq+=1
                            
                        list_images.append(temp)
                        list_ground_truthE[indexer] = temp3
                        list_gtE_names.append(tmpn2)
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                        
                    t_l_imgs.append(list_images)
                    t_l_gtE.append(list_ground_truthE)
                    
                    t_list_gtE_names.append(list_gtE_names)
                
        '''print('length : ',len(t_l_imgs)) #Folder
        print('lengt2 : ',len(t_l_imgs[0])) #all/seq
        print('lengt3 : ',len(t_l_imgs[0][0])) #seq
        print('length4 : ',len(t_l_imgs[0][0][0]))'''
        
        #[folder, all/seq,seq]
        
        self.l_imgs = []
        self.l_gtE = []
        self.list_gtE_names = []
        
        #print('cimage : ',counter_image)
        
        
        if split :
            '''print('splitting')
            self.l_imgs = []
            self.l_gt = []
            self.l_gtE = []
            
            totalData = len(list_images)
            perSplit = int(truediv(totalData, nSplit))
            for x in listSplit :
                print('split : ',x) 
                begin = x*perSplit
                if x == nSplit-1 : 
                    end = begin + (totalData - begin)
                else : 
                    end = begin+perSplit
                print(begin,end,totalData)
                for x2 in range(begin,end) : 
                    #print(x2,totalData)
                    self.l_imgs.append(list_images[x2])
                    self.l_gt.append(list_ground_truth[x2])
                    self.l_gtE.append(list_ground_truthE[x2])'''
            indexer = 0 
            
            self.l_gtE = []
            '''else : 
                self.l_gt = np.zeros([counter_seq,self.seq_length,136])
                self.l_gtE = np.zeros([counter_seq,self.seq_length,2])'''
                
            totalData = len(t_l_imgs)
            perSplit = int(truediv(totalData, nSplit))
            for x in listSplit :
                print('split : ',x) 
                begin = x*perSplit
                if x == nSplit-1 : 
                    end = begin + (totalData - begin)
                else : 
                    end = begin+perSplit
                print(begin,end,totalData)
                
                if not self.isVideo :
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): 
                            #print('append ',t_l_imgs[i][j])
                            self.l_imgs.append(t_l_imgs[i][j])
                            self.l_gtE.append(t_l_gtE[i][j])
                            
                            self.list_gtE_names.append(t_list_gtE_names[i][j])
                            indexer+=1
                            
                else : 
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): #seq counter
                        
                            t_img = []
                            t_gtE = []
                            t_gt_EN = []
                            tmp = 0
                            
                            for k in range(len(t_l_imgs[i][j])): #seq size
                                t_img.append(t_l_imgs[i][j][k])
                                t_gtE.append(t_l_gtE[i][j][k])
                                t_gt_EN.append(t_list_gtE_names[i][j][k])
                                tmp+=1
                                
                            #print('append ',t_img)
                            self.l_imgs.append(t_img)
                            self.l_gtE.append(t_gtE)
                            self.list_gtE_names.append(t_gt_EN)
                            indexer+=1
                    
                print(len(self.l_imgs))
                    
            self.l_gtE = np.asarray(self.l_gtE)
        else :
            if not self.isVideo :
                self.l_gtE = np.zeros([counter_image,2])
                indexer = 0
                
                
                for i in range(len(t_l_imgs)): 
                    for j in range(len(t_l_imgs[i])): 
                        self.l_imgs.append(t_l_imgs[i][j])
                        print(i,j,'-',len(t_l_imgs[i]))
                        self.l_gtE[indexer] = t_l_gtE[i][j]
                        
                        self.list_gtE_names.append(t_list_gtE_names[i][j])
                        indexer+=1
                    
            else : 
                self.l_gtE = np.zeros([counter_seq,self.seq_length,2])
                
                indexer = 0
                
                for i in range(len(t_l_imgs)): #dataset
                    for j in range(len(t_l_imgs[i])): #seq counter
                        
                        t_img = []
                        
                        t_gte = np.zeros([self.seq_length,2])
                        
                        t_gt_n = []
                        t_gt_en = []
                        i_t = 0
                        
                        for k in range(len(t_l_imgs[i][j])): #seq size
                            
                            t_img.append(t_l_imgs[i][j][k])
                            t_gte[i_t] = t_l_gtE[i][j][k]
                            
                            t_gt_en.append(t_list_gtE_names[i][j][k])
                            
                            i_t+=1
                            
                        self.l_imgs.append(t_img)
                        self.l_gtE[indexer] = t_gte
                        self.list_gtE_names.append(t_gt_en)
                        
                        indexer+=1
                        
        print('limgs : ',len(self.l_imgs))

    def __getitem__(self,index):
        #Read all data, transform etc.
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
        
        l_imgs = []; l_ldmrk = []; l_VA = []; l_nc = []; l_qdrnt = []#,torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]
        if self.returnNoisy : 
            l_nimgs = []
        
        if self.returnWeight : 
            l_weights = []
        
        if not self.isVideo : 
            x_l = [self.l_imgs[index]];labelE_l =[self.l_gtE[index].copy()];label_n =[self.list_gtE_names[index]] 
        else : 
            x_l = self.l_imgs[index];labelE_l =self.l_gtE[index].copy();label_n =self.list_gtE_names[index]
        
        
        #print('label n ',label_n)
        for x,labelE,ln in zip(x_l,labelE_l,label_n) : 
            #print(x,labelE,label,ln)
            tImage = Image.open(x).convert("RGB")
            tImageB = None
            
            newChannel = None
            
            if self.augment : 
                '''sel = np.random.randint(0,4)
                #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
                if sel == 0 : 
                    pass
                elif sel == 1 : 
                    flip = RandomHorizontalFlip_WL(1,self.ldmrkNumber)
                    tImage,label,newChannel = flip(tImage,label,newChannel,self.ldmrkNumber)
                elif sel == 2 and not self.align : 
                    rot = RandomRotation_WL(45)
                    tImage,label,newChannel = rot(tImage,label,newChannel,self.ldmrkNumber)
                elif sel == 3 : 
                    occ = Occlusion_WL(1)
                    tImage,label,newChannel = occ(tImage,label,newChannel)
                    
                #random crop
                if not self.align : 
                    rc = RandomResizedCrop_WL(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
                    tImage,label,newChannel= rc(tImage,label,newChannel)
                
                #additional blurring
                if (np.random.randint(1,3)%2==0) and True : 
                    sel_n = np.random.randint(1,6)
                    #sel_n = 4
                    rc = GeneralNoise_WL(1)
                    tImage,label= rc(tImage,label,sel_n,np.random.randint(0,3))'''
                
                if self.returnNoisy :
                    sel = np.random.randint(0,3) #Skip occlusion as noise
                else : 
                    sel = np.random.randint(0,4)
                #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
                if sel == 0 : 
                    pass
                elif sel == 1 : 
                    #flip = RandomHorizontalFlip_WL(1)
                    #tImage,label,newChannel = flip(tImage,label,newChannel)
                    #flip = RandomHorizontalFlip(1)
                    tImage = self.flip(tImage)
                elif sel == 2 : 
                    #rot = RandomRotation_WL(45)
                    #tImage,label,newChannel = rot(tImage,label,newChannel)
                    #rot = RandomRotation(45)
                    tImage = self.rot(tImage)
                elif sel == 3 : 
                    #occ = Occlusion_WL(1)
                    #tImage,label,newChannel = occ(tImage,label,newChannel)
                    #occ = Occlusion(1)
                    tImage = self.occ(tImage)
                    
                #random crop
                if (np.random.randint(1,3)%2==0) : 
                    #rc = RandomResizedCrop_WL(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
                    #tImage,label,newChannel= rc(tImage,label,newChannel)
                    
                    #rc = RandomResizedCrop(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
                    tImage= self.rc(tImage)
                
            if self.returnNoisy :
                nImage = tImage.copy()
            
                #additional blurring
                if (np.random.randint(1,3)%2==0): 
                    #sel_n = np.random.randint(1,6)
                    sel_n = np.random.randint(1,7)
                    
                    #sel_n = 4
                    #gn = GeneralNoise_WL(1)
                    #tImage,label= gn(tImage,label,sel_n,np.random.randint(0,3))
                    
                    if sel_n > 5 : 
                        #occ = Occlusion(1)
                        nImage = self.occ(nImage)
                    else :
                        #rc = GeneralNoise(1)
                        #tImage = rc(tImage,sel_n,np.random.randint(0,3))
                        nImage = self.gn(nImage,sel_n,np.random.randint(0,3))
                    
            label = torch.zeros(1)
            Mt = torch.zeros(1)
            
            
            if self.useIT : 
                tImage = self.transformInternal(tImage)
                if self.returnNoisy : 
                    nImage = self.transformInternal(nImage)
            else : 
                tImage = self.transform(tImage)
                if self.returnNoisy : 
                    nImage = self.transform(nImage)
            
            
            
            l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label))#,x,self.list_gt_names[index]
            if self.returnNoisy : 
                l_nimgs.append(nImage)
            
            if self.returnQ : 
                if self.dbType == 1 :
                    min = 0; max = 1;
                else : 
                    min = -10; max = 10;
                    
                l_qdrnt.append(toQuadrant(labelE, min, max, toOneHot=False))
                
            if self.returnWeight :
                v = labelE[0] 
                a = labelE[0]
                
                if self.dbType :#sewa 
                    v = v*10+1
                    a = a*10+1
                else :
                    v = v+10
                    a = a+10
                
                v,a = int(v),int(a)
                '''print('the v :{} a : {} db : {}'.format(v,a,self.dbType))
                print(self.weight)
                print(self.weight.shape)'''
                l_weights.append([self.weight[v,0],self.weight[a,1]])
                
            l_nc.append(ln)
                
        if not self.isVideo : 
            if self.returnQ : 
                if self.returnNoisy :
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,ln[0],l_qdrnt[0],l_nimgs[0]]
                else : 
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,ln[0],l_qdrnt[0]]
            else :
                if self.returnNoisy : 
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,ln[0],l_nimgs[0]]
                else : 
                    res =  [l_imgs[0], l_VA[0], l_ldmrk[0], Mt,ln[0]]
                    
            if self.returnWeight :
                res.append(torch.tensor(l_weights[0]))
                
            return res 
        else : 
            #lImgs = torch.Tensor(len(l_imgs),3,self.imageHeight,self.imageWidth)
            #lVA = torch.Tensor(len(l_VA),2)
            #lLD = torch.Tensor(len(l_ldmrk),136)
            lImgs = torch.stack(l_imgs)
            lVA = torch.stack(l_VA)
            lLD = torch.stack(l_ldmrk)
            
            #print(l_VA)
            l_qdrnt = torch.tensor((l_qdrnt))
            
            #print(lImgs.shape, l_imgs[0].shape, l_VA[0].shape,len(lImgs))
            
            #torch.cat(l_imgs, out=lImgs)
            #torch.cat(l_VA, out=lVA)
            #torch.cat(l_ldmrk, out=lLD)
            if self.returnQ : 
                if self.returnNoisy : 
                    res = [lImgs, lVA, lLD, Mt,l_nc,l_qdrnt,l_nimgs]
                else : 
                    res = [lImgs, lVA, lLD, Mt,l_nc,l_qdrnt]
            else : 
                if self.returnNoisy : 
                    res = [lImgs, lVA, lLD, Mt,l_nc,l_nimgs]
                else : 
                    res = [lImgs, lVA, lLD, Mt,l_nc]
                    
            if self.returnWeight : 
                l_weights = torch.tensor(l_weights)
                res.append(l_weights)
                
            return res 
    
    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    
    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    
    
    def param2theta(self,param, w, h):
        param = np.linalg.inv(param)
        theta = np.zeros([2,3])
        theta[0,0] = param[0,0]
        theta[0,1] = param[0,1]*h/w
        theta[0,2] = param[0,2]*2/w + theta[0,0] + theta[0,1] - 1
        theta[1,0] = param[1,0]*w/h
        theta[1,1] = param[1,1]
        theta[1,2] = param[1,2]*2/h + theta[1,0] + theta[1,1] - 1
        return theta
    
    def procrustes(self, X, Y, scaling=True, reflection='best'):
        
        n,m = X.shape
        ny,my = Y.shape
    
        muX = X.mean(0)
        muY = Y.mean(0)
        
        X0 = X - muX
        Y0 = Y - muY
        
        ssX = (X0**2.).sum()
        ssY = (Y0**2.).sum()
        
        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)
        
        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY
        
        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)
        
        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U,s,Vt = np.linalg.svd(A,full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)
        
        if reflection is not 'best':
            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0
            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:,-1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)
        
        traceTA = s.sum()
        
        if scaling:
            # optimum scaling of Y
            b = traceTA * normX / normY
            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA**2
            # transformed coords
            Z = normX*traceTA*np.dot(Y0, T) + muX
        
        else:
            b = 1
            d = 1 + ssY/ssX - 2 * traceTA * normY / normX
            Z = normY*np.dot(Y0, T) + muX
        
        # transformation matrix
        if my < m:
            T = T[:my,:]
        
        c = muX - b*np.dot(muY, T)
        
        #transformation values 
        #tform = {'rotation':T, 'scale':b, 'translation':c}
        tform = np.append(b*T,[c],axis = 0).T
        tform = np.append(tform,[[0,0,1]],axis = 0)
        
        return d, Z, tform
    
    def __len__(self):
        return len(self.l_imgs)





class AFEWVA(data.Dataset):
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["AFEW"],dir_gt = None,onlyFace = True, image_size =224, 
                 transform = None,useIT = False,augment = False, step = 1,split = False, 
                 nSplit = 5, listSplit = [0,1,2,3,4],wHeatmap= False,isVideo = False, seqLength = None,
                 returnM = False, toAlign = False, skipCropping = False):
        
        self.seq_length = seqLength 
        self.isVideo = isVideo
        self.align = toAlign
        self.useNudget = False
        self.returnM = returnM
        self.skipCropping = skipCropping
        
        self.transform = transform
        self.onlyFace = onlyFace
        self.augment = augment 
        self.wHeatmap = wHeatmap
        
        self.imageSize = image_size
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.useIT = useIT
        self.curDir = rootDir +"/"#/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/"
        
        self.mean_shape = np.load(curDir+'mean_shape-pad-'+str(image_size)+'.npy')
        
        self.swap = False 
        
        if self.swap : 
            self.ptsDst = np.asarray([
                [self.mean_shape[33+68],self.mean_shape[33]],[self.mean_shape[41+68],self.mean_shape[41]],[self.mean_shape[46+68],self.mean_shape[46]]
                ],dtype= np.float32)
            
            self.ptsTn = [self.mean_shape[33+68],self.mean_shape[33]],[self.mean_shape[41+68],self.mean_shape[41]],[self.mean_shape[46+68],self.mean_shape[46]]
        else : 
            self.ptsDst = np.asarray([
                [self.mean_shape[33],self.mean_shape[33+68]],[self.mean_shape[41],self.mean_shape[41+68]],[self.mean_shape[46],self.mean_shape[46+68]]
                ],dtype= np.float32)
            self.ptsTn = [self.mean_shape[33],self.mean_shape[33+68]],[self.mean_shape[41],self.mean_shape[41+68]],[self.mean_shape[46],self.mean_shape[46+68]]
            
            self.ptsTnFull = np.column_stack((self.mean_shape[:68],self.mean_shape[68:]))
        
        list_gt = []
        list_labels_t = []
        list_labels_tE = []
        
        counter_image = 0
        annotL_name = 'annot'
        annotE_name = 'annot2'
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        for data in data_list : 
            print(("Opening "+data))
            for f in file_walker.walk(self.curDir +data+"/"):
                if f.isDirectory: # Check if object is directory
                    #print((f.name, f.full_path)) # Name is without extension
                    for sub_f in f.walk():
                        if sub_f.isDirectory: # Check if object is directory
                            list_dta = []
                            
                            #print(sub_f.name)
                            if(sub_f.name == annotL_name) : #If that's annot, add to labels_t
                                
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_labels_t.append(sorted(list_dta))
                                
                            elif(sub_f.name == annotE_name) : #If that's annot, add to labels_t
                                
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_labels_tE.append(sorted(list_dta))
                                
                            elif(sub_f.name == 'img'): #Else it is the image
                                
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_gt.append(sorted(list_dta))
                                counter_image+=len(list_dta)
        
        self.length = counter_image
        print("Now opening keylabels")
        
        
        
        list_labelsN = [] 
        list_labelsEN = []
        
        list_labels = [] 
        list_labelsE = []
        
        for ix in range(len(list_labels_t)) : #lbl,lble in (list_labels_t,list_labels_tE) :
            lbl_68 = [] #Per folder
            lbl_2 = [] #Per folder
            
            lbl_n68 = [] #Per folder
            lbl_n2 = [] #Per folder
            for jx in range(len (list_labels_t[ix])): #lbl_sub in lbl :
                
                #print(os.path.basename(list_gt[ix][jx]))
                #print(os.path.basename(list_labels_t[ix][jx]))
                #print(os.path.basename(list_labels_tE[ix][jx]))
                
                lbl_sub = list_labels_t[ix][jx]
                if ('pts' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
                    lbl_n68.append(lbl_sub)
                    
                lbl_subE = list_labels_tE[ix][jx]
                if ('aro' in lbl_subE) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_subE) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_2.append(np.array(x).flatten('F'))
                    lbl_n2.append(lbl_sub)
                
                
            list_labelsN.append(lbl_n68)
            list_labelsEN.append(lbl_n2)
                
            list_labels.append(lbl_68)
            list_labelsE.append(lbl_2)
        
            
        '''for lbl in list_labels_t :
            lbl_68 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('pts' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
            list_labels.append(lbl_68)
        
        list_labelsE = []
        for lbl in list_labels_tE :
            lbl_2 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('aro' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_2.append(np.array(x).flatten('F')) #1 record
            list_labelsE.append(lbl_2)'''
        
        
        
        '''print(len(list_labelsN[0])) 
        print(len(list_labelsEN[0])) 
        print(len(list_labels[0])) 
        print(len(list_labelsE[0]))'''
                
        
        t_l_imgs = []
        t_l_gt = []
        t_l_gtE = []
        
        t_list_gt_names = []
        t_list_gtE_names = []
        
        
        
        if not self.isVideo :
            #Flatten it to one list
            for i in range(0,len(list_gt)): #For each dataset
                
                list_images = []
                list_gt_names = []
                list_gtE_names = []
                indexer = 0
                
                list_ground_truth = np.zeros([len(list_gt[i]),136])
                list_ground_truthE = np.zeros([len(list_gt[i]),2])
                
                for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                    list_images.append(list_gt[i][j])
                    
                    list_gt_names.append(list_labelsN[i][j])
                    list_gtE_names.append(list_labelsEN[i][j])
                    
                    list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                    list_ground_truthE[indexer] = np.array(list_labelsE[i][j]).flatten('F')
                    indexer += 1
                
                t_l_imgs.append(list_images)
                t_l_gt.append(list_ground_truth)
                t_l_gtE.append(list_ground_truthE)
                
                t_list_gt_names.append(list_gt_names)
                t_list_gtE_names.append(list_gtE_names)
        
        else : 
            if self.seq_length is None :
                list_ground_truth = np.zeros([int(counter_image/(self.seq_length*step)),self.seq_length,136])
                indexer = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                    counter = 0
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize
                        
                        temp = []
                        temp2 = np.zeros([self.seq_length,136])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z])
                            temp2[i_temp] = list_labels[i][z]
                            i_temp+=1
                            
                        list_images.append(temp)
                        list_ground_truth[indexer] = temp2
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                self.l_imgs = list_images
                self.l_gt = list_ground_truth
            else : 
                counter_seq = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                
                    indexer = 0;    
                    list_gt_names = []
                    list_gtE_names = []
                    
                    list_ground_truth = np.zeros([int(len(list_gt[i])/(self.seq_length*step)),self.seq_length,136]) #np.zeros([counter_image,136])
                    list_ground_truthE = np.zeros([int(len(list_gt[i])/(self.seq_length*step)),self.seq_length,2])#np.zeros([counter_image,2])
                    
                    counter = 0
                    list_images = []
                    
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize   
                        tmpn68 = []
                        tmpn2 = []
                        
                        temp = []
                        temp2 = np.zeros([self.seq_length,136])
                        temp3 = np.zeros([self.seq_length,2])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z])
                            temp2[i_temp] = list_labels[i][z].flatten('F')
                            temp3[i_temp] = list_labelsE[i][z].flatten('F')
                             
                            tmpn68.append(list_labelsN[i][z])
                            tmpn2.append(list_labelsEN[i][z])
                            
                            i_temp+=1
                            counter_seq+=1
                            
                        list_images.append(temp)
                        list_ground_truth[indexer] = temp2
                        list_ground_truthE[indexer] = temp3
                        
                        list_gt_names.append(tmpn68)
                        list_gtE_names.append(tmpn2)
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                        
                    t_l_imgs.append(list_images)
                    t_l_gt.append(list_ground_truth)
                    t_l_gtE.append(list_ground_truthE)
                    
                    t_list_gt_names.append(list_gt_names)
                    t_list_gtE_names.append(list_gtE_names)
                
        '''print('length : ',len(t_l_imgs)) #Folder
        print('lengt2 : ',len(t_l_imgs[0])) #all/seq
        print('lengt3 : ',len(t_l_imgs[0][0])) #seq
        print('length4 : ',len(t_l_imgs[0][0][0]))'''
        
        #[folder, all/seq,seq]
        
        self.l_imgs = []
        self.l_gt = []
        self.l_gtE = []
        
        self.list_gt_names = []
        self.list_gtE_names = []
        
        #print('cimage : ',counter_image)
        
        
        if split :
            '''print('splitting')
            self.l_imgs = []
            self.l_gt = []
            self.l_gtE = []
            
            totalData = len(list_images)
            perSplit = int(truediv(totalData, nSplit))
            for x in listSplit :
                print('split : ',x) 
                begin = x*perSplit
                if x == nSplit-1 : 
                    end = begin + (totalData - begin)
                else : 
                    end = begin+perSplit
                print(begin,end,totalData)
                for x2 in range(begin,end) : 
                    #print(x2,totalData)
                    self.l_imgs.append(list_images[x2])
                    self.l_gt.append(list_ground_truth[x2])
                    self.l_gtE.append(list_ground_truthE[x2])'''
            indexer = 0 
            
            self.l_gt = []
            self.l_gtE = []
            '''else : 
                self.l_gt = np.zeros([counter_seq,self.seq_length,136])
                self.l_gtE = np.zeros([counter_seq,self.seq_length,2])'''
                
            totalData = len(t_l_imgs)
            perSplit = int(truediv(totalData, nSplit))
            for x in listSplit :
                print('split : ',x) 
                begin = x*perSplit
                if x == nSplit-1 : 
                    end = begin + (totalData - begin)
                else : 
                    end = begin+perSplit
                print(begin,end,totalData)
                
                if not self.isVideo :
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): 
                            #print('append ',t_l_imgs[i][j])
                            self.l_imgs.append(t_l_imgs[i][j])
                            self.l_gt.append(t_l_gt[i][j])
                            self.l_gtE.append(t_l_gtE[i][j])
                            
                            self.list_gt_names.append(t_list_gt_names[i][j])
                            self.list_gtE_names.append(t_list_gtE_names[i][j])
                            indexer+=1
                            
                else : 
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): #seq counter
                        
                            t_img = []
                            t_gt = []
                            t_gtE = []
                            t_gt_N = []
                            t_gt_EN = []
                            tmp = 0
                            
                            for k in range(len(t_l_imgs[i][j])): #seq size
                                t_img.append(t_l_imgs[i][j][k])
                                t_gt.append(t_l_gt[i][j][k])
                                t_gtE.append(t_l_gtE[i][j][k])
                                
                                t_gt_N.append(t_list_gt_names[i][j][k])
                                t_gt_EN.append(t_list_gtE_names[i][j][k])
                                tmp+=1
                                
                            #print('append ',t_img)
                            self.l_imgs.append(t_img)
                            self.l_gt.append(t_gt)
                            self.l_gtE.append(t_gtE)
                            
                            self.list_gt_names.append(t_gt_N)
                            self.list_gtE_names.append(t_gt_EN)
                            indexer+=1
                    
                print(len(self.l_imgs))
                    
            self.l_gt = np.asarray(self.l_gt)
            self.l_gtE = np.asarray(self.l_gtE)
        else :
            if not self.isVideo :
                self.l_gt = np.zeros([counter_image,136])
                self.l_gtE = np.zeros([counter_image,2])
                indexer = 0
                
                
                for i in range(len(t_l_imgs)): 
                    for j in range(len(t_l_imgs[i])): 
                        self.l_imgs.append(t_l_imgs[i][j])
                        print(i,j,'-',len(t_l_imgs[i]))
                        self.l_gt[indexer] = t_l_gt[i][j]
                        self.l_gtE[indexer] = t_l_gtE[i][j]
                        
                        self.list_gt_names.append(t_list_gt_names[i][j])
                        self.list_gtE_names.append(t_list_gtE_names[i][j])
                        indexer+=1
                    
            else : 
                self.l_gt= np.zeros([counter_seq,self.seq_length,136])
                self.l_gtE = np.zeros([counter_seq,self.seq_length,2])
                
                indexer = 0
                
                for i in range(len(t_l_imgs)): #dataset
                    for j in range(len(t_l_imgs[i])): #seq counter
                        
                        t_img = []
                        
                        t_gt = np.zeros([self.seq_length,136])
                        t_gte = np.zeros([self.seq_length,2])
                        
                        t_gt_n = []
                        t_gt_en = []
                        i_t = 0
                        
                        for k in range(len(t_l_imgs[i][j])): #seq size
                            
                            t_img.append(t_l_imgs[i][j][k])
                            t_gt[i_t] = t_l_gt[i][j][k]
                            t_gte[i_t] = t_l_gtE[i][j][k]
                            
                            t_gt_n.append(t_list_gt_names[i][j][k])
                            t_gt_en.append(t_list_gtE_names[i][j][k])
                            
                            i_t+=1
                            
                        self.l_imgs.append(t_img)
                        self.l_gt[indexer] = t_gt
                        self.l_gtE[indexer] = t_gte
                        
                        self.list_gt_names.append(t_gt_n)
                        self.list_gtE_names.append(t_gt_en)
                        
                        indexer+=1
                        
        print('limgs : ',len(self.l_imgs))

    def __getitem__(self,index):
        #Read all data, transform etc.
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
        
        l_imgs = []; l_VA = []; l_ldmrk = []; l_nc = []#,torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]
        
        if not self.isVideo : 
            x_l = [self.l_imgs[index]];labelE_l =[self.l_gtE[index].copy()];label_l = [self.l_gt[index].copy()];label_n =[self.list_gt_names[index]] 
        else : 
            x_l = self.l_imgs[index];labelE_l =self.l_gtE[index].copy();label_l = self.l_gt[index].copy();label_n =self.list_gt_names[index]
        
        for x,labelE,label,ln in zip(x_l,labelE_l,label_l,label_n) : 
            #print(x,labelE,label,ln)
            tImage = Image.open(x).convert("RGB")
            tImageB = None
            
            if self.onlyFace and not self.skipCropping :    
                #crop the face region
                #t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 2,div_y = 2,images = cv2.imread(x),displacementxy = random.uniform(-.5,.5))
                t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label.copy(),div_x = 8,div_y = 8,images = cv2.imread(x))#,displacementxy = random.uniform(-.5,.5))
                
                area = (x1,y1, x2,y2)
                tImage =  tImage.crop(area)
                
                label[:68] -= x_min
                label[68:] -= y_min
                
                tImage = tImage.resize((self.imageWidth,self.imageHeight))
                
                label[:68] *= truediv(self.imageWidth,(x2 - x1))
                label[68:] *= truediv(self.imageHeight,(y2 - y1))
                
                
                #now aliging 
                if self.align and not self.skipCropping: 
                    tImageT = utils.PILtoOpenCV(tImage.copy())
                    if self.swap : 
                        ptsSource = torch.tensor([
                            [label[33+68],label[33]],[label[41+68],label[41]],[label[46+68],label[46]]
                            ])
                        ptsSn = [
                            [label[33+68],label[33]],[label[41+68],label[41]],[label[46+68],label[46]]
                            ]
                    else : 
                        ptsSource = torch.tensor([
                            [label[33],label[33+68]],[label[41],label[41+68]],[label[46],label[46+68]]
                            ])
                        ptsSn =[
                            [label[33],label[33+68]],[label[41],label[41+68]],[label[46],label[46+68]]
                            ]
                        
                        ptsSnFull = np.column_stack((label[:68],label[68:]))
                        ptsSnFull = np.asarray(ptsSnFull,np.float32)
                        
                    ptsSource = ptsSource.numpy()
                    ptsSource = np.asarray(ptsSource,np.float32)
                    
                    if self.useNudget : 
                        trans = nudged.estimate(ptsSn,self.ptsTn)
                        M = np.asarray(trans.get_matrix())[:2,:]
                        #print("Nudged : ",mN,trans.get_scale(),trans.get_rotation())
                    else :
                        #M = cv2.getAffineTransform(ptsSource,self.ptsDst)
                        #_,_,aff  = self.procrustes(ptsSource,self.ptsDst)
                         
                        #print(ptsSource.shape,'-', self.ptsDst.shape)
                        #print(ptsSnFull.shape,'-', self.ptsTnFull.shape)
                        
                        _,_,aff  = self.procrustes(self.ptsTnFull,ptsSnFull)
                        M = aff[:2,:]
                        
                    dst = cv2.warpAffine(tImageT,M,(self.imageWidth,self.imageHeight))
                    
                    #print(np.asarray(ptsSn).shape, np.asarray(self.ptsTn).shape,M.shape)
                    
                    
                    M_full = np.append(M,[[0,0,1]],axis = 0)
                    l_full = np.stack((label[:68],label[68:],np.ones(68)))
                    
                    ldmark = np.matmul(M_full, l_full)
                    
                    if False : 
                        print(ldmark)
                        for i in range(68) :
                            cv2.circle(dst,(int(scale(ldmark[0,i])),int(scale(ldmark[1,i]))),2,(0,255,0) )
                        
                        cv2.imshow('test align',dst)
                        cv2.waitKey(0)
                    
                    label = np.concatenate((ldmark[0],ldmark[1]))
                    tImage = utils.OpenCVtoPIL(dst)
                    
            newChannel = None
            
            if self.wHeatmap and not self.skipCropping: 
                theMiddleName = 'img'
                filePath = x.split(os.sep)
                ifolder = filePath.index(theMiddleName)
                
                print(ifolder)
                image_name = filePath[-1]
                
                annot_name_H = os.path.splitext(image_name)[0]+'.npy'
                
                sDirName = filePath[:ifolder]
                dHeatmaps = '/'.join(sDirName)+'/heatmaps'
                
                finalTargetH = dHeatmaps+'/'+annot_name_H
                print(finalTargetH)
                
                if isfile(finalTargetH) and False: 
                    newChannel  = np.load(finalTargetH)
                    newChannel = Image.fromarray(newChannel)
                else : 
                    checkDirMake(dHeatmaps)
                    
                    tImageTemp = cv2.cvtColor(np.array(tImage),cv2.COLOR_RGB2BGR)
                    #tImageTemp = cv2.imread(x)#tImage.copy()
                
                    b_channel,g_channel,r_channel = tImageTemp[:,:,0],tImageTemp[:,:,1],tImageTemp[:,:,2]
                    newChannel = b_channel.copy(); newChannel[:] = 0
                    
                    t0,t1,t2,t3 = utils.get_bb(label[0:68], label[68:])
                    
                    l_cd,rv = utils.get_list_heatmap(0,None,t2-t0,t3-t1,.05)
                    height, width,_ = tImageTemp.shape
                    
                    wx = t2-t0
                    wy = t3-t1
                
                    scaler = 255/np.max(rv)
                    
                    for iter in range(68) :
                        ix,iy = int(label[iter]),int(label[iter+68])
                        
                        #Now drawing given the center
                        for iter2 in range(len(l_cd)) : 
                            value = int(rv[iter2]*scaler)
                            if newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] < value : 
                                newChannel[utils.inBound(iy+l_cd[iter2][0],0,height-1), utils.inBound(ix + l_cd[iter2][1],0,width-1)] = int(rv[iter2]*scaler)#int(heatmapValue/2 + rv[iter2] * heatmapValue)
                    
                    '''tImage2 = cv2.merge((b_channel, newChannel,newChannel, newChannel))
                    cv2.imshow("combined",tImage2)
                    cv2.waitKey(0)'''
                    
                    np.save(finalTargetH,newChannel)
                    newChannel = Image.fromarray(newChannel)
            
            if self.augment : 
                sel = np.random.randint(0,4)
                #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
                if sel == 0 : 
                    pass
                elif sel == 1 : 
                    flip = RandomHorizontalFlip_WL(1)
                    tImage,label,newChannel = flip(tImage,label,newChannel)
                elif sel == 2 and not self.align : 
                    rot = RandomRotation_WL(45)
                    tImage,label,newChannel = rot(tImage,label,newChannel)
                elif sel == 3 : 
                    occ = Occlusion_WL(1)
                    tImage,label,newChannel = occ(tImage,label,newChannel)
                    
                #random crop
                if not self.align : 
                    rc = RandomResizedCrop_WL(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
                    tImage,label,newChannel= rc(tImage,label,newChannel)
                
                #additional blurring
                if (np.random.randint(1,3)%2==0) and False : 
                    sel_n = np.random.randint(1,6)
                    #sel_n = 4
                    rc = GeneralNoise_WL(1)
                    tImage,label= rc(tImage,label,sel_n,np.random.randint(0,3))
            
            if self.returnM : 
                if self.swap : 
                    ptsSource = torch.tensor([
                        [label[33+68],label[33]],[label[41+68],label[41]],[label[46+68],label[46]]
                        ])
                    ptsSn = [
                        [label[33+68],label[33]],[label[41+68],label[41]],[label[46+68],label[46]]
                        ]
                else : 
                    ptsSource = torch.tensor([
                        [label[33],label[33+68]],[label[41],label[41+68]],[label[46],label[46+68]]
                        ])
                    ptsSn =[
                        [label[33],label[33+68]],[label[41],label[41+68]],[label[46],label[46+68]]
                        ]
                    
                    ptsSnFull = np.column_stack((label[:68],label[68:]))
                    ptsSnFull = np.asarray(ptsSnFull,np.float32)
                    
                ptsSource = ptsSource.numpy()
                ptsSource = np.asarray(ptsSource,np.float32)
                
                if self.useNudget : 
                    trans = nudged.estimate(ptsSn,self.ptsTn)
                    M = np.asarray(trans.get_matrix())[:2,:]
                else :
                    #M = cv2.getAffineTransform(ptsSource,self.ptsDst)
                    _,_,aff  = self.procrustes(self.ptsTnFull,ptsSnFull)
                    M = aff[:2,:]    
                
                if False :
                    tImageT = utils.PILtoOpenCV(tImage.copy())
                    dst = cv2.warpAffine(tImageT,M,(self.imageWidth,self.imageHeight))
                    
                    print(np.asarray(ptsSn).shape, np.asarray(self.ptsTn).shape,M.shape)
                    
                    M_full = np.append(M,[[0,0,1]],axis = 0)
                    l_full = np.stack((label[:68],label[68:],np.ones(68)))
                    
                    ldmark = np.matmul(M_full, l_full)
                    print(ldmark)
                    for i in range(68) :
                        cv2.circle(dst,(int(scale(ldmark[0,i])),int(scale(ldmark[1,i]))),2,(0,0,255) )
                    
                    cv2.imshow('test recovered',dst)
                    cv2.waitKey(0)
                    
                    
                
                Minter = self.param2theta(np.append(M,[[0,0,1]],axis = 0), self.imageWidth,self.imageHeight)
                Mt = torch.from_numpy(Minter).float()
            else : 
                Mt = torch.zeros(1)
            
            
            if self.useIT : 
                tImage = self.transformInternal(tImage)
            else : 
                tImage = self.transform(tImage)
            
            
            
            
            if not self.wHeatmap : 
                l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label))#,x,self.list_gt_names[index]
            else : 
                newChannel = transforms.Resize(224)(newChannel)
                newChannel = transforms.ToTensor()(newChannel)
                newChannel = newChannel.sub(125)
                l_imgs.append(tImage); l_VA.append(torch.FloatTensor(labelE)); l_ldmrk.append(torch.FloatTensor(label)); l_nc.append(newChannel)
                #return tImage,torch.FloatTensor(labelE),torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]
                
                
                
        if not self.isVideo : 
            if self.wHeatmap : 
                return l_imgs[0], l_VA[0], l_ldmrk[0], l_nc[0], Mt
            else : 
                return l_imgs[0], l_VA[0], l_ldmrk[0], Mt
        else : 
            #lImgs = torch.Tensor(len(l_imgs),3,self.imageHeight,self.imageWidth)
            #lVA = torch.Tensor(len(l_VA),2)
            #lLD = torch.Tensor(len(l_ldmrk),136)
            lImgs = torch.stack(l_imgs)
            lVA = torch.stack(l_VA)
            lLD = torch.stack(l_ldmrk)
            
            #print(lImgs.shape, l_imgs[0].shape, l_VA[0].shape,len(lImgs))
            
            #torch.cat(l_imgs, out=lImgs)
            #torch.cat(l_VA, out=lVA)
            #torch.cat(l_ldmrk, out=lLD)
            
            if self.wHeatmap :
                #lnc = torch.Tensor(len(l_nc),1,self.imageHeight,self.imageWidth)
                #torch.cat(l_nc, out=lnc)
                lnc = torch.stack(l_nc)
                return lImgs, lVA, lLD, lnc, Mt
            else : 
                return lImgs, lVA, lLD, Mt
                
    
    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    
    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    
    
    def param2theta(self,param, w, h):
        param = np.linalg.inv(param)
        theta = np.zeros([2,3])
        theta[0,0] = param[0,0]
        theta[0,1] = param[0,1]*h/w
        theta[0,2] = param[0,2]*2/w + theta[0,0] + theta[0,1] - 1
        theta[1,0] = param[1,0]*w/h
        theta[1,1] = param[1,1]
        theta[1,2] = param[1,2]*2/h + theta[1,0] + theta[1,1] - 1
        return theta
    
    def procrustes(self, X, Y, scaling=True, reflection='best'):
        
        n,m = X.shape
        ny,my = Y.shape
    
        muX = X.mean(0)
        muY = Y.mean(0)
        
        X0 = X - muX
        Y0 = Y - muY
        
        ssX = (X0**2.).sum()
        ssY = (Y0**2.).sum()
        
        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)
        
        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY
        
        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)
        
        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U,s,Vt = np.linalg.svd(A,full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)
        
        if reflection is not 'best':
            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0
            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:,-1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)
        
        traceTA = s.sum()
        
        if scaling:
            # optimum scaling of Y
            b = traceTA * normX / normY
            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA**2
            # transformed coords
            Z = normX*traceTA*np.dot(Y0, T) + muX
        
        else:
            b = 1
            d = 1 + ssY/ssX - 2 * traceTA * normY / normX
            Z = normY*np.dot(Y0, T) + muX
        
        # transformation matrix
        if my < m:
            T = T[:my,:]
        
        c = muX - b*np.dot(muY, T)
        
        #transformation values 
        #tform = {'rotation':T, 'scale':b, 'translation':c}
        tform = np.append(b*T,[c],axis = 0).T
        tform = np.append(tform,[[0,0,1]],axis = 0)
        
        return d, Z, tform
    
    def __len__(self):
        return len(self.l_imgs)




class AFEWVAReduced(data.Dataset):
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["AFEW"],dir_gt = None,onlyFace = True, image_size =224, 
                 transform = None,useIT = False,augment = False, step = 1,split = False, 
                 nSplit = 5, listSplit = [0,1,2,3,4],wHeatmap= False,isVideo = False, seqLength = None,
                 returnM = False, toAlign = False, skipCropping = False):
        
        self.seq_length = seqLength 
        self.isVideo = isVideo
        
        self.transform = transform
        self.augment = augment 
        
        self.imageSize = image_size
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.useIT = useIT
        self.curDir = rootDir +"/"#/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/"
        
        list_labels_t = []
        list_labels_tE = []
        list_gt = []
        
        counter_image = 0
        annotE_name = 'annot2'
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        for data in data_list : 
            print(("Opening "+data))
            for f in file_walker.walk(self.curDir +data+"/"):
                if f.isDirectory: # Check if object is directory
                    #print((f.name, f.full_path)) # Name is without extension
                    for sub_f in f.walk():
                        if sub_f.isDirectory: # Check if object is directory
                            list_dta = []
                            #print(sub_f.name)
                            if(sub_f.name == annotE_name) : #If that's annot, add to labels_t
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_labels_tE.append(sorted(list_dta))
                                
                            elif(sub_f.name == 'img-128'): #Else it is the image
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_gt.append(sorted(list_dta))
                                counter_image+=len(list_dta)
        
        self.length = counter_image
        print("Now opening keylabels")
        
        
        
        list_labelsEN = []
         
        list_labelsE = []
        
        for ix in range(len(list_labels_tE)) : #lbl,lble in (list_labels_t,list_labels_tE) :
            lbl_2 = [] #Per folder
            lbl_n2 = [] #Per folder
            for jx in range(len (list_labels_tE[ix])): #lbl_sub in lbl :
                
                #print(os.path.basename(list_gt[ix][jx]))
                #print(os.path.basename(list_labels_t[ix][jx]))
                #print(os.path.basename(list_labels_tE[ix][jx]))
                
                    
                lbl_subE = list_labels_tE[ix][jx]
                if ('aro' in lbl_subE) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_subE) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        #x.append([ float(j) for j in data2[i][0].split()] )
                        temp = [ float(j) for j in data2[i][0].split()]
                        temp.reverse() #to give the valence first. then arousal
                        x.append(temp)
                        
                    lbl_2.append(np.array(x).flatten('F'))
                    lbl_n2.append(lbl_subE)
                
                
            list_labelsEN.append(lbl_n2)
            list_labelsE.append(lbl_2)
        
            
        '''for lbl in list_labels_t :
            lbl_68 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('pts' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
            list_labels.append(lbl_68)
        
        list_labelsE = []
        for lbl in list_labels_tE :
            lbl_2 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('aro' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_2.append(np.array(x).flatten('F')) #1 record
            list_labelsE.append(lbl_2)'''
        
        
        
        '''print(len(list_labelsN[0])) 
        print(len(list_labelsEN[0])) 
        print(len(list_labels[0])) 
        print(len(list_labelsE[0]))'''
                
        
        t_l_imgs = []
        t_l_gtE = []
        t_list_gtE_names = []
        
        
        
        if not self.isVideo :
            #Flatten it to one list
            for i in range(0,len(list_gt)): #For each dataset
                
                list_images = []
                list_gt_names = []
                list_gtE_names = []
                indexer = 0
                
                list_ground_truthE = np.zeros([len(list_gt[i]),2])
                
                for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                    list_images.append(list_gt[i][j])
                    
                    list_gtE_names.append(list_labelsEN[i][j])
                    list_ground_truthE[indexer] = np.array(list_labelsE[i][j]).flatten('F')
                    indexer += 1
                
                t_l_imgs.append(list_images)
                
                t_l_gtE.append(list_ground_truthE)
                t_list_gtE_names.append(list_gtE_names)
        
        else : 
            if self.seq_length is None :
                indexer = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                    counter = 0
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize
                        
                        temp = []
                        temp2 = np.zeros([self.seq_length,136])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z])
                            temp2[i_temp] = list_labels[i][z]
                            i_temp+=1
                            
                        list_images.append(temp)
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                self.l_imgs = list_images
            else : 
                counter_seq = 0;
                
                for i in range(0,len(list_gt)): #For each dataset
                
                    indexer = 0;    
                    list_gtE_names = []
                    list_ground_truthE = np.zeros([int(len(list_gt[i])/(self.seq_length*step)),self.seq_length,2])#np.zeros([counter_image,2])
                    
                    counter = 0
                    list_images = []
                    
                    for j in range(0,int(len(list_gt[i])/(self.seq_length*step))): #for number of data/batchsize
                        
                        tmpn2 = []
                        temp = []
                        temp3 = np.zeros([self.seq_length,2])
                        i_temp = 0
                        
                        for z in range(counter,counter+(self.seq_length*step),step):#1 to seq_size 
                            temp.append(list_gt[i][z])
                            
                            temp3[i_temp] = list_labelsE[i][z].flatten('F')
                            tmpn2.append(list_labelsEN[i][z])
                            
                            i_temp+=1
                            counter_seq+=1
                            
                        list_images.append(temp)
                        list_ground_truth[indexer] = temp2
                        list_ground_truthE[indexer] = temp3
                        
                        list_gt_names.append(tmpn68)
                        list_gtE_names.append(tmpn2)
                            
                        indexer += 1
                        counter+=self.seq_length*step
                        #print counter
                        
                    t_l_imgs.append(list_images)
                    t_l_gtE.append(list_ground_truthE)
                    
                    t_list_gtE_names.append(list_gtE_names)
                
        '''print('length : ',len(t_l_imgs)) #Folder
        print('lengt2 : ',len(t_l_imgs[0])) #all/seq
        print('lengt3 : ',len(t_l_imgs[0][0])) #seq
        print('length4 : ',len(t_l_imgs[0][0][0]))'''
        
        #[folder, all/seq,seq]
        
        self.l_imgs = []
        self.l_gtE = []
        self.list_gtE_names = []
        
        #print('cimage : ',counter_image)
        
        
        if split :
            '''print('splitting')
            self.l_imgs = []
            self.l_gt = []
            self.l_gtE = []
            
            totalData = len(list_images)
            perSplit = int(truediv(totalData, nSplit))
            for x in listSplit :
                print('split : ',x) 
                begin = x*perSplit
                if x == nSplit-1 : 
                    end = begin + (totalData - begin)
                else : 
                    end = begin+perSplit
                print(begin,end,totalData)
                for x2 in range(begin,end) : 
                    #print(x2,totalData)
                    self.l_imgs.append(list_images[x2])
                    self.l_gt.append(list_ground_truth[x2])
                    self.l_gtE.append(list_ground_truthE[x2])'''
            indexer = 0 
            
            self.l_gtE = []
            '''else : 
                self.l_gt = np.zeros([counter_seq,self.seq_length,136])
                self.l_gtE = np.zeros([counter_seq,self.seq_length,2])'''
                
            totalData = len(t_l_imgs)
            perSplit = int(truediv(totalData, nSplit))
            for x in listSplit :
                print('split : ',x) 
                begin = x*perSplit
                if x == nSplit-1 : 
                    end = begin + (totalData - begin)
                else : 
                    end = begin+perSplit
                print(begin,end,totalData)
                
                if not self.isVideo :
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): 
                            #print('append ',t_l_imgs[i][j])
                            self.l_imgs.append(t_l_imgs[i][j])
                            self.l_gtE.append(t_l_gtE[i][j])
                            
                            self.list_gtE_names.append(t_list_gtE_names[i][j])
                            indexer+=1
                            
                else : 
                    for i in range(begin,end) :
                        for j in range(len(t_l_imgs[i])): #seq counter
                        
                            t_img = []
                            t_gtE = []
                            t_gt_EN = []
                            tmp = 0
                            
                            for k in range(len(t_l_imgs[i][j])): #seq size
                                t_img.append(t_l_imgs[i][j][k])
                                t_gtE.append(t_l_gtE[i][j][k])
                                
                                t_gt_EN.append(t_list_gtE_names[i][j][k])
                                tmp+=1
                                
                            #print('append ',t_img)
                            self.l_imgs.append(t_img)
                            self.l_gtE.append(t_gtE)
                            
                            self.list_gtE_names.append(t_gt_EN)
                            indexer+=1
                    
                print(len(self.l_imgs))
                    
            self.l_gtE = np.asarray(self.l_gtE)
        else :
            if not self.isVideo :
                self.l_gtE = np.zeros([counter_image,2])
                indexer = 0
                
                
                for i in range(len(t_l_imgs)): 
                    for j in range(len(t_l_imgs[i])): 
                        self.l_imgs.append(t_l_imgs[i][j])
                        print(i,j,'-',len(t_l_imgs[i]))
                        self.l_gtE[indexer] = t_l_gtE[i][j]
                        self.list_gtE_names.append(t_list_gtE_names[i][j])
                        indexer+=1
                    
            else : 
                self.l_gt= np.zeros([counter_seq,self.seq_length,136])
                self.l_gtE = np.zeros([counter_seq,self.seq_length,2])
                
                indexer = 0
                
                for i in range(len(t_l_imgs)): #dataset
                    for j in range(len(t_l_imgs[i])): #seq counter
                        
                        t_img = []
                        
                        t_gte = np.zeros([self.seq_length,2])
                        
                        t_gt_en = []
                        i_t = 0
                        
                        for k in range(len(t_l_imgs[i][j])): #seq size
                            
                            t_img.append(t_l_imgs[i][j][k])
                            t_gte[i_t] = t_l_gtE[i][j][k]
                            
                            t_gt_en.append(t_list_gtE_names[i][j][k])
                            
                            i_t+=1
                            
                        self.l_imgs.append(t_img)
                        self.l_gtE[indexer] = t_gte
                        
                        self.list_gtE_names.append(t_gt_en)
                        
                        indexer+=1
                        
        print('limgs : ',len(self.l_imgs))

    def __getitem__(self,index):
        #Read all data, transform etc.
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
        
        l_imgs = []; l_VA = []; l_ldmrk = []; l_nc = []#,torch.FloatTensor(label),newChannel#,x,self.list_gt_names[index]
        
        if not self.isVideo : 
            x_l = [self.l_imgs[index]];labelE_l =[self.l_gtE[index].copy()];label_n =[self.list_gtE_names[index]] 
        else : 
            x_l = self.l_imgs[index];labelE_l =self.l_gtE[index].copy();label_n =self.list_gtE_names[index]
        
        for x,labelE,ln in zip(x_l,labelE_l,label_n) : 
            #print(x,labelE,label,ln)
            tImage = Image.open(x).convert("RGB")
            tImageB = None
            
            
            if self.augment : 
                sel = np.random.randint(0,4)
                #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
                if sel == 0 : 
                    pass
                elif sel == 1 : 
                    #flip = RandomHorizontalFlip_WL(1)
                    #tImage,label,newChannel = flip(tImage,label,newChannel)
                    flip = RandomHorizontalFlip(1)
                    tImage = flip(tImage)
                elif sel == 2 : 
                    #rot = RandomRotation_WL(45)
                    #tImage,label,newChannel = rot(tImage,label,newChannel)
                    rot = RandomRotation(45)
                    tImage = rot(tImage)
                elif sel == 3 : 
                    #occ = Occlusion_WL(1)
                    #tImage,label,newChannel = occ(tImage,label,newChannel)
                    occ = Occlusion(1)
                    tImage = occ(tImage)
                    
                #random crop
                if True : 
                    #rc = RandomResizedCrop_WL(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
                    #tImage,label,newChannel= rc(tImage,label,newChannel)
                    
                    rc = RandomResizedCrop(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
                    tImage= rc(tImage)
                
                #additional blurring
                if (np.random.randint(1,3)%2==0) and False : 
                    sel_n = np.random.randint(1,6)
                    #sel_n = 4
                    #rc = GeneralNoise_WL(1)
                    #tImage,label= rc(tImage,label,sel_n,np.random.randint(0,3))
                    
                    rc = GeneralNoise(1)
                    tImage = rc(tImage,sel_n,np.random.randint(0,3))
                    
            Mt = torch.zeros(1)
            #l_ldmrk = torch.zeros(1) 
            
            if self.useIT : 
                tImage = self.transformInternal(tImage)
            else : 
                tImage = self.transform(tImage)
                
            
            l_imgs.append(tImage); 
            l_VA.append(torch.FloatTensor(labelE)); 
            l_ldmrk.append(torch.FloatTensor(Mt))
                
        if not self.isVideo : 
            '''print(l_imgs[0]) 
            print(l_VA[0]) 
            print(l_ldmrk) 
            print(Mt)'''
            return l_imgs[0], l_VA[0], l_ldmrk[0], Mt
        else : 
            #lImgs = torch.Tensor(len(l_imgs),3,self.imageHeight,self.imageWidth)
            #lVA = torch.Tensor(len(l_VA),2)
            #lLD = torch.Tensor(len(l_ldmrk),136)
            lImgs = torch.stack(l_imgs)
            lVA = torch.stack(l_VA)
            lLD = torch.stack(l_ldmrk)
            
            #print(lImgs.shape, l_imgs[0].shape, l_VA[0].shape,len(lImgs))
            
            #torch.cat(l_imgs, out=lImgs)
            #torch.cat(l_VA, out=lVA)
            #torch.cat(l_ldmrk, out=lLD)
            
            if self.wHeatmap :
                #lnc = torch.Tensor(len(l_nc),1,self.imageHeight,self.imageWidth)
                #torch.cat(l_nc, out=lnc)
                lnc = torch.stack(l_nc)
                return lImgs, lVA, lLD, lnc, Mt
            else : 
                return lImgs, lVA, lLD, Mt

    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    
    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    
    def __len__(self):
        return len(self.l_imgs)



class FacialLandmarkDataset(data.Dataset):
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["300VW-Train"],dir_gt = None,onlyFace = True, image_size =224, 
                 transform = None,useIT = False,augment = False, step = 1):
        
        self.transform = transform
        self.onlyFace = onlyFace
        self.augment = augment 
        
        self.imageSize = image_size
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.useIT = useIT
        self.curDir = rootDirLdmrk# "/home/deckyal/eclipse-workspace/FaceTracking/"
        
        list_gt = []
        list_labels_t = []
        
        counter_image = 0
        annot_name = 'annot'
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        for data in data_list : 
            print(("Opening "+data))
            for f in file_walker.walk(self.curDir + "images/"+data+"/"):
                if f.isDirectory: # Check if object is directory
                    #print((f.name, f.full_path)) # Name is without extension
                    for sub_f in f.walk():
                        if sub_f.isDirectory: # Check if object is directory
                            list_dta = []
                            if(sub_f.name == annot_name) : #If that's annot, add to labels_t
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_labels_t.append(sorted(list_dta))
                            elif(sub_f.name == 'img'): #Else it is the image
                                for sub_sub_f in sub_f.walk(): #this is the data
                                    if(".npy" not in sub_sub_f.full_path):
                                        list_dta.append(sub_sub_f.full_path)
                                list_gt.append(sorted(list_dta))
                                counter_image+=len(list_dta)
        
        self.length = counter_image 
        print("Now opening keylabels")
        
        list_labels = []     
        for lbl in list_labels_t :
            lbl_68 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('pts' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
            list_labels.append(lbl_68)
            
        list_images = []
        list_ground_truth = np.zeros([counter_image,136])
        
        #Flatten it to one list
        indexer = 0
        for i in range(0,len(list_gt)): #For each dataset
            for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                list_images.append(list_gt[i][j])
                list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                indexer += 1
                
        self.l_imgs = list_images
        self.l_gt = list_ground_truth

    def __getitem__(self,index):
        #Read all data, transform etc.
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
         
        x,label  = self.l_imgs[index],self.l_gt[index].copy()
        
        tImage = Image.open(x).convert("RGB")
        tImageB = None
        
        if self.onlyFace :    
            #crop the face region
            #t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 2,div_y = 2,images = cv2.imread(x),displacementxy = random.uniform(-.5,.5))
            t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 8,div_y = 8,images = cv2.imread(x))#,displacementxy = random.uniform(-.5,.5))
            
            area = (x1,y1, x2,y2)
            tImage =  tImage.crop(area)
            
            label[:68] -= x_min
            label[68:] -= y_min
            
            tImage = tImage.resize((self.imageWidth,self.imageWidth))
            
            #print(self.imageWidth/(x2 - x1))
            #print(self.imageHeight/(y2 - y1))
            
            label[:68] *= truediv(self.imageWidth,(x2 - x1))
            label[68:] *= truediv(self.imageHeight,(y2 - y1))
        
        '''print(label)
        image = utils.imageLandmarking(tImage,label)
        cv2.imshow('tt',image)
        cv2.waitKey(0)'''
        '''if self.noiseType is not None :
            tImageB = tImage.copy()
            
            #print(tImageB.size)
            
            if self.injectedNoise is None : 
                
                noiseType = np.random.randint(0,5)
                noiseLevel = np.random.randint(0,self.noiseParam)
                noiseParam=noiseParamList[noiseType,noiseLevel]
                
            else : 
                noiseType =self.injectedNoise[0]
                if self.injectedNoise[1] < 0 : 
                    noiseParam = noiseParamListTrain[self.injectedNoise[0],np.random.randint(0,self.noiseParam)]
                else : 
                    noiseParam=noiseParamList[self.injectedNoise[0],self.injectedNoise[1]]
            
            tImageB = generalNoise(tImageB,noiseType,noiseParam)
            
            if self.transform is not None:
                tImageB = self.transform(tImageB)
        '''
        
        if self.augment : 
            sel = np.random.randint(0,4)
            #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
            if sel == 0 : 
                pass
            elif sel == 1 : 
                flip = RandomHorizontalFlip_WL(1)
                tImage,label = flip(tImage,label)
            elif sel == 2 : 
                rot = RandomRotation_WL(45)
                tImage,label = rot(tImage,label)
            elif sel == 3 : 
                occ = Occlusion_WL(1)
                tImage,label = occ(tImage,label)
                
            #random crop
            if True : 
                rc = RandomResizedCrop_WL(size = self.imageSize,scale = (0.75,1), ratio = (0.75, 1.33))
                tImage,label= rc(tImage,label)
            
            #additional blurring
            if (np.random.randint(1,3)%2==0) and True: 
                sel_n = np.random.randint(1,3)
                rc = GeneralNoise_WL(1)
                tImage,label= rc(tImage,label,sel_n,2)
            
        if self.useIT : 
            tImage = self.transformInternal(tImage)
        else : 
            tImage = self.transform(tImage)
        
        return tImage,torch.FloatTensor(label)
    
    
    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    
    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    
    def __len__(self):
        return len(self.l_imgs)






class FamilyDataset(data.Dataset):
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = [''],transforms = None,dirImage = None,withLandmark = False, withHeatmap=False,return_data = False,
                 useIT = False,convertGT = False, multiLabel = False,convertToOneHot  = False, dirList = ['landmarks','heatmaps'],isTest = False,
                 augment = True,rl = False,calculateWeight = False):
        
        self.aug = augment
        self.isTest = isTest
        self.tDirL = curDir+"data/"+dirList[0]+"/FIDs/"
        self.tDirH = curDir+"data/"+dirList[1]+"/FIDs/"
        self.theMiddleName = "FIDs"
        self.useIT = useIT
        self.cGT = convertGT
        self.mLBL = multiLabel
        self.reduce_label = rl
        self.cw = calculateWeight
        self.labelWeight = None
        
        self.toOneHot = convertToOneHot
        self.cIndex = []
        
        self.l_imgs = []
        self.l_gt = []
        self.transforms = transforms
        self.dirImage = dirImage
        self.withLandmark = withLandmark
        self.withHeatmap = withHeatmap
        self.return_data = return_data
        
        tmp = []
        if self.mLBL : 
            for x in data_list : 
                print(x)
                alist = [line.rstrip() for line in open(curDir+"data/"+x)]
                for data in alist :
                    d = data.split(' ')
                    imgName = d[0];
                    if self.isTest : 
                        imgLabel = d[0].split('/')[0][1:]
                    else : 
                        imgLabel = int(d[1][1:])
                    #print('imgName',imgName,imgLabel,'-',x)
                    self.l_imgs.append([imgName])
                    
                    tmp.append([int(imgLabel)])
                    self.l_gt.append([imgLabel])
            print(x)
            if 'trainVal' in x : 
                unique,counts = np.unique(np.sort(np.asarray(tmp)),return_counts = True)
                self.cIndex = unique
                #print(unique,counts)
                if self.cw  :
                    #print(np.asarray(tmp).shape)
                    '''#now normalize
                    counts = np.asarray(counts,dtype =np.float32)
                    print(counts,np.sum(counts))
                    counts/=np.sum(counts)
                    self.label_weight = np.ones(600)
                    #print(counts,'sum here ', np.sum(counts))
                    self.label_weight[:counts.shape[0]]/=(counts)#np.array([unique,counts])
                    self.label_weight/=100'''
                    
                    counts = np.asarray(counts,dtype =np.float32)
                    print(counts,np.sum(counts))
                    counts = np.sum(counts) / counts
                    self.label_weight = np.zeros(540)
                    #print(counts,'sum here ', np.sum(counts))
                    self.label_weight[:counts.shape[0]]+=(counts)#np.array([unique,counts])
                    self.label_weight/=100
                    
                    print(self.label_weight)
                    
                    np.save(curDir+'cWeight.npy',self.label_weight)
                    
                np.save(curDir+'cIndex.npy',self.cIndex)
            else : 
                self.cIndex = np.load(curDir+'cIndex.npy')
                self.label_weight = np.load(curDir+'cWeight.npy')
                
                    
            #exit(0)
            
        else : 
            #open csv put the list files and gt in the list 
            for x in data_list : 
                #print(x)
                with open(curDir+"data/"+x, newline='') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    for row in spamreader:
                        data = row[0].split(',');
                        
                        if not 'p1' in data :
                            if self.isTest : 
                                #self.l_gt.append([data[0],data[1]])
                                id1 = data[0].split('/')[0]
                                id2 = data[1].split('/')[0]
                                if id1 == id2 : 
                                    lbl = 1
                                else : 
                                    lbl = 0
                                self.l_gt.append([0,lbl])
                                self.l_imgs.append([data[0],data[1]])
                                
                                #print(x,'-',data[0],data[1],lbl)
                            else :
                                #print(x,'-',data[0],data[1],data[2],data[3],len(self.l_imgs))
                                self.l_gt.append([data[0],data[1]])
                                self.l_imgs.append([data[2],data[3]])
                                
    def __getitem__(self,index):
        
        #print('self t data index',self.l_gt[index])
        x,label  = self.l_imgs[index],self.l_gt[index]
        #print('source',x,label)
        aImgs = []
        ldmarks = []
        heatmaps = []
        limgs = []
        
        for ix in x :
            if self.withLandmark or self.withHeatmap :
                 
                filePath = (self.dirImage+ix).split(os.sep)
                #print(filePath)
                ifolder = filePath.index(self.theMiddleName)
                
                image_name = filePath[-1]
                
                annot_name_L = os.path.splitext(image_name)[0]+'.pts'
                annot_name_H = os.path.splitext(image_name)[0]+'.npy'
                
                sDirName = filePath[ifolder+1:-1]
                
                dLandmarks = self.tDirL+'/'.join(sDirName)
                dHeatmaps = self.tDirH+'/'.join(sDirName)
                
                checkDirMake(dLandmarks)
                checkDirMake(dHeatmaps)
                
                #print(filePath,sDirName)
                #print(dLandmarks,dHeatmaps)
                
                finalTargetL = dLandmarks+'/'+annot_name_L
                finalTargetH = dHeatmaps+'/'+annot_name_H
                
                if self.withLandmark : 
                    #Get the file name 
                    ldmrk = np.asarray(utils.read_kp_file(finalTargetL)).flatten('F')
                    ldmarks.append(torch.from_numpy(ldmrk))
                
                if self.withHeatmap :
                    htmp = np.load(finalTargetH)
                    
                    '''#Manual transform.
                    if self.useIT : 
                        htmp = torch.from_numpy(htmp).float()
                        #tImage = self.transform(htmp)
                    else :
                        
                        htmp = torch.from_numpy(htmp).float().div(255) 
                        mean = self.transforms['NormalizeM'][0]
                        std = self.transforms['NormalizeM'][1]
                        
                        #print(mean,std,torch.max(htmp))
                        htmp.sub_(mean).div_(std)
                        #print(mean,std,torch.max(htmp))
                        #print(htmp)
                        #img = Image.fromarray(htmp, 'L')'''
                    
                    
                    #heatmaps.append(htmp.unsqueeze(0))
                    #test = self.transforms['train1D'](img)\
            limgs.append(ix)
            #print(self.dirImage+ix)
            tImage = Image.open(self.dirImage+ix).convert("RGB")
            
            '''if self.withHeatmap and False : 
                opencvImage = cv2.cvtColor(np.array(tImage), cv2.COLOR_RGB2BGR)
                b_channel,g_channel,r_channel = opencvImage[:,:,0],opencvImage[:,:,1],opencvImage[:,:,2]
                tImageCV = cv2.merge((b_channel, g_channel,r_channel, htmp))'''
            
            if self.transforms is not None:
                if self.useIT : 
                    if self.withHeatmap :
                        htmp = Image.fromarray(htmp)
                        tImage,htmp= self.transform(tImage,htmp)
                        
                        '''opencvImage = cv2.cvtColor(np.array(ti), cv2.COLOR_RGB2BGR)
                        htmp = np.array(ht)
                        
                        b_channel,g_channel,r_channel = opencvImage[:,:,0],opencvImage[:,:,1],opencvImage[:,:,2]
                        tImageCV = cv2.merge((b_channel, htmp,htmp, htmp))
                        cv2.imshow('t',tImageCV)
                        cv2.waitKey(0)
                        '''
                        heatmaps.append(htmp)
                    else :
                        tImage = self.transform(tImage,None)
                else : 
                    tImage = self.transforms['train'](tImage)
                #print(tImage,torch.max(tImage))
            aImgs.append(tImage)
        
        if self.mLBL : 
            
            theLabel = int(label[0])
            #print(theLabel)
            if self.reduce_label : 
                theLabel = np.argwhere(self.cIndex==theLabel)[0][0] + 1###shifted by one!!! 0 -999
                #print('cgd',theLabel,self.cIndex[theLabel-1],self.cIndex.shape,np.max(self.cIndex))
                #exit(0)
            #print(convertToOneHot(np.asarray([theLabel-1]), 1000).shape)
            if self.toOneHot : 
                gt = torch.LongTensor(convertToOneHot(np.asarray([theLabel-1]), 1000)).squeeze();
            else :
                if self.reduce_label : 
                    gt = torch.LongTensor([theLabel-1]);
                else : 
                    gt = torch.LongTensor([theLabel-1]);
            
            return aImgs[0],gt,heatmaps,ldmarks,limgs[0] 
        else : 
            
            img1 = aImgs[0]; img2 = aImgs[1]; 
            fold = torch.FloatTensor([int(label[0])]); 
            
            theLabel = int(label[1])
            
            if self.cGT : 
                if theLabel > 0 : 
                    theLabel *= 1 
                else : 
                    theLabel = -1
                    
            gt = torch.FloatTensor([theLabel]); 
            
            #print('f',fold,'gt',gt,label[0],label[1])
            #print(img1.size(),img2.size())
            
            if self.return_data : 
                return img1,img2,fold,gt,heatmaps,ldmarks,limgs[0],limgs[1]
            else : 
                return img1,img2,fold,gt,heatmaps,ldmarks

    
    def __len__(self):
        
        return len(self.l_imgs)
    
    
    def transform(self, img,htmp):
        
        '''img2 = np.array(img, dtype=np.uint8)
        cv2.imshow('tt',img2)
        cv2.waitKey(0)'''
        
        img = transforms.Resize(224)(img)
        if not self.isTest : 
            if not self.withHeatmap : 
                img = transforms.RandomResizedCrop(224,scale=(0.5, 1.0))(img)
                img = transforms.RandomHorizontalFlip()(img)
                img = transforms.RandomRotation(45)(img)
            else : 
                img,htmp = RandomResizedCrop_m(224,scale=(0.5, 1.0))(img,htmp)
                img,htmp = RandomHorizontalFlip_m()(img,htmp)
                img,htmp = RandomRotation_m(45)(img,htmp)
            
        imgTemp = np.array(img, dtype=np.uint8)
        #htmpTemp = np.array(htmp, dtype=np.uint8)
        '''cv2.imshow('tt',img)
        cv2.waitKey(0)'''
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = imgTemp.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        if self.withHeatmap : 
            return img,transforms.ToTensor()(htmp).float()
        else:  
            return img
    
    
    
    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl


class ImageDatasets(data.Dataset):
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["300VW-Train"],dir_gt = None,onlyFace = True,step = 1, 
                 transform = None, image_size =224, noiseType = None, 
                 noiseParam = None,injectedNoise = None,return_noise_label = False,useIT = False):
        
        self.return_noise_label = return_noise_label
        self.useIT = useIT
        
        self.injectedNoise = injectedNoise
        self.noiseType  = noiseType
        self.noiseParam =noiseParam
        
        self.transform = transform
        self.onlyFace = onlyFace
        
        self.imageHeight = image_size
        self.imageWidth = image_size
        
        list_gt = []
        list_labels_t = []
        
        counter_image = 0
        annot_name = 'annot'
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        for data in data_list : 
            print(("Opening "+data))
            for f in file_walker.walk(curDir + "images/"+data+"/"):
                if f.isDirectory: # Check if object is directory
                    #print((f.name, f.full_path)) # Name is without extension
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
        
        self.length = counter_image 
        print("Now opening keylabels")
        
        list_labels = []     
        for lbl in list_labels_t :
            lbl_68 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('pts' in lbl_sub) : 
                    x = []
                    #print(lbl_sub)
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
            list_labels.append(lbl_68)
            
        list_images = []
        list_ground_truth = np.zeros([counter_image,136])
        
        #Flatten it to one list
        indexer = 0
        for i in range(0,len(list_gt)): #For each dataset
            for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                list_images.append(list_gt[i][j])
                list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                indexer += 1
                
        self.l_imgs = list_images
        self.l_gt = list_ground_truth

    def __getitem__(self,index):
        
        #Read all data, transform etc.
        
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
         
        x,label  = self.l_imgs[index],self.l_gt[index].copy()
        #print(x,label)
        
        #print(label)
        
        tImage = Image.open(x).convert("RGB")
        tImageB = None
        
        if self.onlyFace :    
            #crop the face region
            #t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 2,div_y = 2,images = cv2.imread(x),displacementxy = random.uniform(-.5,.5))
            t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 8,div_y = 8,images = cv2.imread(x))#,displacementxy = random.uniform(-.5,.5))
            area = (x1,y1, x2,y2)
            tImage =  tImage.crop(area)
            tImage = tImage.resize((self.imageWidth,self.imageWidth))
            
            label[:68] -= x_min
            label[68:] -= y_min
            
            label[:68] *= self.imageWidth/(x2 - x1)
            label[68:] *= self.imageHeight/(y2 - y1)
            
        if self.noiseType is not None :
            tImageB = tImage.copy()
            
            #print(tImageB.size)
            
            if self.injectedNoise is None : 
                
                noiseType = np.random.randint(0,5)
                noiseLevel = np.random.randint(0,self.noiseParam)
                noiseParam=noiseParamList[noiseType,noiseLevel]
                
            else : 
                noiseType =self.injectedNoise[0]
                if self.injectedNoise[1] < 0 : 
                    noiseParam = noiseParamListTrain[self.injectedNoise[0],np.random.randint(0,self.noiseParam)]
                else : 
                    noiseParam=noiseParamList[self.injectedNoise[0],self.injectedNoise[1]]
            
            tImageB = generalNoise(tImageB,noiseType,noiseParam)
            
            if self.transform is not None:
                tImageB = self.transform(tImageB)
        
        if self.useIT : 
            tImage = self.transformInternal(tImage)
        else : 
            tImage = self.transform(tImage)
        
        if self.noiseType : 
            if self.return_noise_label : 
                return tImage,tImageB,torch.FloatTensor(label),noiseType
            else  : 
                return tImage,tImageB,torch.FloatTensor(label)
        else : 
            return tImage,torch.FloatTensor(label)
    
    
    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    
    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    
    def __len__(self):
        return len(self.l_imgs)
####

class VideoDataset(data.Dataset):
    
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])  # from resnet50_ft.prototxt
    
    def __init__(self, data_list = ["300VW-Train"],dir_gt = None,onlyFace = True, seq_length = 2, step = 1,  transform = None,
                 image_size = 224,noiseType = None, noiseParam = None,useIT = False):
        self.useIT = useIT
        self.seq_length = seq_length
        self.onlyFace = onlyFace
        self.noiseType = noiseType
        self.noiseParam = noiseParam
        
        self.transform = transform
        self.imageHeight = image_size
        self.imageWidth = image_size
        
        
        list_gt = []
        list_labels_t = []
        
        counter_image = 0
        annot_name = 'annot'
        
        if dir_gt is not None : 
            annot_name = dir_gt
        for data in data_list : 
            print(("Opening "+data))
            for f in file_walker.walk(curDir + "images/"+data+"/"):
                if f.isDirectory: # Check if object is directory
                    #print((f.name, f.full_path)) # Name is without extension
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
        
        self.length = counter_image 
        print("Now opening keylabels")
        
        list_labels = []     
        for lbl in list_labels_t :
            lbl_68 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('pts' in lbl_sub) : 
                    x = []
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
            list_labels.append(lbl_68)
            
        list_images = []
        
        if seq_length is not None : 
            
            list_ground_truth = np.zeros([int(counter_image/(seq_length*step)),seq_length,136])
            indexer = 0;
            
            for i in range(0,len(list_gt)): #For each dataset
                counter = 0
                for j in range(0,int(len(list_gt[i])/(seq_length*step))): #for number of data/batchsize
                    
                    temp = []
                    temp2 = np.zeros([seq_length,136])
                    i_temp = 0
                    
                    for z in range(counter,counter+(seq_length*step),step):#1 to seq_size 
                        temp.append(list_gt[i][z])
                        temp2[i_temp] = list_labels[i][z]
                        i_temp+=1
                        
                    list_images.append(temp)
                    list_ground_truth[indexer] = temp2
                        
                    indexer += 1
                    counter+=seq_length*step
                    #print counter
        
        self.l_imgs = list_images
        self.l_gt = list_ground_truth
        
        #print(list_images)
    
    def __getitem__(self,index):
        
        #Read all data, transform etc.
        
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
        images = []
        blurred_images = []
        label = []
         
        for x,y in zip(self.l_imgs[index],self.l_gt[index].copy()) : 
            tImage = Image.open(x)
            if self.onlyFace : 
                
                #crop the face region
                t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(y,2,2,cv2.imread(x))
                area = (x1,y1, x2,y2)
                tImage =  tImage.crop(area)
                tImage = tImage.resize((self.imageWidth,self.imageHeight))
            
                
                #align the landmark 
                y[:68] -= x_min
                y[68:] -= y_min
                    
                y[:68] *= 224/(x2 - x1)
                y[68:] *= 224/(y2 - y1)
            
            tImageB = tImage.copy()
            
            if self.noiseType is not None :
                if self.noiseType == 1 : 
                     
                    for i in range(self.noiseParam) :#Scale down (/2) blurLevel times 
                        width, height = tImageB.size
                        tImageB = tImageB.resize((width//2,height//2))
                    
                elif self.noiseType == 2 : 
                    tImageB = tImage.copy().filter(ImageFilter.GaussianBlur(self.noiseParam))
                    #tImageB = tImageB.resize((224,224))
                    
                if self.transform is not None:
                    tImageB = self.transform(tImageB)

                blurred_images.append(tImageB)
                
            if self.useIT : 
                tImage = self.transformInternal(tImage)
            else : 
                tImage = self.transform(tImage)
            
            
            images.append(tImage)
            label.append(y)
            
        #print(images, label )
        
        if self.noiseType : 
            return torch.stack(images),torch.stack(blurred_images),torch.FloatTensor(label)
        else : 
            return torch.stack(images),torch.FloatTensor(label)
    
    
    def transformInternal(self, img):
        transforms.Resize(224)(img)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img
    
    
    def untransformInternal(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    
    
    def __len__(self):
        #Len
        return len(self.l_imgs)



##########
class ImageDatasetsExtract(data.Dataset):
    
    def __init__(self, data_list = ["300VW-Train"],dir_gt = None,onlyFace = True,step = 1, transform = None, image_size =224, noiseType = None, noiseParam = None,injectedNoise = None,return_noise_label = False,injectedLink = None):
        
        #Noise type : 
         
        #1. Downsample
        #2. Gaussian Bluring
        #3. Gaussian Noise 
        
        #4. Most significant bit 
        

        ''''initialization'
        #Read initalizes the list of file path and possibliy label as well. 
        label = [0,1,2,3,4,5,6,7]
        images = []
        labels = []
        
        for l in label : 
            dir = troot+str(l)
            for file in os.listdir(dir):
                if file.endswith(".jpg"):
                    images.append(os.path.join(dir, file))
                
                if file.endswith(".pts"):
                    labels.append(os.path.join(dir, file))
                
        self.labels = labels 
        self.images = images
        self.transform = transform'''
        self.return_noise_label = return_noise_label
        
        self.injectedNoise = injectedNoise
        self.noiseType  = noiseType
        self.noiseParam =noiseParam
        
        self.transform = transform
        self.onlyFace = onlyFace
        
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.injectedLink = injectedLink
        
        
        list_gt = []
        list_labels_t = []
        
        counter_image = 0
        annot_name = 'annot'
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        for data in data_list : 
            print(("Opening "+data))
            
            if self.injectedLink is not None : 
                the_direction = self.injectedLink
            else :
                the_direction = curDir + "images/"+data+"/"
            
            for f in file_walker.walk(the_direction):
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
        
        self.length = counter_image 
        print("Now opening keylabels")
        
        list_labels = []     
        for lbl in list_labels_t :
            lbl_68 = [] #Per folder
            for lbl_sub in lbl :
                #print(lbl_sub)
                if ('pts' in lbl_sub) : 
                    x = []
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
            list_labels.append(lbl_68)
            
        list_images = []
        list_ground_truth = np.zeros([counter_image,136])
        
        #Flatten it to one list
        indexer = 0
        for i in range(0,len(list_gt)): #For each dataset
            for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                list_images.append(list_gt[i][j])
                
                list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                indexer += 1
                
        self.l_imgs = list_images
        self.l_gt = list_ground_truth

    def __getitem__(self,index):
        
        #Read all data, transform etc.
        
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
         
        x,label  = self.l_imgs[index],self.l_gt[index].copy()
        #print(x,label)
        
        #print(label)
        
        tImage = Image.open(x).convert("RGB")
        tImageB = None
        
        if self.onlyFace :    
            #crop the face region
            t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 2,div_y = 2,images = cv2.imread(x))
            area = (x1,y1, x2,y2)
            tImage =  tImage.crop(area)
            tImage = tImage.resize((self.imageWidth,self.imageWidth))
            
            label[:68] -= x_min
            label[68:] -= y_min
            
            label[:68] *= self.imageWidth/(x2 - x1)
            label[68:] *= self.imageHeight/(y2 - y1)
            
        if self.noiseType is not None :
            tImageB = tImage.copy()
            
            #print(tImageB.size)
            
            if self.injectedNoise is None : 
                
                noiseType = np.random.randint(0,5)
                noiseLevel = np.random.randint(0,self.noiseParam)
                noiseParam=noiseParamList[noiseType,noiseLevel]
                
            else : 
                noiseType =self.injectedNoise[0]
                if self.injectedNoise[1] < 0 : 
                    noiseParam = noiseParamListTrain[self.injectedNoise[0],np.random.randint(0,self.noiseParam)]
                else : 
                    noiseParam=noiseParamList[self.injectedNoise[0],self.injectedNoise[1]]
            
            #print(noiseType,noiseParam)
            '''if self.noiseType == 1: #downsample
                for i in range(int(self.noiseParam/2)) :#Scale down (/2) blurLevel times 
                    width, height = tImageB.size
                    tImageB = tImageB.resize((width//2,height//2))
                    #print(tImageB.size)
            elif self.noiseType == 2 : #Gaussian blur
                tImageB = tImageB.filter(ImageFilter.GaussianBlur(self.noiseParam))
            elif self.noiseType == 3 : #Gaussian noise 
                #tImageB = addNoise(tImageB)
                #convert to opencv 
                opencvImage = cv2.cvtColor(np.array(tImageB), cv2.COLOR_RGB2BGR)
                
                #print(opencvImage)
                opencvImage = addNoise(opencvImage)
                pilImage =  cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)
                #tImageB = Image.fromarray(random_noise(opencvImage))
                tImageB = Image.fromarray(pilImage)'''
            
            tImageB = generalNoise(tImageB,noiseType,noiseParam)
            
            if self.transform is not None:
                tImageB = self.transform(tImageB)
        
        if self.transform is not None:
            tImage = self.transform(tImage)
        
        if self.noiseType : 
            if self.return_noise_label : 
                return tImage,tImageB,torch.FloatTensor(label),noiseType
            else  : 
                return tImage,tImageB,torch.FloatTensor(label),x
        else : 
            return tImage,torch.FloatTensor(label),x
    
    
    
    def __len__(self):
        
        return len(self.l_imgs)
    
    
class ImageDatasetsClean(data.Dataset):
    
    def __init__(self, data_list = ["300VW-Train"],dir_gt = None,onlyFace = True,step = 1,
                  transform = None, image_size =224, noiseType = None, noiseParam = None,
                  injectedNoise = None,return_noise_label = False,injectedLink = None,
                  isVideo = False,giveCroppedFace = False,lndmarkNumber = 68):
        
        self.lndmarkNumber = lndmarkNumber
        self.return_noise_label = return_noise_label
        self.giveCroppedFace = giveCroppedFace
        
        self.injectedNoise = injectedNoise
        self.noiseType  = noiseType
        self.noiseParam =noiseParam
        
        self.transform = transform
        self.onlyFace = onlyFace
        
        self.imageHeight = image_size
        self.imageWidth = image_size
        self.injectedLink = injectedLink
        
        
        list_gt = []
        list_labels_t = []
        
        counter_image = 0
        annot_name = 'annot'
        
        
        is_video = isVideo
        
        if dir_gt is not None : 
            annot_name = dir_gt
            
        for data in data_list : 
            print(("Opening "+data))
            
            if self.injectedLink is not None : 
                the_direction = self.injectedLink
            else :
                the_direction = curDir + "images/"+data+"/"
            
            for f in file_walker.walk(the_direction):
                if f.isDirectory: # Check if object is directory
                    print((f.name, f.full_path)) # Name is without extension
                    
                    if is_video : 
                        
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
                                    
                    else : 
                        list_dta = []
                        for sub_f in f.walk(): #this is the data
                            print('subf  ',sub_f.full_path)
                            if(".npy" not in sub_f.full_path):
                                list_dta.append(sub_f.full_path)
                                
                        if(f.name == annot_name) : #If that's annot, add to labels_t 
                            list_labels_t.append(sorted(list_dta))
                        elif(f.name == 'img'): #Else it is the image
                            list_gt.append(sorted(list_dta))
                            counter_image+=len(list_dta)
        
        self.length = counter_image 
        print("Now opening keylabels")
        
        list_labels = []     
        for lbl in list_labels_t :
            lbl_68 = [] #Per folder
            for lbl_sub in lbl :
                print(lbl_sub)
                if ('pts' in lbl_sub) : 
                    x = []
                    with open(lbl_sub) as file:
                        data2 = [re.split(r'\t+',l.strip()) for l in file]
                    for i in range(len(data2)) :
                        if(i not in [0,1,2,len(data2)-1]):
                            x.append([ float(j) for j in data2[i][0].split()] )
                    lbl_68.append(np.array(x).flatten('F')) #1 record
            list_labels.append(lbl_68)
            
        list_images = []
        list_ground_truth = np.zeros([counter_image,136])
        
        #Flatten it to one list
        indexer = 0
        for i in range(0,len(list_gt)): #For each dataset
            for j in range(0,len(list_gt[i]),step): #for number of data #n_skip is usefull for video data
                list_images.append(list_gt[i][j])
                
                list_ground_truth[indexer] = np.array(list_labels[i][j]).flatten('F')
                indexer += 1
                
        self.l_imgs = list_images
        self.l_gt = list_ground_truth
        
        print(list_images)
        print(list_ground_truth)
        
        #print(self.l_imgs,self.l_gt)

    def __getitem__(self,index):
        
        #Read all data, transform etc.
        
        #In video, the output will be : [batch_size, sequence_size, channel, width, height] 
        #Im image : [batch_size, channel, width, height]
         
        x,label  = self.l_imgs[index],self.l_gt[index].copy()
        #print(x,label)
        
        #print(label)
        
        tImage = Image.open(x).convert("RGB")
        tImageB = None
        
        if self.onlyFace :    
            #crop the face region
            t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 2,div_y = 2,images = cv2.imread(x),n_points = self.lndmarkNumber)
            area = (x1,y1, x2,y2)
            tImage =  tImage.crop(area)
            tImage = tImage.resize((self.imageWidth,self.imageWidth))
            
            label[:self.lndmarkNumber] -= x_min
            label[self.lndmarkNumber:] -= y_min
            
            label[:self.lndmarkNumber] *= self.imageWidth/(x2 - x1)
            label[self.lndmarkNumber:] *= self.imageHeight/(y2 - y1)
        
        if self.giveCroppedFace : 
            tImageCr = Image.open(x).convert("RGB")
            t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 =  get_enlarged_bb(the_kp = label,div_x = 2,div_y = 2,images = cv2.imread(x),n_points = self.lndmarkNumber)
            area = (x1,y1, x2,y2)
            #print('the arae',area)
            tImageCr =  tImageCr.crop(area)
            tImageCr = tImageCr.resize((self.imageWidth,self.imageWidth))
            
            if self.transform is not None : 
                tImageCr = self.transform(tImageCr)
            
        if self.noiseType is not None :
            tImageB = tImage.copy()
            
            #print(tImageB.size)
            
            if self.injectedNoise is None : 
                
                noiseType = np.random.randint(0,5)
                noiseLevel = np.random.randint(0,self.noiseParam)
                noiseParam=noiseParamList[noiseType,noiseLevel]
                
            else : 
                noiseType =self.injectedNoise[0]
                if self.injectedNoise[1] < 0 : 
                    noiseParam = noiseParamListTrain[self.injectedNoise[0],np.random.randint(0,self.noiseParam)]
                else : 
                    noiseParam=noiseParamList[self.injectedNoise[0],self.injectedNoise[1]]
            
            #print(noiseType,noiseParam)
            '''if self.noiseType == 1: #downsample
                for i in range(int(self.noiseParam/2)) :#Scale down (/2) blurLevel times 
                    width, height = tImageB.size
                    tImageB = tImageB.resize((width//2,height//2))
                    #print(tImageB.size)
            elif self.noiseType == 2 : #Gaussian blur
                tImageB = tImageB.filter(ImageFilter.GaussianBlur(self.noiseParam))
            elif self.noiseType == 3 : #Gaussian noise 
                #tImageB = addNoise(tImageB)
                #convert to opencv 
                opencvImage = cv2.cvtColor(np.array(tImageB), cv2.COLOR_RGB2BGR)
                
                #print(opencvImage)
                opencvImage = addNoise(opencvImage)
                pilImage =  cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)
                #tImageB = Image.fromarray(random_noise(opencvImage))
                tImageB = Image.fromarray(pilImage)'''
            
            tImageB = generalNoise(tImageB,noiseType,noiseParam)
            
            if self.transform is not None:
                tImageB = self.transform(tImageB)
        
        if self.transform is not None:
            tImage = self.transform(tImage)
        
        if self.noiseType : 
            if self.return_noise_label : 
                return tImage,tImageB,torch.FloatTensor(label),noiseType
            else  : 
                return tImage,tImageB,torch.FloatTensor(label),x
        else : 
            if not self.giveCroppedFace : 
                return tImage,torch.FloatTensor(label),x
            else : 
                return tImage,torch.FloatTensor(label),tImageCr,x
    
    def __len__(self):
        
        return len(self.l_imgs)
    
    
    
    
    
    
    
    
def convertName(input):
    number = int(re.search(r'\d+', input).group())
    if 'train' in input : 
        return number
    elif 'dev' in input :
        return 10+number
    elif  'test' in input :
        return 20+number

def main():
    
    '''data_list = ['r-temp']
    
    curDir = "/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/"
        
    list_gtA = {} #[batch, sequences, [val,arousal]]
    list_gtV = {}
    list_m_images = {} #[batch, sequences, [b,g,r]]
    list_m_physio = {} #[batch, sequences, [ECG,EDA]]
    list_m_audios = {} #[batch, sequences, [audio]]
    list_m_ldmrks = {} #[batch, sequences, [audio]]
    list_syn = {} #[batch, sequences, [time]]
    
    counter_image = 0
    
    dirImageName = 'm-images'
    dirPhysioName = 'm-physio'
    dirAudioName = 'm-audios'
    dirSynName = 'syn-im'
    dirLdmrkNames = 'ldmrks'
    
    dir_gt = 'l_gs'
    dur_gt_ind = 'l_ind'
    
    listKey = []
     
    for data in data_list : 
        print(("Opening "+data))
        for f in file_walker.walk(curDir +data+"/"):
            if f.isDirectory: # Check if object is directory
                print((f.name, f.full_path)) # Name is without extension
                if(f.name == dirImageName) : #If thats image folder
                    for sub_f in f.walk(): #this is the data
                        if sub_f.isDirectory: 
                            list_dta = []
                            for sub_sub_f in sub_f.walk() : 
                                if(".npy" not in sub_sub_f.full_path):
                                    list_dta.append(sub_sub_f.full_path)
                                    counter_image+=len(list_dta)
                        list_m_images[sub_f.name] =  sorted(list_dta)
                        listKey.append(sub_f.name)
                
                
                if(f.name == dirLdmrkNames) : #If thats image folder
                    for sub_f in f.walk(): #this is the data
                        if sub_f.isDirectory: 
                            list_dta = []
                            for sub_sub_f in sub_f.walk() : 
                                if(".npy" not in sub_sub_f.full_path):
                                    list_dta.append(sub_sub_f.full_path)
                                    counter_image+=len(list_dta)
                        list_m_ldmrks[sub_f.name] =  sorted(list_dta)
                
                elif(f.name == dirSynName) : #SynImage
                    for sub_f in f.walk(): #this is the data    
                        print(sub_f.full_path)  
                        if '.csv' in sub_f.full_path :     
                            list_dta = []
                            with open(sub_f.full_path, 'r') as csvFile:
                                reader = csv.reader(csvFile,delimiter=';')
                                for row in reader:
                                    if 'frame' in row : 
                                        pass 
                                    else : 
                                        list_dta.append([int(row[0]),float(row[1])])
                            list_syn[sub_f.name] =  sorted(list_dta)
                            
                        
                elif(f.name == dirPhysioName) : #Physio
                    for sub_f in f.walk(): #this is the data    
                        print(sub_f.full_path)  
                        if '.csv' in sub_f.full_path : 
                            list_dta = []    
                            with open(sub_f.full_path, 'r') as csvFile:
                                reader = csv.reader(csvFile,delimiter=';')
                                for row in reader:
                                    if 'time' in row : 
                                        pass 
                                    else : 
                                        list_dta.append([float(row[0]),float(row[1]),float(row[4])])
                            list_m_physio[sub_f.name] =  sorted(list_dta)
                            
                elif(f.name == dir_gt) : #gt
                    for sub_f in f.walk(): 
                        print(sub_f.full_path)
                        for sub_sub_f in sub_f.walk(): #this is the data    
                            if '.arff' in sub_sub_f.full_path :
                                data = []
                                with open(sub_sub_f.full_path) as f: 
                                    for line in f: 
                                        l = line.rstrip('\r\n').split(',')
                                        if ('v_' in l[0]) or ('n_' in l[0]) : #dev_x, train_x,
                                            #print('in',l[0])
                                            data.append(np.asarray([float(l[1]),float(l[2])]))
                            #print(len(sorted(data)),sub_sub_f.name)
                            if sub_f.name == "valence" :
                                list_gtV[sub_sub_f.name] = data
                            else :
                                list_gtA[sub_sub_f.name] = data
                                
    print("Now opening keylabels")
    print(len(list_m_images))
    #print(list_m_images.keys())
    print(len(list_syn))
    #print(list_syn[0])
    #print(list_m_images[0])
    #print(list_m_physio[0])
    print(counter_image)
    
    #print(len(list_gtA))
    #print(list_gtA.keys())
    
    #exit(0)
    t_listGT = []
    t_listMeta = []
    #1,2,3 = train 1,2,3
    #11,12,13 = dev 1,2,3
    #21,22,23 = test 1,2,3 
    
    t_listImg = []
    t_listLdmrk = []
    t_listPhy = []
    
    #Now aligning them according to the groundtruth  to one long list 
    for key in listKey : 
        print(key)
        
        #first get the val and arousal with the cor time
        listA = list_gtA[key]
        listV = list_gtV[key]
        
        #First ensure the time are the same of V and A\
        #print('gt : ',len(listA),len(listV))
        assert(len(listA[0]) == len(listV[0])) 
        #print('ok')
        
        
        l_data=[]
        #Now merge the V and A as one long list
        for x,y in zip(listA,listV): 
            t_listGT.append([x[1],y[1]])
            t_listMeta.append([x[0],y[0],convertName(key)])
            l_data.append([x[0],y[0],convertName(key)])
        #print(t_listGT[0])
        #Now the actual data 
        #the image
        t_listIm = list_m_images[key]
        t_listSyn = list_syn[key]
        t_listLm = list_m_ldmrks[key]
        
        #print(listIm)
        
        #check the length is the same
        #print('img : ',len(t_listIm),len(t_listSyn))
        assert(len(t_listIm) == len(t_listSyn))
        
        #print('syn',listSyn[5][1])
        
        #Now get the data according the timeframe, if missing will attempt to replicate the data to fill in
        #print('l t',len(l_data))
        i = 0
        j = 0
        for timeFrame in l_data : 
            t = timeFrame[0]
            #print('d',i,len(t_listSyn),j,t,key,len(l_data))
            
            if ((j == len(l_data)-1) and i == len(t_listSyn)):  #anomaly
                t_listImg.append(t_listIm[i-1])
                t_listLdmrk.append(t_listLm[i-1])
                
            elif t_listSyn[i][1] == t:
                #print('equal',i,j,t,key,len(l_data)) 
                t_listImg.append(t_listIm[i])
                t_listLdmrk.append(t_listLm[i])
                i += 1
            else : #missing, fill with the 
                #print('missing',i,key)
                t_listImg.append(t_listIm[i-1])
                t_listLdmrk.append(t_listLm[i-1])
            j+=1
        #Now the physio, it has 10 data/second
        dps = 10
        listP = list_m_physio[key]
        #print(len(listP))
        
        meanECG = 90
        meanEDA = 0
        
        i = 0
        j = 0
        for timeFrame in l_data : 
            t = timeFrame[0]
              
            if listP[i][0] == t:
                #print('equal',i,j,t,key,len(listP),len(l_data))
                temp = []
                for k in range(-5,5) : #get 5 temporal windows
                    idx = k+i
                    if idx < 0 or (idx > len(listP)): 
                        temp.append([meanECG, meanEDA])
                    else :   
                        temp.append([listP[idx][1],listP[idx][2]])
                    i += 1 
                t_listPhy.append(temp)
            j+=1
        
    
    
    print(len(t_listGT))
    print(len(t_listImg))
    print(len(t_listPhy))
    print(len(t_listMeta))
    print(len(t_listLdmrk))
    exit(0)'''
    
    
    '''
    image_size = 224
    batch_size = 1
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])
    
    
    data = Recola(["r-temp"], True, image_size, transform, True, True, 1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True)
    dataloader = torch.utils.data.DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    
    # Plot some training images
    for real_batch,phy,gt,mva,aud in (dataloader) :
        #print(real_batch.shape, gt.shape)
        img = unnormalizedToCV(real_batch, customNormalize = np.array([91.4953, 103.8827, 131.0912])  )
        print(phy.shape)
        print(gt.shape)
        #print(mva.shape)
        print(aud[0].shape)
        for ig in img : 
            cv2.imshow('t',ig)
            cv2.waitKey(0)
    '''
    
    '''
    from data_loader import get_loader
    
    celeba_image_dir='data/celeba/images'
    attr_path='data/celeba/list_attr_celeba.txt'
    rafd_image_dir='data/RaFD/train'
    log_dir='stargan/logs'
    model_save_dir='stargan/models'
    sample_dir='stargan/samples'
    result_dir='stargan/results'
    dataset='CelebA' #, choices=['CelebA', 'RaFD', 'Both'])-
    selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'] 
    celeba_crop_size=178
    image_size = 128
    batch_size = 1
    mode = 'train'
    num_workers = 1
    
    celeba_loader = None
    if dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(celeba_image_dir, attr_path, selected_attrs,celeba_crop_size, image_size, batch_size,'CelebA', mode, num_workers)
    
    data_iter = iter(celeba_loader)
    x_fixed, c_org = next(data_iter)
    
    print(x_fixed.shape,c_org.shape)
    
    print(x_fixed[0].shape,x_fixed[0])
    
        
    plt.figure(figsize=(batch_size,batch_size))
    plt.axis("off")
    plt.title("Trainings Images")
    plt.imshow(np.transpose(vutils.make_grid(x_fixed[0], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    #exit(0)
    
    
    
    
    
    
    
    image_size = 224
    batch_size = 1
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])
    
    #AFEW-VA-PP
    
    isVideo = False
    
    #"SEM-temp"
    #data = AFEWVA(["temp"], None, True, image_size, transform, True, False, 1,split=True, nSplit = 5,listSplit=[4],wHeatmap=True)
    #data = AFEWVA(["temp"], None, True, image_size, transform, True, True, 1,split=True, nSplit = 5,listSplit=[4],wHeatmap=True)#,isVideo=True, seqLength = 7)
    data = AFEWVA(data_list = ["AFEW-VA-Fixed"], dir_gt = None, onlyFace = True, image_size = image_size, 
                  transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=[2],
                  wHeatmap=True,isVideo=isVideo, seqLength = 6,toAlign = False,returnM = True)
    
    #AFEW-VA-Fixed #SEM-temp
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True)
    dataloader = torch.utils.data.DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    print(len(dataloader))
    # Plot some training images
    for real_batch,va,gt,htmp,M in (dataloader) :
        print(real_batch.shape)
        print(htmp[0].shape)
        pass
        
        print(real_batch[0].shape,real_batch[0])
        
        plt.figure(figsize=(batch_size,batch_size))
        plt.axis("off")
        plt.title("Trainings Images")
        #plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device), padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
        
        print(real_batch.shape, va.shape, gt.shape,htmp.shape)
        print(M,M.shape)
        
        if isVideo :
            rb = real_batch[0]
            rgt = gt[0]
            rva = va[0]
        else : 
            rb = real_batch
            rgt = gt
            rva = va
             
        img = unnormalizedAndLandmark(rb, inputPred = rgt,inputGT = None, customNormalize = np.array([91.4953, 103.8827, 131.0912]))
        #print(va,x,lbl)
        for ig,vva in zip(img,rva) :
            print(vva) 
            cv2.imshow('t',ig)
            cv2.waitKey(0)
    
    
    exit(0)
    '''
    
    
    image_size = 112
    batch_size = 1
    
    transform = transforms.Compose([
        #transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])
    
    
    #AFEW-VA-PP
    
    isVideo = True
    
    #"SEM-temp"
    #data = AFEWVA(["temp"], None, True, image_size, transform, True, False, 1,split=True, nSplit = 5,listSplit=[4],wHeatmap=True)
    #data = AFEWVA(["temp"], None, True, image_size, transform, True, True, 1,split=True, nSplit = 5,listSplit=[4],wHeatmap=True)#,isVideo=True, seqLength = 7)
    '''data = AFEWVAReduced(data_list = ["AFEW-VA-Fixed"], dir_gt = None, onlyFace = True, image_size = image_size, 
                  transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=[2],
                  wHeatmap=True,isVideo=isVideo, seqLength = 6,toAlign = False,returnM = True)'''
    
    #SEWA-small
    data = AFFChallenge(data_list = ["AffectChallenge"],listMode = ['tmp'],onlyFace = True, image_size =112, 
                 transform = transform,useIT = False,augment = False, step = 1,isVideo = False, seqLength = None, dbType = 0,
                 returnQuadrant = False, returnNoisy = False, returnWeight = True, returnSound = True)
    
    #AFEW-VA-Fixed #SEM-temp
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True)
    dataloader = torch.utils.data.DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    print(len(dataloader))
    # Plot some training images
    #lImgs, lVA, lLD, Mt,l_nc
    
    for lImgs,va,gt,M,ln,w,s in (dataloader) :
        #print(real_batch.shape)
        
        #print(real_batch[0].shape,real_batch[0])
        #print(ln,va)
        #print(ln,va,q)
        #print(weight,weight.shape)
        plt.figure(figsize=(batch_size,batch_size))
        plt.axis("off")
        plt.title("Trainings Images")
        #plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1,2,0)))
        #plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device), padding=2, normalize=True).cpu(),(1,2,0)))
        
        print(va.shape,w.shape,s.shape)
        print(va,w,s)
        plt.imshow(np.transpose(vutils.make_grid(lImgs[0].to(device), padding=2, normalize=True).cpu(),(1,2,0)))
        
        plt.show()
        
    
    exit(0)
    
    ############################
    
    
    
            
    image_size = 224
    batch_size = 1
    
    transform = transforms.Compose([
        #transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])
    
    
    #AFEW-VA-PP
    
    isVideo = True
    
    #"SEM-temp"
    #data = AFEWVA(["temp"], None, True, image_size, transform, True, False, 1,split=True, nSplit = 5,listSplit=[4],wHeatmap=True)
    #data = AFEWVA(["temp"], None, True, image_size, transform, True, True, 1,split=True, nSplit = 5,listSplit=[4],wHeatmap=True)#,isVideo=True, seqLength = 7)
    '''data = AFEWVAReduced(data_list = ["AFEW-VA-Fixed"], dir_gt = None, onlyFace = True, image_size = image_size, 
                  transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=[2],
                  wHeatmap=True,isVideo=isVideo, seqLength = 6,toAlign = False,returnM = True)'''
    
    #SEWA-small
    data = SEWAFEWReduced(data_list = ["AFEW-VA-Small"], dir_gt = None, onlyFace = True, image_size = image_size, 
                  transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=[2],
                  isVideo=isVideo, seqLength = 6,dbType = 0,returnQuadrant=True,returnNoisy = True,returnWeight=True)
    
    #AFEW-VA-Fixed #SEM-temp
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True)
    dataloader = torch.utils.data.DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    print(len(dataloader))
    # Plot some training images
    for real_batch,va,gt,M,ln,q,noisy_batch,weight in (dataloader) :
        #print(real_batch.shape)
        
        #print(real_batch[0].shape,real_batch[0])
        #print(ln,va)
        print(ln,va,q)
        print(weight,weight.shape)
        plt.figure(figsize=(batch_size,batch_size))
        plt.axis("off")
        plt.title("Trainings Images")
        #plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1,2,0)))
        #plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device), padding=2, normalize=True).cpu(),(1,2,0)))
        plt.imshow(np.transpose(vutils.make_grid(noisy_batch[0].to(device), padding=2, normalize=True).cpu(),(1,2,0)))
        
        plt.show()
        
        #print(real_batch.shape, va.shape, gt.shape)
        #print(noisy_batch.shape, va.shape, gt.shape)
        
        continue
        
        if isVideo :
            #rb = real_batch[0]
            rb = noisy_batch[0]
            rgt = gt[0]
            rva = va[0]
        else : 
            #rb = real_batch
            rb = noisy_batch
            rgt = gt
            rva = va
             
        img = unnormalizedAndLandmark(rb, inputPred = rgt,inputGT = None, customNormalize = np.array([91.4953, 103.8827, 131.0912]))
        #print(va,x,lbl)
        for ig,vva in zip(img,rva) :
            print(vva) 
            cv2.imshow('t',ig)
            cv2.waitKey(0)
    
    
    exit(0)
    
    
    
    
    
    
    
    image_size = 224
    batch_size = 1
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])
    
    #AFEW-VA-PP
    
    isVideo = False
    
    #"SEM-temp"
    #data = AFEWVA(["temp"], None, True, image_size, transform, True, False, 1,split=True, nSplit = 5,listSplit=[4],wHeatmap=True)
    #data = AFEWVA(["temp"], None, True, image_size, transform, True, True, 1,split=True, nSplit = 5,listSplit=[4],wHeatmap=True)#,isVideo=True, seqLength = 7)
    data = SEWAFEW(["SEWA-small"], None, True, image_size, transform, True, True, 1,split=True, nSplit = 5,listSplit=[2],
                  wHeatmap=True,isVideo=isVideo, seqLength = 6,toAlign = False,returnM = True,dbType=1)
    
    #AFEW-VA-Fixed #SEM-temp
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True)
    dataloader = torch.utils.data.DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    print(len(dataloader))
    # Plot some training images
    for real_batch,va,gt,htmp,M in (dataloader) :
        print(real_batch.shape)
        print(htmp[0].shape)
        pass
        
        plt.figure(figsize=(batch_size,batch_size))
        plt.axis("off")
        plt.title("Trainings Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
        
        print(real_batch.shape, va.shape, gt.shape,htmp.shape)
        print(M,M.shape)
        
        if isVideo :
            rb = real_batch[0]
            rgt = gt[0]
            rva = va[0]
        else : 
            rb = real_batch
            rgt = gt
            rva = va
             
        img = unnormalizedAndLandmark(rb, inputPred = rgt,inputGT = None, customNormalize = np.array([91.4953, 103.8827, 131.0912]),ldmarkNumber =49)
        #print(va,x,lbl)
        for ig,vva in zip(img,rva) :
            print(vva) 
            cv2.imshow('t',ig)
            cv2.waitKey(0)
    
    
    
    exit(0)
        
        
        
        
        
        
    
    image_size = 224
    batch_size = 1
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])
    
    data = AVChallenge(["AVC-Train"], dir_gt = None, onlyFace = True, image_size = image_size, 
                       transform = transform, useIT = True, augment = False,isTest = False,wHeatmap=True,
                       split=True,listSplit=[0,1,2,3])#,isVideo=True, seqLength = 7)
    
    '''data = AVChallenge(["AVC-Test"], dir_gt = None, onlyFace = True, image_size = image_size, 
                       transform = transform, useIT = True, augment = False,isTest = True,wHeatmap=True)#,isVideo=True, seqLength = 7)'''
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True)
    dataloader = torch.utils.data.DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    print(len(dataloader))
    # Plot some training images
    for real_batch,va,gt,htmp in (dataloader) :
        
        '''print(htmp[0].shape)
        pass
    
        plt.figure(figsize=(batch_size,batch_size))
        plt.axis("off")
        plt.title("Trainings Images")
        plt.imshow(np.transpose(vutils.make_grid(htmp[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()'''
        
        print(real_batch.shape, va.shape, gt.shape,htmp.shape)
        print(va)
        img = unnormalizedAndLandmark(real_batch, inputPred = gt,inputGT = None, customNormalize = np.array([91.4953, 103.8827, 131.0912]))
        #print(va,x,lbl)
        for ig in img : 
            cv2.imshow('t',ig)
            cv2.waitKey(0)

def cropImage():
    
    batch_size = 20
    image_size = 224
    isVideo = False
    doConversion = False
    lndmrkNumber =49
    
    isSewa = True
    
    ratio = truediv(128,224)
    if ratio : 
        displaySize = str(128)
    else : 
        displaySize = str(image_size)
    
    err_denoised = curDir+"de-label-"+'sewa'+".txt"
    checkDirMake(os.path.dirname(err_denoised))
    print('file of denoising : ',err_denoised)
    fileOfDen = open(err_denoised,'w')
    fileOfDen.close()
    
    #theDataSet = "AFEW-VA-Small"
    #theDataSet = "AFEW-VA-Fixed"
    #theDataSet = "SEWA-small"
    theDataSet = "SEWA"
    
    
    oriDir = '/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/'+theDataSet
    #oriDir = '/media/deckyal/INT-2TB/comparisons/'+theDataSet + "/" + str(theNoiseType)+"-"+str(theNoiseParam)+'/'
    
    targetDir = '/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/'+theDataSet+'-ext'
    checkDirMake(targetDir)
 
    data_transform   = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ID = ImageDataset(data_list = [theDataSet],onlyFace=True,transform=data_transform,image_size=image_size
                            ,injectedLink = oriDir,isVideo = isVideo,giveCroppedFace=True,
                            annotName='annotOri',lndmarkNumber=lndmrkNumber,isSewa = isSewa)
    
    dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = False)
    
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    notNeutral = 0
    list_nn = []
    list_name_nn = []
    
    print('inside',len(dataloader))
    
    
    
    GD = GeneralDAEX(nClass = 3)

    dircl1 = '/home/deckyal/eclipse-workspace/FaceTracking/src/toBeUsedT-5Aug/'+'Mix3-combineAE.pt'
    dircl2 = '/home/deckyal/eclipse-workspace/FaceTracking/src/toBeUsedT-5Aug/'+'Mix3-combineCL.pt'
    outDir = "mix3-"
    model_lg = LogisticRegression(512, 3)
        
    
    netAEC = DAEE()
    netAEC.load_state_dict(torch.load(dircl1))
    netAEC = netAEC.cuda()
    netAEC.eval()
    
    #theDataSetOut = theDataVideo+outDir
    model_lg.load_state_dict(torch.load(dircl2))
    model_lg = model_lg.cuda()
    model_lg.eval()
    
    #print(netAEC.fce.weight)
    print(model_lg.linear2.weight)
    #exit(0)
    isVideo = False
    #exit(0)
    data_transform   = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        #transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
    ])

    
    # Plot some training images
    for real_batch,gt,cr,x,gtcr,gtcr2 in dataloader : # = next(iter(dataloader))
        print(real_batch.size())
        for imgs,gts,imgcrs,fileName,gtcrs,gts2 in zip(real_batch.cuda(),gt.numpy(),cr.cuda(),x,gtcr.numpy(),gtcr2.numpy()): 
            
            print(fileName)
            
            #first save the original image 
            
            #now getting the name and file path 
            filePath = fileName.split(os.sep)
            annotPath = copy.copy(filePath)
            if isSewa : 
                annotPathSewa = copy.copy(filePath)
                
            filePathCleaned = copy.copy(filePath)
            
            filePath[-2]+='-'+displaySize
            filePathCleaned[-2]+='-'+displaySize+'-C'
            if isSewa : 
                annotPath[-2]='annotOri-'+displaySize
                annotPathSewa[-2]='annot-'+displaySize
            else : 
                annotPath[-2]='annot-'+displaySize
            
            newFilePath = '/'.join(filePath[:-1])
            newAnnotPath = '/'.join(annotPath[:-1])
            newAnnotPathSewa = '/'.join(annotPathSewa[:-1])
            newClFilePath = '/'.join(filePathCleaned[:-1])
            #print(filePath,annotPath)
            print(newFilePath, newAnnotPath)
            #ifolder = filePath.index(theDataVideo)
            
            image_name = filePath[-1]
            annot_name = os.path.splitext(image_name)[0]+'.pts'
            
            '''if isVideo :  
                middle = filePath[ifolder+2:-2]
                print(middle)
                middle = '/'.join(middle)
                
                finalTargetPathI = targetDir+middle+'/img/'
                finalTargetPathA = targetDir+middle+'/annot/'
            else : 
                finalTargetPathI = targetDir+'img/'
                finalTargetPathA = targetDir+'annot/' '''
                
            checkDirMake(newFilePath)
            checkDirMake(newAnnotPath)
            checkDirMake(newAnnotPathSewa)
            checkDirMake(newClFilePath)
            
            finalTargetImage = newFilePath+'/'+image_name
            finalTargetImageCl = newClFilePath+'/'+image_name
            finalTargetAnnot = newAnnotPath+'/'+annot_name
            finalTargetAnnotSewa = newAnnotPathSewa+'/'+annot_name
            
            
            theOri = unorm(imgcrs.detach().cpu()).numpy()*255
            theOri = cv2.cvtColor(theOri.transpose((1,2,0)).astype(np.uint8 ),cv2.COLOR_RGB2BGR)
            
            if ratio :
                theOri = cv2.resize(theOri,(128,128))
            cv2.imwrite(finalTargetImage,theOri)
            
            if ratio : 
                gtcrs[:lndmrkNumber] *= ratio
                gtcrs[lndmrkNumber:] *= ratio
                
                if isSewa : 
                    gts2[:68] *= ratio
                    gts2[68:] *= ratio
            
            write_kp_file(finalTargetAnnot,gtcrs,length = lndmrkNumber)
            
            if isSewa : 
                write_kp_file(finalTargetAnnotSewa,gts2,length = 68)
            
            
            #print(gtcrs)
            
            #Now see the result back
            
            r_image = cv2.imread(finalTargetImage)
            
            print(finalTargetAnnot)
            predicted = utils.read_kp_file(finalTargetAnnot, True)
            for z22 in range(lndmrkNumber) :
                #print(z22)
                cv2.circle(r_image,(int(predicted[z22]),int(predicted[z22+lndmrkNumber])),2,(0,255,0))
                
            if isSewa:
                predicted2 = utils.read_kp_file(finalTargetAnnotSewa, True)
                for z22 in range(68) :
                    cv2.circle(r_image,(int(predicted2[z22]),int(predicted2[z22+68])),2,(255,255,255))
                    
            cv2.imshow('test',r_image)
            cv2.waitKey(1)
            
            #exit(0)
            
            #second get the cleaned one 
            
            #if cl_type == 1 : 
            recon_batch,xe = netAEC(imgs.unsqueeze(0))
            #else :  
            #    xe = netAEC(imgs.unsqueeze(0))
                
            labels = model_lg(xe)
            x, y = torch.max(labels, 1)
            
            ll = y.cpu()[0]
            
            print('res',ll)
            
            #res = GD.forward(imgs.unsqueeze(0), y[0])[0].detach().cpu()
            res = GD.forward(imgcrs.unsqueeze(0), y[0])[0].detach().cpu()
            
            theRest = unorm(res).numpy()*255
            print(theRest.shape)
            theRest = cv2.cvtColor(theRest.transpose((1,2,0)).astype(np.uint8 ),cv2.COLOR_RGB2BGR)
            
            
            if ratio : 
                theRest = cv2.resize(theRest,(128,128))
            
            
            theOri = unorm(imgs.detach().cpu()).numpy()*255
            print(theOri.shape)
            theOri = cv2.cvtColor(theOri.transpose((1,2,0)).astype(np.uint8 ),cv2.COLOR_RGB2BGR)
            
            cv2.imshow('theori',theRest)
            cv2.waitKey(1)
            
            
            cv2.imwrite(finalTargetImageCl,theRest)
            #third save the cleaned one
            
            #exit(0)
            
            '''
            #print(theRest.shape)
            
            theImage = theRest
            
            #now getting the name and file path 
            filePath = fileName.split(os.sep)
            ifolder = filePath.index(theDataVideo)
            
            image_name = filePath[-1]
            annot_name = os.path.splitext(image_name)[0]+'.pts'
            
            if isVideo :  
                middle = filePath[ifolder+2:-2]
                print(middle)
                middle = '/'.join(middle)
                
                finalTargetPathI = targetDir+middle+'/img/'
                finalTargetPathA = targetDir+middle+'/annot/'
            else : 
                finalTargetPathI = targetDir+'img/'
                finalTargetPathA = targetDir+'annot/'
                
            checkDirMake(finalTargetPathI)
            checkDirMake(finalTargetPathA)
            
            finalTargetImage = finalTargetPathI+image_name
            finalTargetAnnot = finalTargetPathA+annot_name
            
            print(finalTargetImage,finalTargetAnnot)'''
            
            
            if ll != 0 or True: 
                if ll != 0:
                    notNeutral+=1
                    list_nn.append(ll)
                    list_name_nn.append(finalTargetImage)
            
            
                fileOfDen = open(err_denoised,'a')
                fileOfDen.write(str(int(ll))+','+finalTargetImage+"\n")
                fileOfDen.close()
                
                print('status : ',ll)
                '''
                cv2.imshow('ori',theOri)
                cv2.waitKey(0)
                cv2.imshow('after',theRest)
                cv2.waitKey(0)'''
                
            print(y,labels)
            
    
    print("not neutral count : ",notNeutral)            



def getDistributionAC():
        
    import matplotlib.pyplot as plt 
        
    targetDir = '/home/deckyal/eclipse-workspace/FaceTracking/FaceTracking-NR/StarGAN_Collections/stargan-master/distribution/'
        
    
    tname = "AC"
    
            
    image_size = 112
    batch_size = 20000
    
    transform = transforms.Compose([
        #transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])
    
    if False :
         
        a = np.array(range(20))
        v = np.array(range(20))
        
        tx = np.array(range(20))
        
        for i in range(5) : 
            
            z = np.load(targetDir+'VA-Train-'+str(i)+'.npy')
            la = z[:,0]
            lv = z[:,1]
            
            #print(la,la.shape)
            
            a+=la
            v+=lv 
            
            z = np.load(targetDir+'VA-Test-'+str(i)+'.npy')
            la = z[:,0]
            lv = z[:,1]
            
            #print(la,la.shape)
            
            a+=la
            v+=lv
            
            
        
        fig = plt.figure()
    
        ax = plt.subplot(2, 2, 1)
        ax.bar(tx,a)
        ax.set_title('a')
        
        ax = plt.subplot(2, 2, 2)
        ax.bar(tx,v)
        ax.set_title('v')
        
        plt.show()
        
        
        
        #print(a)
        #print(v)
        exit(0)
    
    
    
    ID = AFFChallenge(data_list = ["AffectChallenge"],mode = 'Train',onlyFace = True, image_size =112, 
                 transform = transform,useIT = False,augment = False, step = 1,isVideo = False, seqLength = None, dbType = 0,
                 returnQuadrant = False, returnNoisy = False, returnWeight = False)
    
    VD = AFFChallenge(data_list = ["AffectChallenge"],mode = 'Val',onlyFace = True, image_size =112, 
                 transform = transform,useIT = False,augment = False, step = 1,isVideo = False, seqLength = None, dbType = 0,
                 returnQuadrant = False, returnNoisy = False, returnWeight = False)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = torch.utils.data.DataLoader(dataset = data, batch_size = batch_size, shuffle = True)
    
    
    dataloaderTrn = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = False)
    dataloaderVal = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = False)

    
    listV = np.array(range(0,21))
    listA = np.array(range(0,21))
    
    listVx = np.array(range(0,21))
    listAx = np.array(range(0,21))
    
    listVt = np.array(range(0,21))
    listAt = np.array(range(0,21))
        
    #for real_batch,vas,gt,M,_ in (dataloaderTrn) :
    x = 0 
    for lImgs,vas,gt,M,ex in (dataloaderTrn) :
        
        for va in vas : 
            print(x,len(dataloaderTrn)*batch_size)
            #print(ex,gt,M)
            #print(va,vas)
            lva = (va.cpu().numpy()) * 10+10
            name = 'AC-Train'
            print(va)
            print(lva)
            listV[int(round(lva[0]))]+=1
            listA[int(round(lva[1]))]+=1
        
            x+=1
    
    x = 0
    
    print(listV,listA)
    np.save(targetDir+name+'.npy',np.column_stack((listV,listA)))
    
    for real_batch,vas,gt,M,ex in (dataloaderVal) :
        for va in vas : 
            print(x,len(dataloaderVal)*batch_size) 
            lva = (va.cpu().numpy()) * 10+10
            name = 'AC-Test-'
            listVt[int(round(lva[0]))]+=1
            listAt[int(round(lva[1]))]+=1
            x+=1
        
    print(listVt,listAt)
    np.save(targetDir+name+'.npy',np.column_stack((listVt,listAt)))
    
    
    '''fig, ax = plt.subplots(nrows=1, ncols=2)
    
    for row in ax:
        for col in row:
            col.plot(x, y)'''
    
    fig = plt.figure()

    ax = plt.subplot(2, 2, 1)
    ax.bar(listVx,listV)
    ax.set_title('v train')
    
    ax = plt.subplot(2, 2, 2)
    ax.bar(listAx,listA)
    ax.set_title('A train')
    
    ax = plt.subplot(2, 2, 3)
    ax.bar(listVx,listVt)
    ax.set_title('v test')
    
    ax = plt.subplot(2, 2, 4)
    ax.bar(listAx,listAt)
    ax.set_title('A test')
    
    #plt.show()
    plt.savefig(tname+".png")
        
    exit(0)
    
    

def getDistribution():
        
    import matplotlib.pyplot as plt 
        
    targetDir = '/home/deckyal/eclipse-workspace/FaceTracking/FaceTracking-NR/StarGAN_Collections/stargan-master/distribution/'
        
    
    isAFEW = True
    
    name = "AFEW"
    if not isAFEW : 
        name = "SEWA"
    
            
    image_size = 224
    batch_size = 5000
    
    transform = transforms.Compose([
        #transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])
    
    
    
    if False : 
        a = np.array(range(20))
        v = np.array(range(20))
        
        tx = np.array(range(20))
        
        for i in range(5) : 
            
            z = np.load(targetDir+'VA-Train-'+str(i)+'.npy')
            la = z[:,0]
            lv = z[:,1]
            
            #print(la,la.shape)
            
            a+=la
            v+=lv 
            
            z = np.load(targetDir+'VA-Test-'+str(i)+'.npy')
            la = z[:,0]
            lv = z[:,1]
            
            #print(la,la.shape)
            
            a+=la
            v+=lv
            
            
        
        fig = plt.figure()
    
        ax = plt.subplot(2, 2, 1)
        ax.bar(tx,a)
        ax.set_title('a')
        
        ax = plt.subplot(2, 2, 2)
        ax.bar(tx,v)
        ax.set_title('v')
        
        plt.show()
        
        
        
        #print(a)
        #print(v)
        exit(0)
    
    
    for split in range(5) : 
        #split = 1
        multi_gpu = False
        testSplit = split
        print("Test split " , testSplit)
        nSplit = 5
        listSplit = []
        for i in range(nSplit):
            if i!=testSplit : 
                listSplit.append(i)
        print(listSplit)
        
        if not isAFEW : 
        
            ID = SEWAFEWReduced(data_list = ["SEWA"], dir_gt = None, onlyFace = True, image_size = image_size, 
                      transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=listSplit,
                      isVideo=False, seqLength = 6,dbType = 1)
            
            VD = SEWAFEWReduced(data_list = ["SEWA"], dir_gt = None, onlyFace = True, image_size = image_size, 
                      transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=[testSplit],
                      isVideo=False, seqLength = 6,dbType = 1)
        else : 
        
            
            ID = SEWAFEWReduced(data_list = ["AFEW-VA-Fixed"], dir_gt = None, onlyFace = True, image_size = image_size, 
                      transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=listSplit,
                      isVideo=False, seqLength = 6,dbType = 0)
            
            VD = SEWAFEWReduced(data_list = ["AFEW-VA-Fixed"], dir_gt = None, onlyFace = True, image_size = image_size, 
                      transform = transform, useIT = True, augment = False, step = 1,split=True, nSplit = 5,listSplit=[testSplit],
                      isVideo=False, seqLength = 6,dbType = 0)
            
        
        dataloaderTrn = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True)
        dataloaderVal = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = True)
        
        
        if isAFEW : 
            listV = np.array(range(0,20))
            listA = np.array(range(0,20))
            
            listVx = np.array(range(0,20))
            listAx = np.array(range(0,20))
            
            listVt = np.array(range(0,20))
            listAt = np.array(range(0,20))
        else :
            listV = np.array(range(0,12))
            listA = np.array(range(0,12))
            
            listVx = np.array(range(0,12))
            listAx = np.array(range(0,12))
            
            listVt = np.array(range(0,12))
            listAt = np.array(range(0,12))
            
        
        
        
        x = 0
        
        for real_batch,vas,gt,M,_ in (dataloaderTrn) :
            for va in vas : 
                print(x)
                #print(va,vas)
                if not isAFEW : 
                    lva = (va.cpu().numpy()) * 10+1
                    name = 'S-VA-Train-'
                else : 
                    lva = (va.cpu().numpy())+10
                    name = 'VA-Train-'
                    
                listV[int(round(lva[0]))]+=1
                listA[int(round(lva[1]))]+=1
            
                x+=1
        
        x = 0
        
        print(listV,listA)
        np.save(targetDir+name+str(testSplit)+'.npy',np.column_stack((listV,listA)))
        
        for real_batch,vas,gt,M,_ in (dataloaderVal) :
            for va in vas : 
                print(x)
                if not isAFEW : 
                    lva = (va.cpu().numpy()) * 10+1
                    name = 'S-VA-Test-'
                else : 
                    lva = (va.cpu().numpy())+10
                    name = 'VA-Test-'
                    
                
                listVt[int(round(lva[0]))]+=1
                listAt[int(round(lva[1]))]+=1
                x+=1
            
        print(listVt,listAt)
        np.save(targetDir+name+str(testSplit)+'.npy',np.column_stack((listVt,listAt)))
        
        
        '''fig, ax = plt.subplots(nrows=1, ncols=2)
        
        for row in ax:
            for col in row:
                col.plot(x, y)'''
        
        fig = plt.figure()
    
        ax = plt.subplot(2, 2, 1)
        ax.bar(listVx,listV)
        ax.set_title('v train')
        
        ax = plt.subplot(2, 2, 2)
        ax.bar(listAx,listA)
        ax.set_title('A train')
        
        ax = plt.subplot(2, 2, 3)
        ax.bar(listVx,listVt)
        ax.set_title('v test')
        
        ax = plt.subplot(2, 2, 4)
        ax.bar(listAx,listAt)
        ax.set_title('A test')
        
        #plt.show()
        plt.savefig(name+'-'+str(split)+".png")
            
    exit(0)
    
    
def checkQuadrant() : 
    
    #Val, arou
    x = [-10,-10]
    y = [-10,10]
    z = [10,-10]
    a = [10,10]
    
    def toQuadrant(inputData = None, min = -10, max = 10,  toOneHot = False):
        threshold = truediv(min,max)
        vLow = False
        aLow = False
        q = 0
        
        if inputData[0] < threshold : 
            vLow = True
        
        if inputData[1] < threshold : 
            aLow = True
        
        if vLow and aLow : 
            q = 2
        elif vLow and not aLow : 
            q = 1 
        elif not vLow and not aLow : 
            q = 0 
        else : 
            q = 3 
        
        if toOneHot : 
            rest = np.zeros(4)
            rest[q]+=1
            return rest 
        else : 
            return q 
        
    
    print(toQuadrant(inputData = x,toOneHot = True))
    print(toQuadrant(inputData = y,toOneHot = True))
    print(toQuadrant(inputData = z,toOneHot = True))
    print(toQuadrant(inputData = a,toOneHot = True))

        
#main()
#cropImage()
#getDistributionAC()

