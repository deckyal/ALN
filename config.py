'''
Created on Feb 7, 2018

@author: deckyal
'''

channels = 3
crop_size = 128
n_neurons = 512
n_o = 136
crop_size = 128

num_balls = 1;

imWidth = 1280#480
imHeight = 720#360    

runServer = False
useFullModel = False
experimental = False

addChannel = False
useDoubleLSTM = True



curDir = "/home/deckyal/eclipse-workspace/FaceTracking/FaceTracking-NR/StarGAN_Collections/stargan-master/"
serverAdder = 0

if runServer : 
    curDir = '/homedtic/daspandilatif/workspace/FaceTracking/FaceTracking-NR/StarGAN_Collections/stargan-master/'
    
if useFullModel : 
    n_neurons = 1024 
    crop_size = 256
    useDoubleLSTM = True
