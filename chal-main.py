import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from model import Generator
from model import Discriminator,DiscriminatorM,DiscriminatorMST,DiscriminatorMZ
from torch.autograd import Variable
from torchvision.utils import save_image
from FacialDataset import AFEWVA,AFEWVAReduced,SEWAFEWReduced,AFFChallenge
from utils import *
import time
import torch.nn.functional as F
import numpy as np
import torch
import datetime
from torchvision import transforms
from torch import nn
from calcMetrix import *
from config import *

def str2bool(v):
    return v.lower() in ('true')

##############################################################
def train_only_disc():
    includeVal = True
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-split', nargs='?', const=1, type=int, default=1)#0,1,2
    parser.add_argument('-addLoss', nargs='?', const=1, type=int, default=1)#0,1,2
    parser.add_argument('-singleTask', nargs='?', const=1, type=int, default=0)#0,1,2
    
    parser.add_argument('-trainQuadrant', nargs='?', const=1, type=int, default=0)#0,1,2
    
    parser.add_argument('-dataset', nargs='?', const=1, type=int, default=2)#0,1,2, 0 is afew, 1 is sewa, 2 is challege
    parser.add_argument('-useWeightNormalization', nargs='?', const=1, type=int, default=1)#0,1,2
    
    
    args = parser.parse_args()
    split = args.split
    addLoss = args.addLoss 
    singleTask = args.singleTask 
    dataset = args.dataset
    useWeight = args.useWeightNormalization
    
    trainQuadrant = args.trainQuadrant
    alterQuadrant = True
    
    
    toLoad = True
    resume_iters=None #, help='resume training from this step') 

    # Model configuration.
    #curDir = "/home/deckyal/eclipse-workspace/FaceTracking/"
    c_dim=2
    c2_dim=8
    image_size=128
    g_conv_dim=32
    d_conv_dim=32
    g_repeat_num=6
    d_repeat_num=6
    lambda_cls=1
    lambda_rec=10
    lambda_gp=10
    
    # Training configuration.
    batch_size=1000##400 #500, help='mini-batch size')
    num_iters=200000 #, help='number of total iterations for training D')
    num_iters_decay=100000 #, help='number of iterations for decaying lr')
    g_lr=0.0001 #, help='learning rate for G')
    d_lr=0.0001 #, help='learning rate for D')
    n_critic=5 #, help='number of D updates per each G update')
    beta1=0.5 #, help='beta1 for Adam optimizer')
    beta2=0.999 #, help='beta2 for Adam optimizer')
    
    isVideo = False
    toAlign = False
    seq_length = 2
    
    # Test configuration.
    test_iters=200000 #, help='test model from this step')

    # Miscellaneous.
    num_workers=1
    mode='train' #, choices=['train', 'test'])
    use_tensorboard=False

    # Directories.
    celeba_image_dir='data/celeba/images'
    attr_path='data/celeba/list_attr_celeba.txt'
    rafd_image_dir='data/RaFD/train'
    
    log_dir='stargan/logs'
    model_save_dir='stargan/models'
    sample_dir='stargan/samples'
    result_dir='stargan/results'

    # Step size.
    log_step=10
    sample_step=1000
    model_save_step=10000
    lr_update_step=100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For fast training.
    cudnn.benchmark = True

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    #Split 
    #split = 0
    multi_gpu = False
    testSplit = split
    print("Test split " , testSplit)
    nSplit = 5
    listSplit = []
    for i in range(nSplit):
        if i!=testSplit : 
            listSplit.append(i)
    print(listSplit)
    
    if dataset == 0 : 
        main_name = 'AF-'
        d_name = 'AFEW-VA-Fixed'
        dbType = 0
    elif dataset == 1 : 
        main_name = 'SE-'
        d_name = 'SEWA'
        dbType = 1
    else : 
        main_name = 'CH-'
    
    if singleTask : 
        main_name+='ST-'
        
    if addLoss : 
        main_name+='AL-'
    
    if useWeight : 
        main_name+='W-'
    
    if trainQuadrant : 
        
        if alterQuadrant : 
            main_name+="-QDAL"
            c_dim = 1
        else :  
            main_name+="-QD"
            c_dim = 4
    
    save_name = main_name+str(testSplit)
    err_file = curDir+save_name+".txt"
    
    
    transform =transforms.Compose([
            transforms.Resize((image_size,image_size)),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    if dataset <= 1 : 
        
    
        ID = SEWAFEWReduced([d_name], None, True, image_size, transform, False, True, 1,split=True, nSplit = nSplit ,listSplit=listSplit
                    ,isVideo=isVideo, seqLength = seq_length, returnQuadrant=trainQuadrant,dbType = dbType,returnWeight = useWeight)
        
        #ID =FacialLandmarkDataset(['300W-Train','Menpo_Challenge/2D','300W_LP'], None, True, image_size, transform, use_internal_n, True, 1)
        dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True,worker_init_fn=worker_init_fn)
    
        #VD = AFEWVA(["AFEW-VA-PP"], None, True, image_size, transform, True, False, 1,split=True, nSplit = nSplit,listSplit=[testSplit],wHeatmap=False)
        VD = SEWAFEWReduced([d_name], None, True, image_size, transform, False, False, 1,split=True, nSplit = nSplit,listSplit=[testSplit]
                    ,isVideo=isVideo, seqLength = seq_length, returnQuadrant=trainQuadrant,dbType = dbType)
        
        dataloaderV = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = True)
    
    else : 
        listModeTrain = ['Train']
        if includeVal: 
            listModeTrain.append('Val')
            
        ID = AFFChallenge(data_list = ["AffectChallenge"],listMode = listModeTrain,onlyFace = True, image_size =112, 
            transform = transform,useIT = False,augment = False, step = 1,isVideo = False, seqLength = None,
            returnQuadrant = False, returnNoisy = False, returnWeight = useWeight)
        
        #ID =FacialLandmarkDataset(['300W-Train','Menpo_Challenge/2D','300W_LP'], None, True, image_size, transform, use_internal_n, True, 1)
        dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True,worker_init_fn=worker_init_fn)
    
        #VD = AFEWVA(["AFEW-VA-PP"], None, True, image_size, transform, True, False, 1,split=True, nSplit = nSplit,listSplit=[testSplit],wHeatmap=False)
        
        VD = AFFChallenge(data_list = ["AffectChallenge"],listMode = ['Val'],onlyFace = True, image_size =112, 
            transform = transform,useIT = False,augment = False, step = 1,isVideo = False, seqLength = None,
            returnQuadrant = False, returnNoisy = False, returnWeight = useWeight)
        
        dataloaderV = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = True)
        
    #Build model 
    """Create a generator and a discriminator."""
    model_ft = DiscriminatorMZ(image_size, d_conv_dim, c_dim, d_repeat_num)

    d_optimizer = torch.optim.Adam(model_ft.parameters(), d_lr, [beta1, beta2])
    print_network(model_ft, 'D')
    
    if toLoad:
        print('loading previous model ')
        model_ft.load_state_dict(torch.load(curDir+'t-models/'+save_name))
    else : 
        model_ft.apply(weights_init_uniform_rule)
    
    model_ft.to(device)
    
    
    
    save_name = main_name+'-IV-'+str(testSplit)
    err_file = curDir+save_name+".txt"
    
    d_lr = d_lr

    start_iters = 0
    
    '''if resume_iters:
        start_iters = resume_iters
        restore_model(resume_iters)'''

    # Start training.
    print('Start training...')
    start_time = time.time()
    
    f = open(err_file,'w+')
    f.write("err : ")
    f.close()
    
    #best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = 99999
    
    lMSA,lMSV,lCCV,lCCA,lICA,lICV,lCRA, lCRV, total = 9999,9999,-9999, -9999, -9999, -9999, -9999, -9999, -9999
    
    w,wv,wa = None,None,None
        
    for i in range(start_iters, num_iters):
        
        random.seed()
        manualSeed = random.randint(1, 10000) # use if you want new results
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        
        print('Epoch {}/{}'.format(i, num_iters - 1))
        print('-'*10)
        
        running_loss = 0
        
        for x,(data) in enumerate(dataloader,0) : 
            
            rinputs, rlabels,rldmrk,_ = data[0],data[1],data[2],data[3]
            
            if useWeight : 
                w = data[5].cuda()
                
                #print(w)
                
                #wv = w[:,1]
                #wa = w[:,0]
                
                #print(wv)
            
            model_ft.train()
            
            inputs = rinputs.cuda()#to(device)
            labels = rlabels.cuda()#to(device)
            
            d_optimizer.zero_grad() 
            
            _, outputs = model_ft(inputs)
            
            #print(outputs[:4],labels[:4])
                
            '''loss = criterion(outputs, labels)
            
            if useWeight : 
                loss *= w
            
            loss = loss.mean()'''
            loss = calcMSET(outputs,labels,w)
            
            if addLoss :
                
                ov,oa,lv,la = outputs[:,0],outputs[:,1], labels[:,0], labels[:,1]
                
                corV = -calcCORT(ov, lv, wv)
                corA = -calcCORT(oa, la, wa)
                
                cccV = -calcCCCT(ov, lv, wv)
                cccA = -calcCCCT(oa, la, wa)
                
                iccV = -calcICCT(ov, lv, wv)
                iccA = -calcICCT(oa, la, wa)
                
                #print(calcCORT(x, y))
                #print(calcICCT(x, y))
                #print(calcCCCT(x, y))
                
                lossO =corV+corA +cccV+cccA+iccV+iccA
            
            if not addLoss : 
                print("{}/{} loss : {}".format(x,int(len(dataloader.dataset)/batch_size),loss.item()))
            else : 
                print("{}/{} loss : {:.8f}, cor : {:.8f}/{:.8f}, ccc : {:.8f}/{:.8f}, icc : {:.8f}/{:.8f}".format(x,int(len(dataloader.dataset)/batch_size),
                        loss.item(),corV.item(),corA.item(),cccV.item(),cccA.item(),iccV.item(),iccA.item()))
            
            f = open(err_file,'a')
            if not addLoss : 
                f.write("{}/{} loss : {}\n".format(x,int(len(dataloader.dataset)/batch_size),loss.item()))
            else : 
                f.write("{}/{} loss : {:.3f}, cor : {:.3f}/{:.3f}, ccc : {:.3f}/{:.3f}, icc : {:.3f}/{:.3f}\n".format(x,int(len(dataloader.dataset)/batch_size),
                        loss.item(),corV.item(),corA.item(),cccV.item(),cccA.item(),iccV.item(),iccA.item()))
            f.close()
            
            if addLoss : 
                loss = loss+lossO
            
            loss.backward()
            d_optimizer.step()
                
            #statistics 
            running_loss += loss.item() * inputs.size(0)
            
            
        epoch_loss = running_loss / len(dataloader.dataset)
        print('Loss : {:.4f}'.format(epoch_loss))
        
        
        
        # Decay learning rates.
        if (i+1) % lr_update_step == 0 and (i+1) > 50 : #(num_iters - num_iters_decay):
            d_lr -= (d_lr / float(num_iters_decay))
            update_lr(d_lr,d_optimizer)
            print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
            
        
        if i %2 == 0 : 
            if multi_gpu : 
                torch.save(model_ft.module.state_dict(),curDir+'t-models/'+save_name)
            else : 
                torch.save(model_ft.state_dict(),curDir+'t-models/'+save_name)
    
        #Deep copy the model_ft 
        if i%5 == 0 :#epoch_loss < lowest_loss : 
            lowest_loss = lowest_loss 
            
            print("outp8ut : ",outputs[0])
            print("labels : ",labels[0])
            
            if True : 
            
                listValO = []
                listAroO = []
                
                listValL = []
                listAroL = []
                
                tvo = [];tao=[];tvl = []; tal = [];
                anyDiffer = False
               
                for x,(data) in enumerate(dataloaderV,0) :
                    
                    rinputs, rlabels,rldmrk,_ = data[0],data[1],data[2],data[3]
                    
                    model_ft.eval()    
                    inputs = rinputs.cuda()#to(device) 
                    labels = rlabels.cuda()#to(device)
                    
                    with torch.set_grad_enabled(False) : 
                        if not singleTask : 
                            _, outputs = model_ft(inputs)
                        else : 
                            outputs = model_ft(inputs)
                        print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs[:,0].shape[0],outputs.shape)
                        
                        if outputs[:,0].shape[0] != batch_size : #in case the batch size is differ, usually at end of iter
                            anyDiffer = True 
                            print('differ')
                            tvo.append(outputs[:,0].detach().cpu())
                            tao.append(outputs[:,1].detach().cpu())
                            
                            tvl.append(labels[:,0].detach().cpu())
                            tal.append(labels[:,1].detach().cpu())
                        else :
                            print('equal')
                            listValO.append(outputs[:,0].detach().cpu())
                            listAroO.append(outputs[:,1].detach().cpu())
                            
                            listValL.append(labels[:,0].detach().cpu())
                            listAroL.append(labels[:,1].detach().cpu())
                        
                
                est_V = np.asarray(torch.stack(listValO)).flatten()
                est_A = np.asarray(torch.stack(listAroO)).flatten()
                
                gt_V = np.asarray(torch.stack(listValL)).flatten()
                gt_A = np.asarray(torch.stack(listAroL)).flatten()
                
                if anyDiffer : 
                    est_Vt = np.asarray(torch.stack(tvo)).flatten()
                    est_At = np.asarray(torch.stack(tao)).flatten()
                    
                    gt_Vt = np.asarray(torch.stack(tvl)).flatten()
                    gt_At = np.asarray(torch.stack(tal)).flatten()
                    
                    #now concatenate
                    est_V = np.concatenate((est_V,est_Vt))
                    est_A = np.concatenate((est_A,est_At))
                    
                    gt_V = np.concatenate((gt_V,gt_Vt))
                    gt_A = np.concatenate((gt_A,gt_At))
                    
                print(est_V.shape, gt_V.shape)
                
                mseV = calcMSE(est_V, gt_V)
                mseA = calcMSE(est_A, gt_A)
                
                corV = calcCOR(est_V, gt_V)
                corA = calcCOR(est_A, gt_A)
                
                iccV = calcICC(est_V, gt_V)
                iccA = calcICC(est_A, gt_A)
                
                iccV2 = calcICC(gt_V, gt_V)
                iccA2 = calcICC(gt_A, gt_A)
                
                cccV = calcCCC(est_V, gt_V)
                cccA = calcCCC(est_A, gt_A)
                
                cccV2 = calcCCC(gt_V, gt_V)
                cccA2 = calcCCC(gt_A, gt_A)
                
                
                if lMSA > mseA : 
                    lMSA = mseA
                if lMSV > mseV : 
                    lMSV = mseV
                    
                if corA > lCRA : 
                    lCRA = corA
                if corV > lCRV : 
                    lCRV = corV
                    
                if cccA > lCCA : 
                    lCCA = cccA
                if cccV > lCCV : 
                    lCCV = cccV
                    
                if iccA > lICA : 
                    lICA = iccA
                if iccV > lICV : 
                    lICV = iccV
                    
                if (corA+corV+cccA+cccV+iccA+iccV) > total : 
                    total = (corA+corV+cccA+cccV+iccA+iccV)
                    if multi_gpu : 
                        torch.save(model_ft.module.state_dict(),curDir+'t-models/'+save_name+'-best')
                    else : 
                        torch.save(model_ft.state_dict(),curDir+'t-models/'+save_name+'-best')
                
                print('Best, MSEA : '+str(lMSA)+', CORA : '+str(lCRA)+', CCCA : '+str(lCCA)+', ICCA : '+str(lICA)+ ', MSEV : ' +str(lMSV)+ ', CORV : ' +str(lCRV)+', CCCV : '+str(lCCV) +', ICCV : '+str(lICV)+', Total : '+str(total))
                
                print('MSEV : ',mseV, ', CORV : ',corV,', CCCV : ',cccV,', CCCV2 : ',cccV2,', ICCV : ',iccV,', ICCV2 : ',iccV2)
                print('MSEA : ',mseA, ', CORA : ',corA,', CCCA : ',cccA,', CCCA2 : ',cccA2,', ICCA : ',iccA,', ICCA2 : ',iccA2)
                
                f = open(err_file,'a')
                res = 'MSEV : '+str(mseV)+ ', CORV : ' +str(corV)+', CCCV : '+str(cccV) +', ICCV : '+str(iccV)+' \n '
                f.write(res) 
                res = 'MSEA : '+str(mseA)+ ', CORA : '+str(corA) +', CCCA : '+str(cccA) +', ICCA : '+str(iccA)+' \n '
                f.write(res)
                res = 'Best, MSEA : '+str(lMSA)+', CORA : '+str(lCRA)+', CCCA : '+str(lCCA)+', ICCA : '+str(lICA)+ ', MSEV : ' +str(lMSV)+ ', CORV : ' +str(lCRV)+', CCCV : '+str(lCCV) +', ICCV : '+str(lICV)+', Total : '+str(total)+' \n '
                f.write(res)
                 
                f.close()

    print('Best val Acc: {:4f}'.format(lowest_loss))
    return 



def test_only_disc():
    
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-split', nargs='?', const=1, type=int, default=3)#0,1,2
    parser.add_argument('-addLoss', nargs='?', const=1, type=int, default=1)#0,1,2
    parser.add_argument('-singleTask', nargs='?', const=1, type=int, default=0)#0,1,2
    
    parser.add_argument('-trainQuadrant', nargs='?', const=1, type=int, default=0)#0,1,2
    
    parser.add_argument('-sewa', nargs='?', const=1, type=int, default=0)#0,1,2
    parser.add_argument('-useWeightNormalization', nargs='?', const=1, type=int, default=1)#0,1,2
    
    
    args = parser.parse_args()
    split = args.split
    addLoss = args.addLoss 
    singleTask = args.singleTask 
    isSewa = args.sewa 
    useWeight = args.useWeightNormalization
    
    trainQuadrant = args.trainQuadrant
    alterQuadrant = True
    
    
    toLoad = True
    resume_iters=None #, help='resume training from this step') 

    # Model configuration.
    #curDir = "/home/deckyal/eclipse-workspace/FaceTracking/"
    c_dim=2
    c2_dim=8
    image_size=128
    g_conv_dim=64
    d_conv_dim=64
    g_repeat_num=6
    d_repeat_num=6
    
    # Training configuration.
    batch_size=100#400 #500, help='mini-batch size')
    isVideo = False
    toAlign = False
    seq_length = 2
    

    # Miscellaneous.
    num_workers=1

    # Directories.
    celeba_image_dir='data/celeba/images'
    attr_path='data/celeba/list_attr_celeba.txt'
    rafd_image_dir='data/RaFD/train'
    
    log_dir='stargan/logs'
    model_save_dir='stargan/models'
    sample_dir='stargan/samples'
    result_dir='stargan/results'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For fast training.
    cudnn.benchmark = True
    
    #Split 
    #split = 0
    multi_gpu = False
    testSplit = split
    print("Test split " , testSplit)
    nSplit = 5
    listSplit = []
    for i in range(nSplit):
        if i!=testSplit : 
            listSplit.append(i)
    print(listSplit)
    
    if not isSewa : 
        main_name = 'AF-'
        d_name = 'AFEW-VA-Fixed'
        dbType = 0
    else : 
        main_name = 'SE-'
        d_name = 'SEWA'
        dbType = 1
    
    if singleTask : 
        main_name+='ST-'
        
    if addLoss : 
        main_name+='AL-'
    
    if useWeight : 
        main_name+='W-'
    
    if trainQuadrant : 
        
        if alterQuadrant : 
            main_name+="-QDAL"
            c_dim = 1
        else :  
            main_name+="-QD"
            c_dim = 4
    
    save_name = main_name+str(testSplit)
    
    
    transform =transforms.Compose([
            transforms.Resize((image_size,image_size)),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    #VD = AFEWVA(["AFEW-VA-PP"], None, True, image_size, transform, True, False, 1,split=True, nSplit = nSplit,listSplit=[testSplit],wHeatmap=False)
    VD = SEWAFEWReduced([d_name], None, True, image_size, transform, False, False, 1,split=True, nSplit = nSplit,listSplit=[testSplit]
                ,isVideo=isVideo, seqLength = seq_length, returnQuadrant=trainQuadrant,dbType = dbType)
    
    dataloaderV = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = False)
   
    #Build model 
    """Create a generator and a discriminator."""
    if not singleTask : 
        model_ft = DiscriminatorM112(image_size, d_conv_dim, c_dim, d_repeat_num)
    else : 
        model_ft = DiscriminatorMST(image_size, d_conv_dim, c_dim, d_repeat_num)

    print_network(model_ft, 'D')
    
    if toLoad:
        print('loading previous model ')
        model_ft.load_state_dict(torch.load(curDir+'t-models/'+save_name))
        
    model_ft.to(device)
    
    print('Starting evaluation...')

    listValO = []
    listAroO = []
    
    listValL = []
    listAroL = []
    
    tvo = [];tao=[];tvl = []; tal = [];
    anyDiffer = False
   
    for x,(data) in enumerate(dataloaderV,0) :
        
        rinputs, rlabels,rldmrk,_ = data[0],data[1],data[2],data[3]
        
        print('evalling')
        model_ft.eval()    
        inputs = rinputs.cuda()#to(device) 
        labels = rlabels.cuda()#to(device)
        
        with torch.set_grad_enabled(False) : 
            if not singleTask : 
                _, outputs = model_ft(inputs,False)
            else : 
                outputs = model_ft(inputs,False)
            print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs[:,0].shape[0],outputs.shape)
            
            if outputs[:,0].shape[0] != batch_size : #in case the batch size is differ, usually at end of iter
                anyDiffer = True 
                print('differ')
                tvo.append(outputs[:,0].detach().cpu())
                tao.append(outputs[:,1].detach().cpu())
                
                tvl.append(labels[:,0].detach().cpu())
                tal.append(labels[:,1].detach().cpu())
            else :
                print('equal')
                listValO.append(outputs[:,0].detach().cpu())
                listAroO.append(outputs[:,1].detach().cpu())
                
                listValL.append(labels[:,0].detach().cpu())
                listAroL.append(labels[:,1].detach().cpu())
            
    
    est_V = np.asarray(torch.stack(listValO)).flatten()
    est_A = np.asarray(torch.stack(listAroO)).flatten()
    
    gt_V = np.asarray(torch.stack(listValL)).flatten()
    gt_A = np.asarray(torch.stack(listAroL)).flatten()
    
    if anyDiffer : 
        est_Vt = np.asarray(torch.stack(tvo)).flatten()
        est_At = np.asarray(torch.stack(tao)).flatten()
        
        gt_Vt = np.asarray(torch.stack(tvl)).flatten()
        gt_At = np.asarray(torch.stack(tal)).flatten()
        
        #now concatenate
        est_V = np.concatenate((est_V,est_Vt))
        est_A = np.concatenate((est_A,est_At))
        
        gt_V = np.concatenate((gt_V,gt_Vt))
        gt_A = np.concatenate((gt_A,gt_At))
        
    print(est_V.shape, gt_V.shape)
    
    mseV = calcMSE(est_V, gt_V)
    mseA = calcMSE(est_A, gt_A)
    
    corV = calcCOR(est_V, gt_V)
    corA = calcCOR(est_A, gt_A)
    
    iccV = calcICC(est_V, gt_V)
    iccA = calcICC(est_A, gt_A)
    
    iccV2 = calcICC(gt_V, gt_V)
    iccA2 = calcICC(gt_A, gt_A)
    
    cccV = calcCCC(est_V, gt_V)
    cccA = calcCCC(est_A, gt_A)
    
    cccV2 = calcCCC(gt_V, gt_V)
    cccA2 = calcCCC(gt_A, gt_A)
    
    print('MSEV : ',mseV, ', CORV : ',corV,', CCCV : ',cccV,', CCCV2 : ',cccV2,', ICCV : ',iccV,', ICCV2 : ',iccV2)
    print('MSEA : ',mseA, ', CORA : ',corA,', CCCA : ',cccA,', CCCA2 : ',cccA2,', ICCA : ',iccA,', ICCA2 : ',iccA2)

    pass


if __name__ == '__main__':
    train_only_disc()
    #test_only_disc()