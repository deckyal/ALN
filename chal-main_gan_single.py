import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from model import GeneratorM, GeneratorMZ, DiscriminatorM,DiscriminatorMST, DiscriminatorMZ

from torch.autograd import Variable
from torchvision.utils import save_image
from FacialDataset import SEWAFEWReduced,AFFChallenge
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
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-split', nargs='?', const=1, type=int, default=0)#0,1,2
parser.add_argument('-sewa', nargs='?', const=1, type=int, default=0)#0,1,2

#dont change 
parser.add_argument('-useSkip', nargs='?', const=1, type=int, default=0)#0,1,2 #To use skip : no difference
parser.add_argument('-useLatent', nargs='?', const=1, type=int, default=0)#0,1,2 #To use linear latent : bad
parser.add_argument('-tryDenoise', nargs='?', const=1, type=int, default=1)#0,1,2. Helpfull
parser.add_argument('-useWeightNormalization', nargs='?', const=0, type=int, default=1)#0,1,2. helpfull
parser.add_argument('-addLoss', nargs='?', const=1, type=int, default=1)#0,1,2. helpfull
parser.add_argument('-singleTask', nargs='?', const=1, type=int, default=0)#0,1,2. Multitask is slightly better

parser.add_argument('-addZ', nargs='?', const=1, type=int, default=1)#0,1,2. Multitask is slightly better
parser.add_argument('-addS', nargs='?', const=1, type=int, default=1)#0,1,2. Multitask is slightly better

parser.add_argument('-dataset', nargs='?', const=1, type=int, default=1)# 1 is ac , 0 is other

#may change
parser.add_argument('-trainQuadrant', nargs='?', const=1, type=int, default=0)#0,1,2
parser.add_argument('-alterQuadrant', nargs='?', const=1, type=int, default=0)#0,1,2

args = parser.parse_args()

def str2bool(v):
    return v.lower() in ('true')
##############################################################

def train_w_g_adl(): #training g and d using semi-supervision loss + cycle loss. 
    #stripped : 
    
    includeVal = True
    '''
    onlyAdversaryLoss = False  # to see the effect of adversary only loss
    normalizeVA = False #to normalize the VA, seems no substantial impact 
    tryReduced = True #to reduce the dataset
    toAlign = False #we don't need to align the data
    
    onlyAdversaryLoss = False
    normalizeVA = False
    tryReduced = True
    
    '''
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    split = args.split
    isSewa = args.sewa 
    
    dataset = args.dataset
    
    toLoadModel = True
    resume_iters=27
    
    use_skip = args.useSkip
    useLatent = args.useLatent
    tryDenoise = args.tryDenoise
    addLoss = args.addLoss
    useWeight = args.useWeightNormalization 
    
    singleTask = args.singleTask 
    trainQuadrant = args.trainQuadrant
    alterQuadrant = args.alterQuadrant
    addZ = args.addZ
    addS = args.addS
    
    #curDir = "/home/deckyal/eclipse-workspace/FaceTracking/"
    c_dim=2
    c2_dim=8
    image_size=128
    g_conv_dim=16
    d_conv_dim=32
    g_repeat_num=6
    d_repeat_num=6
    lambda_cls=1
    lambda_rec=10
    lambda_gp=10
    inputC = 3#input channel for discriminator 
    
    
    # Training configuration.
    batch_size=500#500#50#40#70#20 #, help='mini-batch size')
    num_iters=200000 #, help='number of total iterations for training D')
    num_iters_decay=100000 #, help='number of iterations for decaying lr')
    g_lr=0.0001 #, help='learning rate for G')
    d_lr=0.0001 #, help='learning rate for D')
    n_critic=5 #, help='number of D updates per each G update')
    beta1=0.5 #, help='beta1 for Adam optimizer')
    beta2=0.999 #, help='beta2 for Adam optimizer')
    #selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'] 
    #', '--list', nargs='+', help='selected attributes for the CelebA dataset',default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    
    isVideo = False
    seq_length = 2
    
    # Test configuration.
    test_iters=200000 #, help='test model from this step')

    # Miscellaneous.
    num_workers=1

    log_dir='stargan/logs'
    model_save_dir='stargan/models'
    sample_dir='stargan/samples-g_adl'
    result_dir='stargan/results'

    # Step size.
    log_step=20
    sample_step=5#1000
    model_save_step=2
    lr_update_step=100#1000
    
    #model_save_step=10000
    #lr_update_step=1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    multi_gpu = True
    testSplit = split
    print("Test split " , testSplit)
    nSplit = 5
    listSplit = []
    for i in range(nSplit):
        if i!=testSplit : 
            listSplit.append(i)
    print(listSplit)
    
    if dataset == 1 : 
        d_name = 'AFEW-VA-Fixed'
        dbType = 0
        additionName = "AC"+str(split)+"-"
    else : 
        if not isSewa : 
            d_name = 'AFEW-VA-Fixed'
            dbType = 0
            additionName = "AF"+str(split)+"-"
        else : 
            d_name = 'SEWA'
            dbType = 1
            additionName = "SW"+str(split)+"-"
    
    
    if singleTask : 
        additionName+='ST-'    
    if addLoss : 
        additionName+='AL-'
    if useWeight : 
        additionName+='W-'
    if addZ : 
        additionName+='Z-'
    if addS : 
        additionName+='S-'
        inputC = 4
        
    if trainQuadrant : 
        if alterQuadrant : 
            additionName+="QDAL-"
            c_dim = 1
        else :  
            additionName+="QD-"
            c_dim = 4
    if tryDenoise :
        additionName+="Den-"
    
    
    transform =transforms.Compose([
            #transforms.Resize((image_size,image_size)),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    #AFEW-VA-Small
    
    listModeTrain = ['Train']
    if includeVal: 
        listModeTrain.append('Val')
    
    if dataset == 1 : 
        
        ID = AFFChallenge(data_list = ["AffectChallenge"],listMode = listModeTrain, onlyFace = True, image_size =112, 
            transform = transform,useIT = False,augment = False, step = 1,isVideo = False, seqLength = None, dbType = 0,
            returnQuadrant = False, returnNoisy = tryDenoise, returnWeight = True,returnSound = addS)
        
        VD = AFFChallenge(data_list = ["AffectChallenge"],listMode = ['Val'],onlyFace = True, image_size =112, 
            transform = transform,useIT = False,augment = False, step = 1,isVideo = False, seqLength = None, dbType = 0,
            returnQuadrant = False, returnNoisy = tryDenoise, returnWeight = True,returnSound = addS)
    else : 
        
        ID = SEWAFEWReduced([d_name], None, True, image_size, transform, False, True, 1,split=True, nSplit = nSplit ,listSplit=listSplit
                    ,isVideo=isVideo, seqLength = seq_length, returnQuadrant=trainQuadrant, returnNoisy = tryDenoise,dbType = dbType,returnWeight = useWeight)
        #ID = AFEWVA([d_name], None, True, image_size, transform, False, True, 1,split=True, nSplit = nSplit ,listSplit=listSplit
        #           ,isVideo=isVideo, seqLength = seq_length, returnQuadrant=trainQuadrant, returnNoisy = tryDenoise,dbType = dbType,returnWeight = useWeight)
        
        VD = SEWAFEWReduced([d_name], None, True, image_size, transform, False, False, 1,split=True, nSplit = nSplit,listSplit=[testSplit]
                    ,isVideo=isVideo, seqLength = seq_length, returnQuadrant=trainQuadrant, returnNoisy = tryDenoise,dbType = dbType)
        #VD = AFEWVA([d_name], None, True, image_size, transform, False, False, 1,split=True, nSplit = nSplit,listSplit=[testSplit]
        #            ,isVideo=isVideo, seqLength = seq_length, returnNoisy = tryDenoise,dbType = dbType)
    
    dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True,worker_init_fn=worker_init_fn)
        
    dataloaderV = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = False)
   
    #Build model 
    """Create a generator and a discriminator.""" 
    G = GeneratorMZ(g_conv_dim, 0, g_repeat_num,use_skip,useLatent)
    #if not singleTask : 
    D = DiscriminatorMZ(image_size, d_conv_dim, c_dim, d_repeat_num,inputC=inputC)
    '''
    else : 
        D = DiscriminatorMST(image_size, d_conv_dim, c_dim, d_repeat_num,asDiscriminator = True)'''
        
        
    #G = Generator(g_conv_dim, 0, g_repeat_num,use_skip)
    #D = Discriminator(image_size, d_conv_dim, c_dim, d_repeat_num)
    
    print_network(G, 'G')
    print_network(D, 'D')
    
    if toLoadModel :
        print('Loading models from iterations : ',resume_iters) 
        G_path = os.path.join(curDir+model_save_dir, '{}G-{}.ckpt'.format(additionName,resume_iters))
        D_path = os.path.join(curDir+model_save_dir, '{}D-{}.ckpt'.format(additionName,resume_iters))
        print('loading ',G_path)
        print('loading ',D_path)
        G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    else : 
        print('Initiating models')
        G.apply(weights_init_uniform_rule)
        D.apply(weights_init_uniform_rule)
    
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])
    
    G.to(device)
    D.to(device)
    
    if multi_gpu : 
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
    
    if includeVal : 
        additionName+='-IV-'
        
    
    save_name = additionName+str(testSplit)
    err_file = curDir+save_name+".txt"
    
    print('err file : ',err_file)
    
    
    #print(D.linear41.weight.data)

    # Set data loader.
    data_loader = dataloader
    
    if not trainQuadrant or (alterQuadrant): 
        criterion = nn.MSELoss()
    else : 
        criterion = nn.CrossEntropyLoss() #F.cross_entropy(logit, target)

    # Fetch fixed inputs for debugging.
    data = next(iter(dataloader))
    x_fixed, rlabels,rldmrk,_ = data[0],data[1],data[2],data[3]#    x_fixed, c_org
    
    if trainQuadrant :
        if tryDenoise : 
            x_fixed = data[6].cuda()
            x_target = data[0].cuda()
    else : 
        if tryDenoise : 
            x_fixed = data[5].cuda()
            x_target = data[0].cuda()
        
    x_fixed = x_fixed.to(device)
    #c_fixed_list = create_labels(c_org, c_dim, dataset, selected_attrs)

    # Learning rate cache for decaying.
    d_lr = d_lr

    # Start training from scratch or resume training.
    start_iters = 0
    
    '''if resume_iters:
        start_iters = resume_iters
        restore_model(resume_iters)'''

    # Start training.
    print('Start training...')
    start_time = time.time()
    
    if trainQuadrant : 
        q1 = data[4]
    
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
        
        G.train()
        D.train()
        
        for x,(data) in enumerate(dataloader,0) :
            rinputs, rlabels,rldmrk,_ =data[0],data[1],data[2],data[3]
            if trainQuadrant :
                if alterQuadrant : 
                    quadrant = data[5].float().cuda()
                else : 
                    quadrant = data[5].cuda()
                
                if tryDenoise : 
                    noisy = data[6].cuda()
                
            else : 
                if tryDenoise : 
                    noisy = data[5].cuda()
                
                    if useWeight : 
                        w = data[6].cuda()
                        #print(w)
                        #wv = w[:,1]
                        #wa = w[:,0]
                else : 
                    if useWeight : 
                        w = data[5].cuda()
                        #print(w)
                        #wv = w[:,1]
                        #wa = w[:,0]
            if addS : 
                sound = data[-1].cuda()
            else : 
                sound = None
            
            inputs = rinputs.cuda()#to(device)
            labels = rlabels.cuda()#to(device)
            
            # Compute loss with real images.
            out_src, out_cls = D(inputs,s=sound)
            d_loss_real = - torch.mean(out_src)
            
            if not trainQuadrant: 
                if useWeight :
                    d_loss_cls = calcMSET(out_cls,labels,w) #criterion(out_cls, labels)
                else : 
                    d_loss_cls = criterion(out_cls, labels) #classification_loss(out_cls, label_org, dataset)
                    
                if addLoss :
                    ov,oa,lv,la = out_cls[:,0],out_cls[:,1], labels[:,0], labels[:,1]
                    
                    corV = -calcCORT(ov, lv, wv)
                    corA = -calcCORT(oa, la, wa)
                    
                    cccV = -calcCCCT(ov, lv, wv)
                    cccA = -calcCCCT(oa, la, wa)
                    
                    iccV = -calcICCT(ov, lv, wv)
                    iccA = -calcICCT(oa, la, wa)
                    
                    d_loss_cls = d_loss_cls + corV+corA +cccV+cccA+iccV+iccA
            else :
                #print('q ',quadrant)
                #print(out_cls.shape, quadrant.shape )
                if alterQuadrant : 
                    d_loss_cls = criterion(torch.squeeze(out_cls), quadrant)
                else : 
                    d_loss_cls = criterion(out_cls, quadrant)
            
            if x%10 == 0 : 
                if not trainQuadrant: 
                    print(x,'-',len(dataloader)," Res - label-G : ", out_cls[:3],labels[:3])
                else : 
                    if alterQuadrant :
                        print(x,'-',len(dataloader)," Res - label-G : ", torch.round(out_cls[:3]),quadrant[:3]) 
                    else : 
                        print(x,'-',len(dataloader)," Res - label-G : ", torch.max(out_cls[:3],1)[1],quadrant[:3])
            
            # Compute loss with fake images.
            if tryDenoise : 
                x_fake,inter = G(noisy,returnInter=True)
            else : 
                x_fake,inter = G(inputs,returnInter=True)
            
            if addZ : 
                out_src, out_cls = D(x_fake.detach(), z = inter,s=sound)
            else : 
                out_src, out_cls = D(x_fake.detach(),s=sound)
                
            d_loss_fake = torch.mean(out_src)
    
            # Compute loss for gradient penalty.
            alpha = torch.rand(inputs.size(0), 1, 1, 1).to(device)
            x_hat = (alpha * inputs.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = D(x_hat,s=sound)
            d_loss_gp = gradient_penalty(out_src, x_hat)
    
            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + lambda_cls * d_loss_cls + lambda_gp * d_loss_gp
            
            #reset_grad()
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            
            d_loss.backward()
            d_optimizer.step()
            
            
            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            
            ###! Actual training of the generator 
                    
            if (i+1) % n_critic == 0:
                
                # Original-to-target domain.
                if tryDenoise : 
                    x_fake,inter = G(noisy,returnInter=True)
                else : 
                    x_fake,inter = G(inputs,returnInter=True)
                    
                if addZ: 
                    out_src, out_cls = D(x_fake, z = inter,s=sound)
                else : 
                    out_src, out_cls = D(x_fake,s=sound)
                
                if x%10 == 0 : 
                    print("Res - label-D : ", out_cls[:3],labels[:3])
                    
                g_loss_fake = - torch.mean(out_src)
                
                if not trainQuadrant: 
                    #g_loss_cls = criterion(out_cls, labels) #classification_loss(out_cls, label_org, dataset)
                    
                    if useWeight :
                        g_loss_cls = calcMSET(out_cls,labels,w) #criterion(out_cls, labels)
                    else : 
                        g_loss_cls = criterion(out_cls, labels) #classification_loss(out_cls, label_org, dataset)
                        
                    if addLoss :
                        ov,oa,lv,la = out_cls[:,0],out_cls[:,1], labels[:,0], labels[:,1]
                        
                        corV = -calcCORT(ov, lv, wv)
                        corA = -calcCORT(oa, la, wa)
                        
                        cccV = -calcCCCT(lv, lv, wv)
                        cccA = -calcCCCT(oa, la, wa)
                        
                        iccV = -calcICCT(ov, lv, wv)
                        iccA = -calcICCT(oa, la, wa)
                        
                        g_loss_cls = g_loss_cls + corV+corA +cccV+cccA+iccV+iccA
                        
                    
                else : 
                    if alterQuadrant : 
                        g_loss_cls = criterion(torch.squeeze(out_cls), quadrant)
                    else : 
                        g_loss_cls = criterion(out_cls, quadrant)
                    
                #g_loss_cls = criterion(out_cls, labels) #classification_loss(out_cls, label_trg, dataset)
    
                # Target-to-original domain.
                x_reconst = G(x_fake)
                g_loss_rec = torch.mean(torch.abs(inputs - x_reconst))
    
                # Backward and optimize.
                g_loss = g_loss_fake + lambda_rec * g_loss_rec + lambda_cls * g_loss_cls
                
                #reset_grad()    
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()
                
                g_loss.backward()
                g_optimizer.step()
    
                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                
                ###! Getting the training metrics and samples    
                #running_loss += loss.item() * inputs.size(0)
                #print("{}/{} loss : {}/{}".format(x,int(len(dataloader.dataset)/batch_size),lossC.item(),lossR.item()))
                
                '''
                g_optimizer.zero_grad() 
                inputsM = G(inputs)
                
                _, outputs = D(inputsM)
                lossC = criterion(outputs, labels)
                lossR = criterion(inputsM, inputs)
                
                loss=lossC+lossR
                
                loss.backward()
                g_optimizer.step()
                    
                #statistics 
                running_loss += loss.item() * inputs.size(0)
                print("{}/{} loss : {}/{}".format(x,int(len(dataloader.dataset)/batch_size),lossC.item(),lossR.item()))
                '''
             
            if (x+1) % 10 == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}], Inner {}/{} \n".format(et, i+1, num_iters,x,int(len(dataloader.dataset)/batch_size))
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                
                
                f = open(err_file,'a')
                f.write("Elapsed [{}], Iteration [{}/{}], Inner {}/{} \n".format(et, i+1, num_iters,x,int(len(dataloader.dataset)/batch_size)))
                f.write(log) 
                f.close()
                    
            
        # Translate fixed images for debugging.
        if (i+1) % sample_step == 0 or True:
            with torch.no_grad():
                x_fake_list = [x_fixed]
                #for c_fixed in c_fixed_list:
                #    x_fake_list.append(G(x_fixed, c_fixed))
                #x_concat = torch.cat(x_fake_list, dim=3)
                x_concat = G(x_fixed)
                sample_path = os.path.join(curDir+sample_dir, '{}{}-images-denoised.jpg'.format(i+1,additionName))
                save_image(denorm(x_concat.data.cpu()), sample_path, nrow=int(round(16)), padding=0)
                print('Saved real and fake denoised images into {}...'.format(sample_path))
                
                if tryDenoise : 
                    x_concat = x_fixed
                    sample_path = os.path.join(curDir+sample_dir, '{}{}-images-original.jpg'.format(i+1,additionName))
                    save_image(denorm(x_concat.data.cpu()), sample_path, nrow=int(round(16)), padding=0)
                    print('Saved real and fake real images into {}...'.format(sample_path))
                    
                    x_concat = x_target
                    sample_path = os.path.join(curDir+sample_dir, '{}{}-images-groundtruth.jpg'.format(i+1,additionName))
                    save_image(denorm(x_concat.data.cpu()), sample_path, nrow=int(round(16)), padding=0) #batch_size/4
                    print('Saved real and fake real images into {}...'.format(sample_path))

        # Save model checkpoints.
        if (i+1) % model_save_step == 0:
            G_path = os.path.join(curDir+model_save_dir, '{}G-{}.ckpt'.format(additionName,i))
            D_path = os.path.join(curDir+model_save_dir, '{}D-{}.ckpt'.format(additionName,i))
            
            if multi_gpu : 
                #torch.save(D.module.state_dict(),curDir+'t-models/'+'-D'+save_name)
                #torch.save(G.module.state_dict(),curDir+'t-models/'+'-G'+save_name)
                torch.save(G.module.state_dict(), G_path)
                torch.save(D.module.state_dict(), D_path)
            else : 
                torch.save(G.state_dict(), G_path)
                torch.save(D.state_dict(), D_path)
            print('Saved model checkpoints into {}...'.format(model_save_dir))
            print(G_path)

        # Decay learning rates.
        if (i+1) % lr_update_step == 0 and (i+1) > 50:
            g_lr -= (g_lr / float(num_iters_decay))
            d_lr -= (d_lr / float(num_iters_decay))
            update_lr_ind(d_optimizer,d_lr)
            update_lr_ind(g_optimizer,g_lr)
            print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

            
            
        epoch_loss = running_loss / len(dataloader.dataset)
        print('Loss : {:.4f}'.format(epoch_loss))
        
        if i %2 == 0 : 
            G_path = os.path.join(curDir,'t-models/', '{}G-{}.ckpt'.format(additionName,i))
            D_path = os.path.join(curDir,'t-models/',  '{}D-{}.ckpt'.format(additionName,i))
            
            #torch.save(G.state_dict(), G_path)
            #torch.save(D.state_dict(), D_path)
            
            if multi_gpu : 
                #torch.save(D.module.state_dict(),curDir+'t-models/'+'-D'+save_name)
                #torch.save(G.module.state_dict(),curDir+'t-models/'+'-G'+save_name)
                
                torch.save(G.module.state_dict(), G_path)
                torch.save(D.module.state_dict(), D_path)
            else : 
                torch.save(G.state_dict(), G_path)
                torch.save(D.state_dict(), D_path)
                
                #torch.save(D.state_dict(),curDir+'t-models/'+'-D'+save_name)
                #torch.save(G.state_dict(),curDir+'t-models/'+'-G'+save_name)
    
        #Deep copy the model_ft 
        if i%5 == 0 :#epoch_loss < lowest_loss : 
            
            
            if trainQuadrant : 
                a = 0
                b = 0
            else : 
                a = 0
                b = 1
            
            lowest_loss = lowest_loss 
            
            print("outp8ut : ",out_cls[0])
            print("labels : ",labels[0])
            
            if True : 
            
                listValO = []
                listAroO = []
                
                listValL = []
                listAroL = []
                
                tvo = [];tao=[];tvl = []; tal = [];
                anyDiffer = False
               
                for x,(data) in enumerate(dataloaderV,0) :
                    
                    if trainQuadrant: 
                        rinputs, rlabels,rldmrk = data[0],data[5],data[2]
                    else : 
                        rinputs, rlabels,rldmrk = data[0],data[1],data[2]
                        
                    if addS : 
                        sound = data[-1].cuda()
                    else : 
                        sound = None
                    
                    G.eval()    
                    D.eval()
                    
                    inputs = rinputs.cuda()#to(device) 
                    labels = rlabels.cuda()#to(device)
                    
                    with torch.set_grad_enabled(False) : 
                        
                        if addZ : 
                            inputsM,z = G(inputs,returnInter=True)
                        else : 
                            inputsM,z = G(inputs),None
                        #print('s',sound.shape, z.shape)
                        
                        _, outputs = D(inputsM,s=sound,z=z)
                        
                        if trainQuadrant: 
                            if alterQuadrant :
                                outputs = torch.round(outputs) 
                            else : 
                                _,outputs = torch.max(outputs,1)
                        
                        if trainQuadrant : 
                            print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs.shape)
                        else : 
                            print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs[:,0].shape[0],outputs.shape)
                        #print(outputs.shape)
                        
                        if not trainQuadrant : 
                            shape = outputs[:,0].shape[0]
                        else : 
                            shape = outputs.shape[0]
                        
                        if shape != batch_size : #in case the batch size is differ, usually at end of iter
                            anyDiffer = True 
                            print('differ')
                            if trainQuadrant: 
                                tvo.append(outputs.detach().cpu())
                                tao.append(outputs.detach().cpu())
                                
                                tvl.append(labels.detach().cpu())
                                tal.append(labels.detach().cpu())
                            else : 
                                tvo.append(outputs[:,a].detach().cpu())
                                tao.append(outputs[:,b].detach().cpu())
                                
                                tvl.append(labels[:,a].detach().cpu())
                                tal.append(labels[:,b].detach().cpu())
                        else :
                            print('equal')
                            if trainQuadrant : 
                                listValO.append(outputs.detach().cpu())
                                listAroO.append(outputs.detach().cpu())
                                
                                listValL.append(labels.detach().cpu())
                                listAroL.append(labels.detach().cpu())
                            else : 
                                listValO.append(outputs[:,a].detach().cpu())
                                listAroO.append(outputs[:,b].detach().cpu())
                                
                                listValL.append(labels[:,a].detach().cpu())
                                listAroL.append(labels[:,b].detach().cpu())
                                
                        
                
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
                
                cccV = calcCCC(est_V, gt_V)
                cccA = calcCCC(est_A, gt_A)
                
                iccV2 = calcCCC(gt_V, gt_V)
                iccA2 = calcCCC(gt_A, gt_A)
                
                
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
                    G_path = os.path.join(curDir+model_save_dir, '{}G-best-{}.ckpt'.format(additionName,i))
                    D_path = os.path.join(curDir+model_save_dir, '{}D-best-{}.ckpt'.format(additionName,i))
                    
                    #G_path = os.path.join(curDir+model_save_dir, '{}{}-G-adl-best.ckpt'.format(i+1,additionName))
                    #D_path = os.path.join(curDir+model_save_dir, '{}{}-D-adl-best.ckpt'.format(i+1,additionName))
                    
                    if multi_gpu :
                        torch.save(G.module.state_dict(), G_path)
                        torch.save(D.module.state_dict(), D_path)
                    else : 
                        torch.save(G.state_dict(), G_path)
                        torch.save(D.state_dict(), D_path)
                
                
                print('Best, MSEA : '+str(lMSA)+', CORA : '+str(lCRA)+', CCCA : '+str(lCCA)+', ICCA : '+str(lICA)+ ', MSEV : ' +str(lMSV)+ ', CORV : ' +str(lCRV)+', CCCV : '+str(lCCV) +', ICCV : '+str(lICV)+', Total : '+str(total))
                
                print('MSEV : ',mseV, ', CORV : ',corV,', CCCV : ',cccV,', ICCV : ',iccV)
                print('MSEA : ',mseA, ', CORA : ',corA,', CCCA : ',cccA,', ICCA : ',iccA)
                
                f = open(err_file,'a')
                res = 'MSEV : '+str(mseV)+ ', CORV : ' +str(corV)+', CCCV : '+str(cccV) +', ICCV : '+str(iccV)+' \n '
                f.write(res) 
                res = 'MSEA : '+str(mseA)+ ', CORA : '+str(corA) +', CCCA : '+str(cccA) +', ICCA : '+str(iccA)+' \n '
                f.write(res)
                res = 'Best, MSEA : '+str(lMSA)+', CORA : '+str(lCRA)+', CCCA : '+str(lCCA)+', ICCA : '+str(lICA)+ ', MSEV : ' +str(lMSV)+ ', CORV : ' +str(lCRV)+', CCCV : '+str(lCCV) +', ICCV : '+str(lICV)+', Total : '+str(total)+' \n '
                f.write(res)
                 
                f.close()

    print('Best val Acc: {:4f}'.format(lowest_loss)) 
    pass





def do_test(): #training g and d on standard l2 loss
    #stripped : 
    
    '''
    onlyAdversaryLoss = False  # to see the effect of adversary only loss
    normalizeVA = False #to normalize the VA, seems no substantial impact 
    tryReduced = True #to reduce the dataset
    toAlign = False #we don't need to align the data
    
    onlyAdversaryLoss = False
    normalizeVA = False
    tryReduced = True
    
    '''
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    split = args.split
    isSewa = args.sewa 
    
    dataset = args.dataset
    
    toLoadModel = True
    
    #includeVal = False
    #resume_iters=27
    
    
    #includeVal = False
    #resume_iters=27
    
    includeVal = True
    resume_iters=7
    
    use_skip = args.useSkip
    useLatent = args.useLatent
    tryDenoise = args.tryDenoise
    addLoss = args.addLoss
    useWeight = args.useWeightNormalization 
    
    singleTask = args.singleTask 
    trainQuadrant = args.trainQuadrant
    alterQuadrant = args.alterQuadrant
    addZ = args.addZ
    addS = args.addS
    
    #curDir = "/home/deckyal/eclipse-workspace/FaceTracking/"
    c_dim=2
    c2_dim=8
    image_size=128
    g_conv_dim=16
    d_conv_dim=32
    g_repeat_num=6
    d_repeat_num=6
    lambda_cls=1
    lambda_rec=10
    lambda_gp=10
    inputC = 3#input channel for discriminator 
    
    
    # Training configuration.
    batch_size=500#500#50#40#70#20 #, help='mini-batch size')
    num_iters=200000 #, help='number of total iterations for training D')
    num_iters_decay=100000 #, help='number of iterations for decaying lr')
    g_lr=0.0001 #, help='learning rate for G')
    d_lr=0.0001 #, help='learning rate for D')
    n_critic=5 #, help='number of D updates per each G update')
    beta1=0.5 #, help='beta1 for Adam optimizer')
    beta2=0.999 #, help='beta2 for Adam optimizer')
    #selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'] 
    #', '--list', nargs='+', help='selected attributes for the CelebA dataset',default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    
    isVideo = False
    seq_length = 2
    
    # Test configuration.
    test_iters=200000 #, help='test model from this step')

    # Miscellaneous.
    num_workers=1

    log_dir='stargan/logs'
    model_save_dir='stargan/models'
    sample_dir='stargan/samples-g_adl'
    result_dir='stargan/results'

    # Step size.
    log_step=20
    sample_step=5#1000
    model_save_step=2
    lr_update_step=100#1000
    
    #model_save_step=10000
    #lr_update_step=1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    multi_gpu = False
    testSplit = split
    print("Test split " , testSplit)
    nSplit = 5
    listSplit = []
    for i in range(nSplit):
        if i!=testSplit : 
            listSplit.append(i)
    print(listSplit)
    
    if dataset == 1 : 
        d_name = 'AFEW-VA-Fixed'
        dbType = 0
        additionName = "AC"+str(split)+"-"
    else : 
        if not isSewa : 
            d_name = 'AFEW-VA-Fixed'
            dbType = 0
            additionName = "AF"+str(split)+"-"
        else : 
            d_name = 'SEWA'
            dbType = 1
            additionName = "SW"+str(split)+"-"
    
    
    if singleTask : 
        additionName+='ST-'    
    if addLoss : 
        additionName+='AL-'
    if useWeight : 
        additionName+='W-'
    if addZ : 
        additionName+='Z-'
    if addS : 
        additionName+='S-'
        inputC = 4
        
    if trainQuadrant : 
        if alterQuadrant : 
            additionName+="QDAL-"
            c_dim = 1
        else :  
            additionName+="QD-"
            c_dim = 4
    if tryDenoise :
        additionName+="Den-"
    
    
    if includeVal : 
        additionName+='-IV-'
    
    transform =transforms.Compose([
            #transforms.Resize((image_size,image_size)),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    
    #Build model 
    """Create a generator and a discriminator.""" 
    G = GeneratorMZ(g_conv_dim, 0, g_repeat_num,use_skip,useLatent)
    #if not singleTask : 
    D = DiscriminatorMZ(image_size, d_conv_dim, c_dim, d_repeat_num,inputC=inputC)
    '''
    else : 
        D = DiscriminatorMST(image_size, d_conv_dim, c_dim, d_repeat_num,asDiscriminator = True)'''
        
        
    #G = Generator(g_conv_dim, 0, g_repeat_num,use_skip)
    #D = Discriminator(image_size, d_conv_dim, c_dim, d_repeat_num)
    
    print_network(G, 'G')
    print_network(D, 'D')
    
    if toLoadModel :
        print('Loading models from iterations : ',resume_iters) 
        G_path = os.path.join(curDir+model_save_dir, '{}G-{}.ckpt'.format(additionName,resume_iters))
        D_path = os.path.join(curDir+model_save_dir, '{}D-{}.ckpt'.format(additionName,resume_iters))
        print('loading ',G_path)
        print('loading ',D_path)
        G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    else : 
        print('Initiating models')
        G.apply(weights_init_uniform_rule)
        D.apply(weights_init_uniform_rule)
    
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])
    
    G.to(device)
    D.to(device)
    #Deep copy the model_ft 
    
    
    
    img,gt,ln,snd = test(['Test'])
    #print(len(img[0]),len(gt[0]))
   # print(img[0][-1],gt[0][-1])
   # print(ln[0])
   
    tgtFolder = "/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/AffectChallenge/"+additionName+"/"
    
    if not os.path.isdir(tgtFolder): 
        os.makedirs(tgtFolder)
    
    augment = False
    returnNoisy = False
    transform =transforms.Compose([
        #transforms.Resize((image_size,image_size)),
        #transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if augment : 
        flip = RandomHorizontalFlip(1)
        rot = RandomRotation(45)
        occ = Occlusion(1)
        rc = RandomResizedCrop(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
        
    if returnNoisy : 
        gn = GeneralNoise(1)
        cc = Occlusion(1)
    
            
    batch_size = 1
    a = 0
    b = 1
    
    listValO = []
    listAroO = []
    
    listValL = []
    listAroL = []
    
    tvo = [];tao=[];tvl = []; tal = [];
    anyDiffer = False
    
    
    G.eval()    
    D.eval()
    
    
    '''
    for i in range(len(img)):
        if len(img[i]) < 1:
            print('LESS THAN') 
            print(len(img[i]),img[i])
        else : 
            print(img[i][0])
    
    exit(0)
    
    '''
    for i in range(len(img)): 
        images,gts,lns,snds = img[i],gt[i],ln[i],snd[i]
        
        print(lns,images[0])
        print('**********')
        fname = os.path.basename(lns[0])
        tgtFile = tgtFolder+fname
        
        if os.path.isfile(tgtFile): 
            continue
        
        f = open(tgtFile,'w+')
        f.write("valence,arousal\n")
        f.close()
        
        print('target : ',tgtFile)
        
        for image,va,name,sound in zip(images, gts, lns, snds): 
            #print(name,image,va,sound)
            #exit(0)
            '''
            if trainQuadrant: 
                rinputs, rlabels,rldmrk = data[0],data[5],data[2]
            else : 
                rinputs, rlabels,rldmrk = data[0],data[1],data[2]
                
            if addS : 
                sound = data[-1].cuda()
            else : 
                sound = None
            
            G.eval()    
            D.eval()
            
            inputs = rinputs.cuda()#to(device) 
            labels = rlabels.cuda()#to(device)'''
            
            
            '''image2 = cv2.imread(image)
            cv2.imshow(fname,image2)
            cv2.waitKey(100)'''
    
            tImage = Image.open(image).convert("RGB")
            #tImage.show()
            if augment : 
                if returnNoisy :
                    sel = np.random.randint(0,3) #Skip occlusion as noise
                else : 
                    sel = np.random.randint(0,4)
                #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
                if sel == 0 : 
                    pass
                elif sel == 1 : 
                    tImage = flip(tImage)
                elif sel == 2 : 
                    tImage = rot(tImage)
                elif sel == 3 : 
                    tImage = occ(tImage)
                    
                #random crop
                if (np.random.randint(1,3)%2==0) :
                    tImage= rc(tImage)
                
            if returnNoisy :
                nImage = tImage.copy()
            
                if (np.random.randint(1,3)%2==0): 
                    sel_n = np.random.randint(1,7)
                    if sel_n > 5 : 
                        nImage = occ(nImage)
                    else :
                        nImage = gn(nImage,sel_n,np.random.randint(0,3))
                    
            tImage = transform(tImage)
            if returnNoisy : 
                nImage = transform(nImage)
            
            
            
            the_image = torch.unsqueeze(tImage,0)
            the_label = torch.unsqueeze(torch.FloatTensor(va),0)
            if returnNoisy : 
                the_n_image = torch.unsqueeze(nImage,0)
            the_sound = torch.unsqueeze(torch.tensor(torch.FloatTensor(sound)),0)
            the_name = name 
            ######
            
            inputs = the_image.cuda()#to(device) 
            labels = the_label.cuda()#to(device)
            soundi = the_sound.cuda()
            
            with torch.set_grad_enabled(False) : 
                
                if addZ : 
                    inputsM,z = G(inputs,returnInter=True)
                else : 
                    inputsM,z = G(inputs),None
                #print('s',sound.shape, z.shape)
                
                if addS : 
                    pass
                else : 
                    soundi = None
                
                _, outputs = D(inputsM,s=soundi,z=z)
                
                #print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs[:,0].shape[0],outputs.shape)
                shape = outputs[:,0].shape[0]
            
            
            
            print('output : ',outputs)
            res = outputs.detach().cpu().numpy()[0]#torch.tensor([-99,-99])
            
            
            
            #print(the_image.shape, the_label.shape,the_sound.shape,the_name)
            #print(the_label)
            #print(the_sound)
            
            
            
            f = open(tgtFile,'a')
            f.write("{},{}\n".format(res[0],res[1]))
            f.close()
            
            #return
    
        
    
            
            if shape != batch_size : #in case the batch size is differ, usually at end of iter
                anyDiffer = True 
                print('differ')
                tvo.append(outputs[:,a].detach().cpu())
                tao.append(outputs[:,b].detach().cpu())
                
                tvl.append(labels[:,a].detach().cpu())
                tal.append(labels[:,b].detach().cpu())
            else :
                print('equal')
                listValO.append(outputs[:,a].detach().cpu())
                listAroO.append(outputs[:,b].detach().cpu())
                
                listValL.append(labels[:,a].detach().cpu())
                listAroL.append(labels[:,b].detach().cpu())
                
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
        
        cccV = calcCCC(est_V, gt_V)
        cccA = calcCCC(est_A, gt_A)
        
        iccV2 = calcCCC(gt_V, gt_V)
        iccA2 = calcCCC(gt_A, gt_A)
        
        print('MSEV : ',mseV, ', CORV : ',corV,', CCCV : ',cccV,', ICCV : ',iccV)
        print('MSEA : ',mseA, ', CORA : ',corA,', CCCA : ',cccA,', ICCA : ',iccA)

def train_w_g_adl_old(): #training g and d on standard l2 loss
   
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    split = args.split
    isSewa = args.sewa 
    
    toLoadModel = True
    resume_iters=179
    
    use_skip = args.useSkip
    useLatent = args.useLatent
    tryDenoise = args.tryDenoise
    addLoss = args.addLoss
    useWeight = args.useWeightNormalization 
    
    singleTask = args.singleTask 
    trainQuadrant = args.trainQuadrant
    alterQuadrant = args.alterQuadrant
    
    #curDir = "/home/deckyal/eclipse-workspace/FaceTracking/"
    c_dim=2
    c2_dim=8
    image_size=128
    g_conv_dim=64
    d_conv_dim=64
    g_repeat_num=6
    d_repeat_num=6
    lambda_cls=1
    lambda_rec=10
    lambda_gp=10
    
    
    # Training configuration.
    batch_size=20#50#40#70#20 #, help='mini-batch size')
    
    isVideo = False
    seq_length = 2
    
    
    # Miscellaneous.
    num_workers=1

    log_dir='stargan/logs'
    model_save_dir='stargan/models'
    sample_dir='stargan/samples-g_adl'
    result_dir='stargan/results'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        d_name = 'AFEW-VA-Fixed'
        dbType = 0
        additionName = "AF"+str(split)+"-"
    else : 
        d_name = 'SEWA'
        dbType = 1
        additionName = "SW"+str(split)+"-"
    
    
    if singleTask : 
        additionName+='ST-'    
    if addLoss : 
        additionName+='AL-'
    if useWeight : 
        additionName+='W-'
    if trainQuadrant : 
        if alterQuadrant : 
            additionName+="QDAL-"
            c_dim = 1
        else :  
            additionName+="QD-"
            c_dim = 4
    if tryDenoise :
        additionName+="Den-"
    
    save_name = additionName+str(testSplit)
    err_file = curDir+save_name+".txt"
    
    
    print('err file : ',err_file)
    
    transform =transforms.Compose([
            #transforms.Resize((image_size,image_size)),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    VD = SEWAFEWReduced([d_name], None, True, image_size, transform, False, False, 1,split=True, nSplit = nSplit,listSplit=[testSplit]
                ,isVideo=isVideo, seqLength = seq_length, returnQuadrant=trainQuadrant, returnNoisy = tryDenoise,dbType = dbType)
    #VD = AFEWVA([d_name], None, True, image_size, transform, False, False, 1,split=True, nSplit = nSplit,listSplit=[testSplit]
    #            ,isVideo=isVideo, seqLength = seq_length, returnNoisy = tryDenoise,dbType = dbType)
    dataloaderV = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = False)
   
    #Build model 
    """Create a generator and a discriminator.""" 
    G = GeneratorM(g_conv_dim, 0, g_repeat_num,use_skip,useLatent)
    if not singleTask : 
        D = DiscriminatorM(image_size, d_conv_dim, c_dim, d_repeat_num)
    else : 
        D = DiscriminatorMST(image_size, d_conv_dim, c_dim, d_repeat_num,asDiscriminator = True)
        
    #G = Generator(g_conv_dim, 0, g_repeat_num,use_skip)
    #D = Discriminator(image_size, d_conv_dim, c_dim, d_repeat_num)
    
    print_network(G, 'G')
    print_network(D, 'D')
    
    if toLoadModel :
        print('Loading models from iterations : ',resume_iters) 
        G_path = os.path.join(curDir+model_save_dir, '{}G-{}.ckpt'.format(additionName,resume_iters))
        D_path = os.path.join(curDir+model_save_dir, '{}D-{}.ckpt'.format(additionName,resume_iters))
        print('loading ',G_path)
        print('loading ',D_path)
        G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        
    G.to(device)
    D.to(device)
    #print(D.linear41.weight.data)

    if trainQuadrant : 
        a = 0
        b = 0
    else : 
        a = 0
        b = 1
    
    if True : 
    
        listValO = []
        listAroO = []
        
        listValL = []
        listAroL = []
        
        tvo = [];tao=[];tvl = []; tal = [];
        anyDiffer = False
       
        for x,(data) in enumerate(dataloaderV,0) :
            
            if trainQuadrant: 
                rinputs, rlabels,rldmrk = data[0],data[5],data[2]
            else : 
                rinputs, rlabels,rldmrk = data[0],data[1],data[2]
            
            G.eval()    
            D.eval()
            
            inputs = rinputs.cuda()#to(device) 
            labels = rlabels.cuda()#to(device)
            
            with torch.set_grad_enabled(False) : 
                
                inputsM = G(inputs)
                _, outputs = D(inputsM)
                
                if trainQuadrant: 
                    if alterQuadrant :
                        outputs = torch.round(outputs) 
                    else : 
                        _,outputs = torch.max(outputs,1)
                
                if trainQuadrant : 
                    print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs.shape)
                else : 
                    print(x,',',int(truediv(len(VD),batch_size)),outputs[:2], labels[:2],outputs[:,0].shape[0],outputs.shape)
                #print(outputs.shape)
                
                if not trainQuadrant : 
                    shape = outputs[:,0].shape[0]
                else : 
                    shape = outputs.shape[0]
                
                if shape != batch_size : #in case the batch size is differ, usually at end of iter
                    anyDiffer = True 
                    print('differ')
                    if trainQuadrant: 
                        tvo.append(outputs.detach().cpu())
                        tao.append(outputs.detach().cpu())
                        
                        tvl.append(labels.detach().cpu())
                        tal.append(labels.detach().cpu())
                    else : 
                        tvo.append(outputs[:,a].detach().cpu())
                        tao.append(outputs[:,b].detach().cpu())
                        
                        tvl.append(labels[:,a].detach().cpu())
                        tal.append(labels[:,b].detach().cpu())
                else :
                    print('equal')
                    if trainQuadrant : 
                        listValO.append(outputs.detach().cpu())
                        listAroO.append(outputs.detach().cpu())
                        
                        listValL.append(labels.detach().cpu())
                        listAroL.append(labels.detach().cpu())
                    else : 
                        listValO.append(outputs[:,a].detach().cpu())
                        listAroO.append(outputs[:,b].detach().cpu())
                        
                        listValL.append(labels[:,a].detach().cpu())
                        listAroL.append(labels[:,b].detach().cpu())
                        
                
        
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
        
        cccV = calcCCC(est_V, gt_V)
        cccA = calcCCC(est_A, gt_A)
        
        iccV2 = calcCCC(gt_V, gt_V)
        iccA2 = calcCCC(gt_A, gt_A)
        
        
        print('MSEV : ',mseV, ', CORV : ',corV,', CCCV : ',cccV,', ICCV : ',iccV)
        print('MSEA : ',mseA, ', CORA : ',corA,', CCCA : ',cccA,', ICCA : ',iccA)
        
    pass





def test(listMode = []):
    
    import shutil
    import file_walker
    import csv
    
    fill = True
    step = 1

    data_list = ["AffectChallenge"]
    
    rootDir = "/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data"
    curDir = rootDir +"/"#/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/"
    
    list_gt = []
    list_labels_tE = []
    list_labels_sound = []
    
    counter_image = 0
    
    for data in data_list : 
        print(("Opening "+data))
        
        for mode in listMode :
            fullDir = curDir +data+"/images/VA/"+mode
            fullDirLbl = curDir +data+"/labels/VA/"+mode
            fullDirLblSnd = curDir +data+"/sounds/VA/"+mode
            
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
                list_sndsx.append(np.array(row[1:]).astype(np.float))
        list_snds.append(list_sndsx)
           
        
        list_labelsE.append(np.array(x))
        list_labelsEN.append(dataFile)
        #break
        
    t_l_imgs = []
    t_l_gtE = []
    t_l_snd = []
    t_list_gtE_names = []
    
    countVideo=0
    countGT = 0
    
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
        
        print('emotion ',list_labelsEN[i],' len : ',length)
        
        
        countVideo+=len(list_gt[i])
        countGT+=len(list_labelsE[i])
        
        realIndex = 0  
        
        for j in range(0,length,step): #for number of data #n_skip is usefull for video data
            
            if fill : 
                #additional check
                fnameIndex = int(os.path.basename(list_gt[i][realIndex]).split('.')[0])-1
                if fnameIndex != j :
                    list_images.append(list_gt[i][realIndex])
                else : 
                    list_images.append(list_gt[i][realIndex])
                    
                    if realIndex < len(list_gt[i])-1:
                        realIndex+=1 
                    
                list_gtE_names.append(list_labelsEN[i])
                list_ground_truthE[indexer] = np.array(list_labelsE[i][j]).flatten('F')
                if j >= len(list_snds[i]): 
                    list_sounds[indexer] = list_snds[i][-1]
                else : 
                    list_sounds[indexer] = list_snds[i][j]
                
            else :
                fnameIndex = int(os.path.basename(list_gt[i][j]).split('.')[0])-1
                list_images.append(list_gt[i][j])
                #print(list_labelsEN)
                list_gtE_names.append(list_labelsEN[i])
                #print(list_labelsEN[i])
                list_ground_truthE[indexer] = np.array(list_labelsE[i][fnameIndex]).flatten('F')
                list_sounds[indexer] = list_snd[i][fnameIndex]
            
            indexer += 1
        
        t_l_snd.append(list_sounds)
        t_l_imgs.append(list_images)
        t_l_gtE.append(list_ground_truthE)
        t_list_gtE_names.append(list_gtE_names)
    
    #print(len(t_l_imgs),len(t_l_snd))
    #print(len(t_l_imgs[0]),len(t_l_imgs[1]))
    
    return [t_l_imgs,t_l_gtE,t_list_gtE_names,t_l_snd]

def test_data():
    
    img,gt,ln,snd = test(['tmp'])
    #print(len(img[0]),len(gt[0]))
   # print(img[0][-1],gt[0][-1])
   # print(ln[0])
   
    tgtFolder = "/home/deckyal/eclipse-workspace/DeepModel/src/MMTVA/data/AffectChallenge/res/"
    augment = False
    returnNoisy = False
    transform =transforms.Compose([
        #transforms.Resize((image_size,image_size)),
        #transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if augment : 
        flip = RandomHorizontalFlip(1)
        rot = RandomRotation(45)
        occ = Occlusion(1)
        rc = RandomResizedCrop(size = self.imageSize,scale = (0.5,1), ratio = (0.5, 1.5))
        
    if returnNoisy : 
        gn = GeneralNoise(1)
        cc = Occlusion(1)
    
    for i in range(len(img)): 
        images,gts,lns,snds = img[i],gt[i],ln[i],snd[i]
        
        
        fname = os.path.basename(lns[0])
        tgtFile = tgtFolder+fname
        
        f = open(tgtFile,'w+')
        f.write("valence,arousal\n")
        f.close()
        
        print('target : ',tgtFile)
        
        for image,va,name,sound in zip(images, gts, lns, snds): 
            #print(name,image,va,sound)
            #exit(0)
            
            image2 = cv2.imread(image)
            cv2.imshow(fname,image2)
            cv2.waitKey(0)
    
            tImage = Image.open(image).convert("RGB")
            #tImage.show()
            if augment : 
                if returnNoisy :
                    sel = np.random.randint(0,3) #Skip occlusion as noise
                else : 
                    sel = np.random.randint(0,4)
                #0 : neutral, 1 : horizontal flip, 2:random rotation, 3:occlusion
                if sel == 0 : 
                    pass
                elif sel == 1 : 
                    tImage = flip(tImage)
                elif sel == 2 : 
                    tImage = rot(tImage)
                elif sel == 3 : 
                    tImage = occ(tImage)
                    
                #random crop
                if (np.random.randint(1,3)%2==0) :
                    tImage= rc(tImage)
                
            if returnNoisy :
                nImage = tImage.copy()
            
                if (np.random.randint(1,3)%2==0): 
                    sel_n = np.random.randint(1,7)
                    if sel_n > 5 : 
                        nImage = occ(nImage)
                    else :
                        nImage = gn(nImage,sel_n,np.random.randint(0,3))
                    
            tImage = transform(tImage)
            if returnNoisy : 
                nImage = transform(nImage)
            
            
            
            the_image = torch.unsqueeze(tImage,0)
            the_label = torch.unsqueeze(torch.FloatTensor(va),0)
            if returnNoisy : 
                the_n_image = torch.unsqueeze(nImage,0)
            the_sound = torch.unsqueeze(torch.tensor(torch.FloatTensor(sound)),0)
            the_name = name 
            
            
            res = torch.tensor([-99,-99])
            
            
            
            #print(the_image.shape, the_label.shape,the_sound.shape,the_name)
            #print(the_label)
            print(the_sound)
            
            
            
            f = open(tgtFile,'a')
            f.write("{},{}\n".format(res[0],res[1]))
            f.close()
            
            
            continue
            #return
            
            #l_imgs.append(tImage); 
            #l_VA.append(torch.FloatTensor(labelE)); 
            #l_SD.append(torch.FloatTensor(ls))
            
            if self.returnNoisy : 
                l_nimgs.append(nImage)
            
            l_nc.append(ln)
            
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
        
            
    
    
    
    

if __name__ == '__main__':
    #train_standard()
    #train_only_disc()
    #train_w_g_std()
    #train_w_g_adl()
    #test_data()
    do_test()
    
