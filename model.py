import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operator import truediv


class CombiningBottleNeckO(nn.Module):
    def __init__(self, dim_in, dim_out,toCombine = False):
        '''
        input : 
            1. the z of G
            2. the prev results 
            3. the rought estimation from D
        
        combinations : 
            1. series of 2d conv
            2. linear with previous 
        
        '''
        super(CombiningBottleNeckO, self).__init__()
        
        '''#from generator 
        self.conv1 = nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.i1 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        
        
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        
        #from discriminator 
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        ######################'''
        
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.01)
        
        #8x8, 32/4 = 8 . 32x32x256
        self.conv1 = nn.Conv2d(dim_in, 512, kernel_size=8, stride=4, padding=2, bias=False)
        
        #4x4, 8/2 = 4
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=6, stride=2, padding=1, bias=False)
        
        #4x4, 4/2 = 4
        
        #3x3, 4/2 = 2~1
        self.conv3 = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.toCombine = toCombine
        
        if self.toCombine :
            self.linear11 = nn.Linear(2048, 512)
        else :  
            self.linear1 = nn.Linear(2048, 1024)
            
        self.linear2 = nn.Linear(1024, dim_out)
                
        '''self.main = nn.Sequential(
            
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        
        
        self.conv1 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)'''

    def forward(self, x,y = None):
        
        #print(x.shape)
        
        x1 = self.lrelu(self.conv1(x))
        
        #print(x1.shape)
        
        x2 = self.lrelu(self.conv2(x1))
        
        #print(x2.shape)
        
        x3 = self.lrelu(self.conv3(x2))
        
        #print(x3.shape)
        
        x3 = x3.view(x3.size(0),-1)
        
        #print(x3.shape)
        
        #x4 = self.relu(self.linear1(x3))
        if not self.toCombine : 
            x4 = self.lrelu(self.linear1(x3))
        else : 
            x41 = self.lrelu(self.linear11(x3))
            #x42 = torch.tensor(y)
            x42 = y.view(y.size(0),1)
            
            #print('x42 : ',x42.shape)
            #1, 1, x.size(2), x.size(3)
            
            #print(x41.size(1))
            x42 = x42.repeat(1,x41.size(1)).float()
            
            #print(x41.shape,x42.shape)
            x4 = torch.cat((x41,x42),1)
        
        x5 = self.linear2(x4)
        #print(x4.shape)
        return x5
        #return x + self.main(x)



class CombiningBottleNeck(nn.Module):
    def __init__(self, dim_in, dim_out,toCombine = False):
        '''
        input : 
            1. the z of G
            2. the prev results 
            3. the rought estimation from D
        
        combinations : 
            1. series of 2d conv
            2. linear with previous 
        
        '''
        super(CombiningBottleNeck, self).__init__()
        
        '''#from generator 
        self.conv1 = nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.i1 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        
        
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        
        #from discriminator 
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        ######################'''
        
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.01)
        self.selu = nn.SELU()
        
        #8x8, 32/4 = 8 . 32x32x256
        self.conv1 = nn.Conv2d(dim_in, 512, kernel_size=8, stride=4, padding=2, bias=False)
        
        #4x4, 8/2 = 4
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=6, stride=2, padding=1, bias=False)
        
        #4x4, 4/2 = 4
        
        #3x3, 4/2 = 2~1
        self.conv3 = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.toCombine = toCombine
        
        if self.toCombine :
            self.linear11 = nn.Linear(2048, 512)
        else :  
            self.linear1 = nn.Linear(2048, 1024)
            
        self.linear2 = nn.Linear(1024, dim_out)
                
        '''self.main = nn.Sequential(
            
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        
        
        self.conv1 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)'''

    def forward(self, x,y = None):
        
        #print(x.shape)
        
        x1 = self.lrelu(self.conv1(x))
        
        #print(x1.shape)
        
        x2 = self.lrelu(self.conv2(x1))
        
        #print(x2.shape)
        
        x3 = self.lrelu(self.conv3(x2))
        
        #print(x3.shape)
        
        x3 = x3.view(x3.size(0),-1)
        
        #print(x3.shape)
        
        #x4 = self.relu(self.linear1(x3))
        if not self.toCombine : 
            x4 = self.lrelu(self.linear1(x3))
        else : 
            x41 = self.lrelu(self.linear11(x3))
            #x42 = torch.tensor(y)
            x42 = y.view(y.size(0),1)
            
            #print('x42 : ',x42.shape)
            #1, 1, x.size(2), x.size(3)
            
            #print(x41.size(1))
            x42 = x42.repeat(1,x41.size(1)).float()
            
            #print(x41.shape,x42.shape)
            x4 = torch.cat((x41,x42),1)
        
        x5 = self.linear2(x4)
        #print(x4.shape)
        return x5
        #return x + self.main(x)



class CombiningBottleNeckSeq(nn.Module):
    def __init__(self, dim_in, dim_out,toCombine = False,batch_length = 10, seq_length = 2, withPrev = False,reduced = False,
                 lstmNeuron = 512):
        '''
        input : 
            1. the z of G
            2. the prev results 
            3. the rought estimation from D
        
        combinations : 
            1. series of 2d conv
            2. linear with previous 
        
        '''
        super(CombiningBottleNeckSeq, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.01)
        
        self.btl = batch_length
        self.sql = seq_length 
        self.dim_out = dim_out
        
        self.lstmNeuron = lstmNeuron 
        
        #8x8, 32/4 = 8 . 32x32x256
        self.conv1 = nn.Conv2d(dim_in, 512, kernel_size=8, stride=4, padding=2, bias=False)
        
        #4x4, 8/2 = 4
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=6, stride=2, padding=1, bias=False)
        
        #4x4, 4/2 = 4
        
        #3x3, 4/2 = 2~1
        self.conv3 = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.toCombine = toCombine
        
        if self.toCombine :
            self.linear11 = nn.Linear(2048, 512)
        else :  
            self.linear1 = nn.Linear(2048, 1024)
        
        self.withPrev = withPrev
        
        if self.withPrev : #this is to use the prev result 
            self.linear2p =  nn.Linear(1024, 512)
            
        
        self.linear2 = nn.LSTM(1024, self.lstmNeuron)
        
        self.linear3 = nn.Linear(self.lstmNeuron, int(truediv(self.lstmNeuron,2)))
        self.linear4 = nn.Linear(int(truediv(self.lstmNeuron,2)), dim_out)
        
        self.initialize(self.btl)
                
    def forward(self, x,y = None, y_prev = None):
        
        #print(x.shape)
        x1 = self.lrelu(self.conv1(x))
        
        #print(x1.shape)
        x2 = self.lrelu(self.conv2(x1))
        
        #print(x2.shape)
        x3 = self.lrelu(self.conv3(x2))
        
        #print(x3.shape)
        x3 = x3.view(x3.size(0),-1)
        
        #print(x3.shape)
        
        #x4 = self.relu(self.linear1(x3))
        if not self.toCombine : 
            x4 = self.relu(self.linear1(x3))
        else : 
            x41 = self.relu(self.linear11(x3))
            #x42 = torch.tensor(y)
            x42 = y.view(y.size(0),1)
            
            #print('x42 : ',x42.shape)
            #1, 1, x.size(2), x.size(3)
            
            #print(x41.size(1))
            x42 = x42.repeat(1,x41.size(1)).float()
            
            #print(x41.shape,x42.shape)
            x4 = torch.cat((x41,x42),1)
            
        if self.withPrev: # y_prev is not None :
            x4 = self.relu(self.linear2p(x4))
             
            x43 = y_prev.view(y_prev.size(0),1)
            x43 = x43.repeat(1,x4.size(1)).float()
            
            #print(x43.shape,x4.shape,'shape')
            
            x4 = torch.cat((x43,x4),1)
        
        #The input dimensions are (seq_len, batch, input_size).
        #print(x4.shape,self.linear2_hdn[0].shape,self.linear2_hdn[1].shape)
        
        x4 = torch.unsqueeze(x4,0)
        #print('x4 shape ',x4.shape,self.linear2_hdn[0].shape,self.linear2_hdn[1].shape)
        
        x4,self.linear2_hdn = self.linear2(x4,self.linear2_hdn)
        
        x5 = self.lrelu(self.linear3(torch.squeeze(x4,0)))
        
        x6 = self.linear4(x5)
        
        #print(x4.shape)
        return x6
        #return x + self.main(x)


    def initialize(self,batch_size = 10):
        #self.linear2Hidden = (torch.randn(1, 1, 3),
        #torch.randn(1, 1, 3))
        self.linear2_hdn = (torch.zeros(1, batch_size, self.lstmNeuron).cuda(),torch.zeros(1, batch_size, self.lstmNeuron).cuda())
        








class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, use_skip = True):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        
        self.use_skip = use_skip

    def forward(self, x):
        if self.use_skip : 
            return x + self.main(x)
        else : 
            return self.main(x)





class GeneratorM(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, use_skip = True, compressLatent = False):
        super(GeneratorM, self).__init__()
        
        self.compressLatent = compressLatent
        
        self.conv1 = nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.i1 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        
        # Down-sampling layers.
        curr_dim = conv_dim
        
        self.conv21 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.i21 = nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)
        curr_dim = curr_dim * 2
        
        self.conv22 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.i22 = nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)
        curr_dim = curr_dim * 2
        
        #Bottleneck layers
        self.conv31 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = True)
        self.conv32 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = True)
        
        #self.conv34 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = use_skip)
        #Latent space if required 
        
        
        if self.compressLatent : 
            self.linear331 = nn.Linear(262144, 512)
            self.linear332 = nn.Linear(512, 262144)
        else : 
            self.conv33 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = use_skip)    
        
        self.conv34 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = True)
        self.conv35 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = True)
        #self.conv36 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = True)
        
        
        #Upsampling layers
        self.conv41 = nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.i41 = nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)
        curr_dim = curr_dim // 2
        
        self.conv42 = nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.i42 = nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)
        curr_dim = curr_dim // 2
        
        #Last Layer
        self.conv51 = nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()
        
        ##################################################################################
        
        '''
        layers = [] 
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num/2):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = True))
        
        layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = False))    
        
        for i in range(repeat_num/2):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = True))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)
        '''

    def forward(self, x, c = None,returnInter = False):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        if c is not None : 
            c = c.view(c.size(0), c.size(1), 1, 1)
            c = c.repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat([x, c], dim=1)
        
        
        x1 = self.relu(self.i1(self.conv1(x)))
        
        
        x21 = self.relu(self.i21(self.conv21(x1)))
        x22 = self.relu(self.i22(self.conv22(x21)))
        
        
        x31 = self.conv31(x22)
        x32 = self.conv32(x31)
        
        
        if self.compressLatent : 
            z = self.relu(self.linear331(x32.view(x32.size(0), -1)))
            x33 = self.relu(self.linear332(z)).view(z.size(0), 256,32,32)
            #print(' z shape : ',z.shape)
        else : 
            x33 = self.conv33(x32) 
        
        
        x34 = self.conv34(x33)
        x35 = self.conv35(x34)
        #x37 = self.conv37(x36) 
        
        
        x41 = self.relu(self.i41(self.conv41(x35)))
        x42 = self.relu(self.i42(self.conv42(x41)))
        
        x51 = self.tanh(self.conv51(x42))
        
        #return self.main(x)
        if returnInter : 
            return x51, x33
        else : 
            return x51
    




class DiscriminatorM112(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=112, conv_dim=64, c_dim=5, repeat_num=6):
        super(DiscriminatorM112, self).__init__()
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)
        
        self.conv1 =  nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)
        
        
        curr_dim = conv_dim
        
        self.conv21 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv22 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv23 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv24 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv25 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        
        kernel_size = int(image_size / np.power(2, repeat_num))
        
        self.conv31 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv32 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
        self.linear41 = nn.Linear(4, c_dim)
        
        ################################################################
        
    def forward(self, x,printH = False):
        
        
        x1 = self.lrelu(self.conv1(x))
        
        x21 = self.lrelu(self.conv21(x1))
        x22 = self.lrelu(self.conv22(x21))
        x23 = self.lrelu(self.conv23(x22))
        x24 = self.lrelu(self.conv24(x23))
        h = self.lrelu(self.conv25(x24))
        
        #h = self.main(x)
        
        out_src = self.conv31(h)
        tmp = self.conv32(h)
        
        if printH and False : 
            print('tmp : ',tmp[:2])
        
        out_cls = tmp
            
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))



class GeneratorMZ(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, use_skip = True, compressLatent = False):
        super(GeneratorMZ, self).__init__()
        
        self.compressLatent = compressLatent
        
        self.conv1 = nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.i1 = nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        
        # Down-sampling layers.
        curr_dim = conv_dim
        
        self.conv21 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.i21 = nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)
        curr_dim = curr_dim * 2
        
        self.conv22 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.i22 = nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True)
        curr_dim = curr_dim * 2
        
        #Bottleneck layers
        self.conv3 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = use_skip)
        
        
        #Upsampling layers
        self.conv41 = nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.i41 = nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)
        curr_dim = curr_dim // 2
        
        self.conv42 = nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.i42 = nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True)
        curr_dim = curr_dim // 2
        
        #Last Layer
        self.conv51 = nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()
        

    def forward(self, x, c = None,returnInter = False):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        if c is not None : 
            c = c.view(c.size(0), c.size(1), 1, 1)
            c = c.repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat([x, c], dim=1)
        
        debug = False
        
        x1 = self.relu(self.i1(self.conv1(x)))
        
        x21 = self.relu(self.i21(self.conv21(x1)))
        x22 = self.relu(self.i22(self.conv22(x21)))
        
        
        '''x31 = self.conv31(x22)
        x32 = self.conv32(x31) #2d latent
        x33 = self.conv33(x32)'''
        x3 = self.conv3(x22) #2d latent
        
        x41 = self.relu(self.i41(self.conv41(x3)))
        x42 = self.relu(self.i42(self.conv42(x41)))
        
        x51 = self.tanh(self.conv51(x42))
        
        if debug :
            print('G-x0',x.shape) 
            print('x1',x1.shape)
            print('x21',x21.shape)
            print('x22',x22.shape)
            '''print('x31',x31.shape)
            print('x32',x32.shape)
            print('x33',x33.shape)'''
            #print('x31',x31.shape)
            print('x3',x3.shape)
            #print('x33',x33.shape)
            
            print('x41',x41.shape)
            print('x42',x42.shape)
            print('x51',x51.shape)
        
        #return self.main(x)
        if returnInter : 
            #return x51, x33
            return x51, x3
        else : 
            return x51
    



class DiscriminatorMZ(nn.Module):
    """Discriminator network with with external z """
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, inputC = 3):
        super(DiscriminatorMZ, self).__init__()
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)
        
        self.conv1 =  nn.Conv2d(inputC, conv_dim, kernel_size=4, stride=2, padding=1)
        
        
        curr_dim = conv_dim
        
        self.conv21 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv22 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv23 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv24 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=2)
        curr_dim = curr_dim * 2
        self.conv25 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        
        kernel_size = int(image_size / np.power(2, repeat_num))
        
        self.conv31 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv32 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
        self.linear1 = nn.Linear(46, 112)
        
        ################################################################
        
    def forward(self, x, s = None, z = None ):
        
        debug = False
        
        if s is not None : 
            s2 = self.tanh(self.linear1(s))
            #print(s2)
            #print(s2.shape)
            s2 = torch.unsqueeze(torch.unsqueeze(s2,1),2)
            #print(s2.shape,s2.size(0))
            #s2 = s2.repeat(1, 1, x.size(2), x.size(3))
            s2 = s2.expand(s2.size(0),1,112,112)
            #print(s2.shape,x.shape)
            #print(x)
            x = torch.cat([x, s2], dim=1)
        
        x1 = self.lrelu(self.conv1(x))
        
        if z is not None : 
            #print('combining')
            x21 = self.lrelu(self.conv21(x1))+z
        else :
            x21 = self.lrelu(self.conv21(x1))
            
        x22 = self.lrelu(self.conv22(x21))
        x23 = self.lrelu(self.conv23(x22))
        
        x24 = self.lrelu(self.conv24(x23))
        h = self.lrelu(self.conv25(x24))
        
        if debug : 
            print('D-x0',x1.shape)
            print('x1',x1.shape)
            print('x21',x21.shape)
            print('x22',x22.shape)
            print('x23',x23.shape)
            print('x24',x24.shape)
            print('xh',h.shape)
            
            #h = self.main(x)
            
        out_src = self.conv31(h)
        tmp = self.conv32(h)
        
        out_cls = tmp
            
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
    

class DiscriminatorM(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(DiscriminatorM, self).__init__()
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)
        
        self.conv1 =  nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)
        
        
        curr_dim = conv_dim
        
        self.conv21 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv22 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv23 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv24 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv25 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        
        kernel_size = int(image_size / np.power(2, repeat_num))
        
        self.conv31 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv32 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
        self.linear41 = nn.Linear(4, c_dim)
        
        ################################################################
        
        '''layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
        self.linear1 = nn.Linear(4, c_dim)'''
        
    def forward(self, x,printH = False):
        
        
        x1 = self.lrelu(self.conv1(x))
        
        x21 = self.lrelu(self.conv21(x1))
        x22 = self.lrelu(self.conv22(x21))
        x23 = self.lrelu(self.conv23(x22))
        x24 = self.lrelu(self.conv24(x23))
        h = self.lrelu(self.conv25(x24))
        
        #h = self.main(x)
        
        out_src = self.conv31(h)
        tmp = self.conv32(h)
        
        if printH and False : 
            print('tmp : ',tmp[:2])
        
        out_cls = tmp
            
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
    
    '''def forward(self, x,useTanh = True,printH = False):
        x1 = self.lrelu(self.conv1(x))
        
        x21 = self.lrelu(self.conv21(x1))
        x22 = self.lrelu(self.conv22(x21))
        x23 = self.lrelu(self.conv23(x22))
        x24 = self.lrelu(self.conv24(x23))
        h = self.lrelu(self.conv25(x24))
        
        #h = self.main(x)
        
        
        out_src = self.conv31(h)
        tmp = self.conv32(h)
        
        if printH and False : 
            print('tmp : ',tmp[:2])
        if useTanh : 
            out_cls = self.tanh(tmp)
        else : 
            out_cls = tmp
            
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))'''
    
    

class DiscriminatorMST(nn.Module):
    """Discriminator network -single task """
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, asDiscriminator = False):
        super(DiscriminatorMST, self).__init__()
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)
        self.asD = asDiscriminator 
        
        self.conv1 =  nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)
        
        
        curr_dim = conv_dim
        
        self.conv21 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv22 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv23 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv24 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        self.conv25 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)
        curr_dim = curr_dim * 2
        
        kernel_size = int(image_size / np.power(2, repeat_num))
        
        #self.conv31 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv31 = nn.Conv2d(curr_dim, 1, kernel_size=kernel_size, bias=False)
        self.conv32 = nn.Conv2d(curr_dim, 1, kernel_size=kernel_size, bias=False)
        
        if asDiscriminator : 
            self.conv30 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        #self.linear41 = nn.Linear(4, c_dim)
        
    def forward(self, x,printH = False):
        
        
        x1 = self.lrelu(self.conv1(x))
        
        x21 = self.lrelu(self.conv21(x1))
        x22 = self.lrelu(self.conv22(x21))
        x23 = self.lrelu(self.conv23(x22))
        x24 = self.lrelu(self.conv24(x23))
        h = self.lrelu(self.conv25(x24))
        
        out_A = self.conv31(h)
        out_V = self.conv32(h)
        
        out_A = out_A.view(out_A.size(0), out_A.size(1))
        out_V = out_V.view(out_V.size(0), out_V.size(1))
        outs = torch.cat((out_A,out_V),1)
        '''print('out a : ',out_A.shape)
        print('out v : ',out_V.shape)
        
        out_A = torch.unsqueeze(out_A,1)
        out_V = torch.unsqueeze(out_V,1)'''
        
        if self.asD : 
            out_src = self.conv30(h)
            return out_src,outs
        else : 
            return outs #out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
        


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, use_skip = True):
        super(Generator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num/2):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = True))
        
        layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = False))    
        
        for i in range(repeat_num/2):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,use_skip = True))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c = None):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        if c is not None : 
            c = c.view(c.size(0), c.size(1), 1, 1)
            c = c.repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat([x, c], dim=1)
            
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
        self.linear1 = nn.Linear(4, c_dim)
        
    def forward(self, x,useTanh = True,printH = False):
        
        h = self.main(x)
        out_src = self.conv1(h)
        
        tmp = self.conv2(h)
        
        if printH : 
            print('tmp : ',tmp[:2])
        if useTanh : 
            out_cls = self.tanh(tmp)
        else : 
            out_cls = tmp
            
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

