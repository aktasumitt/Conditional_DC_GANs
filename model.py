import torch.nn as nn
import torch


class Generator(nn.Module):
    
    def __init__(self,f_dim,channel_img,num_classes,embed_size):
        super(Generator,self).__init__()
        self.embed_size=embed_size
        
        self.conv_g=nn.Sequential(nn.ConvTranspose2d(in_channels=f_dim+embed_size,out_channels=1024,kernel_size=4),                            #(f_dim=100,1,1)-->(1024,4,4)
                      nn.BatchNorm2d(num_features=1024),
                      nn.ReLU(), 
                      
                      nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=4,stride=2,padding=1),           # (512,8,8) 
                      nn.BatchNorm2d(num_features=512), 
                      nn.ReLU(),
                      
                      nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1),            # (256,16,16) 
                      nn.BatchNorm2d(num_features=256), 
                      nn.ReLU(),
                      
                      nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1),            # (128,32,32)
                      nn.BatchNorm2d(num_features=128), 
                      nn.ReLU(),
                      
                      nn.ConvTranspose2d(in_channels=128,out_channels=channel_img,kernel_size=4,stride=2,padding=1),    # (3,64,64)
                      nn.Tanh())  
        
        self.embed=nn.Embedding(num_embeddings=num_classes,embedding_dim=self.embed_size)  
        
    def forward(self,data,labels):
        x=self.embed(labels).unsqueeze(2).unsqueeze(3)
        data=torch.cat([data,x],dim=1)
        out=self.conv_g(data) 
        return out
    
    
class Discriminator(nn.Module):
    def __init__(self,channel_img,num_classes,img_size) -> None:
        super(Discriminator,self).__init__()
        self.img_size=img_size
        
        self.conv_d=nn.Sequential(nn.Conv2d(in_channels=channel_img+1,out_channels=128,kernel_size=4,stride=2,padding=1),    # (3,64,64)--->(128,32,32)
                             nn.InstanceNorm2d(128,affine=True),
                             nn.LeakyReLU(0.2),
                             
                             nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1),             # (256,16,16)
                             nn.InstanceNorm2d(256,affine=True),
                             nn.LeakyReLU(0.2),
                             
                             nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=1),              # (512,8,8)
                             nn.InstanceNorm2d(512,affine=True),
                             nn.LeakyReLU(0.2),
                             
                             nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=4,stride=2,padding=1),             # (1024,4,4)
                             nn.InstanceNorm2d(1024,affine=True),
                             nn.LeakyReLU(0.2),
                             
                             nn.Conv2d(in_channels=1024,out_channels=1,kernel_size=4),                                   # (1,1,1)
                            )        
        self.embed=nn.Embedding(num_embeddings=num_classes,embedding_dim=img_size*img_size)
    
    
    def forward(self,data,labels):
        
        x=self.embed(labels)
        x=x.view(labels.shape[0],1,self.img_size,self.img_size)
        
        condition_cat=torch.cat([data,x],dim=1)
        
        out=self.conv_d(condition_cat)
        return out
        
        
        