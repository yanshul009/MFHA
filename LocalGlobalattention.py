import torch 
import torch.nn as nn
import torch.nn.functional as f
import math
from Attention import SelfAttention

class MultiheadAttention(nn.Module):
    def __init__(self,input_size=1024,output_size=1024,freq=10000,pos_enc=None,num_segments=None,heads=1,fusion=None):
        super(MultiheadAttention,self).__init__()

        self.attention=SelfAttention(input_size=input_size,output_size=output_size,freq=freq,pos_enc=pos_enc,heads=heads)
        
        self.num_segments=num_segments

        if self.num_segments is not None:
            self.local_attention=nn.ModuleList()
            for _ in range(self.num_segments):
                self.local_attention.append(SelfAttention(input_size=input_size,output_size=output_size//num_segments,freq=freq,pos_enc=pos_enc,heads=1))
            
            self.fusion=fusion
    
    def forward(self,x):

        # weighted_value,attn_weight=self.attention(x)
        if self.num_segments is not None and self.fusion is not None:
            segment_size=math.ceil(x.shape[0]/self.num_segments)
            for segment in range(self.num_segments):
                left_pos=segment*segment_size
                right_pos=(segment+1)*segment_size
                # x[left_pos:right_pos]=x[left_pos:right_pos].clone()
                local_x=x[left_pos:right_pos]
                weighted_local_value,attn_local_weight=self.local_attention[segment](local_x.clone())

                x[left_pos:right_pos]=f.normalize(x[left_pos:right_pos].clone(),p=2,dim=1)
                weighted_local_value=f.normalize(weighted_local_value,p=2,dim=1)
                if self.fusion=='add':
                    x[left_pos:right_pos]=x[left_pos:right_pos]+weighted_local_value
                
        weighted_value,attn_weight=self.attention(x)
        
        return weighted_value,attn_weight
    

class LocalGlobalattention(nn.Module):

    def __init__(self, input_size=1024,output_size=1024,freq=1000,pos_enc=None,num_segments=None,heads=1,fusion=None):
        super(LocalGlobalattention,self).__init__()
        
        self.attention=MultiheadAttention(input_size=input_size,output_size=output_size,freq=freq,pos_enc=pos_enc,num_segments=num_segments,heads=heads,fusion=fusion)
        self.linear1=nn.Linear(in_features=input_size,out_features=output_size)
        self.linear2=nn.Linear(in_features=self.linear1.out_features,out_features=1)
        self.drop=nn.Dropout(p=0.5)
        self.normy=nn.LayerNorm(normalized_shape=input_size,eps=1e-6)
        self.norm_Linear=nn.LayerNorm(normalized_shape=self.linear1.out_features,eps=1e-6)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    
    def forward(self,frame_feature_list,framesequence):

        y_out_ls=[]
        att_weights_=[]
        for i in range(len(frame_feature_list)):
            x = frame_feature_list[i].view(-1, frame_feature_list[i].shape[2])
            y, att_weights = self.attention(x)
            att_weights_  = att_weights
            y = y + x
            y = self.drop(y)    
            y = self.normy(y)
            y_out_ls.append(y)
        
        y=y_out_ls[0]
        for i in range(1,len(y_out_ls)):
            y=y+y_out_ls[i]



        y=self.linear1(y)
        y=self.relu(y)
        y=self.drop(y)
        y=self.norm_Linear(y)


        y=self.linear2(y)
        y=self.sigmoid(y)
        y=y.view(1,-1)

        return y,att_weights_
    

if __name__=='__main__':
    pass
    # model=LocalGlobalattention(input_size=1024,output_size=1024,num_segments=4,fusion='add')
    # _input=torch.randn(500,1024)
    # output,weight=model(_input)
    # print(output.shape)

