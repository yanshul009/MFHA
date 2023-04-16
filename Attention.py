import torch 
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self,input_size=1024,output_size=1024,freq=10000,heads=1,pos_enc=None):
        super(SelfAttention,self).__init__()
        
        self.permitted_ecnoding='absolute'
        self.input_size=1024
        self.output_size=1024
        self.heads=heads
        self.pos_enc=pos_enc
        self.freq=freq

        self.Wk=nn.ModuleList()
        self.Wq=nn.ModuleList()
        self.Wv=nn.ModuleList()

        for _ in range(self.heads):
            self.Wk.append(nn.Linear(in_features=input_size,out_features=output_size//heads,bias=False))
            self.Wq.append(nn.Linear(in_features=input_size,out_features=output_size//heads,bias=False))
            self.Wv.append(nn.Linear(in_features=input_size,out_features=output_size//heads,bias=False))
        
        self.out=nn.Linear(in_features=output_size,out_features=input_size,bias=False)
        self.softmax=nn.Softmax(dim=-1)
        self.drop=nn.Dropout(p=0.5)

    
    def getabsolute_position(self,T):
        
        freq=self.freq
        d=self.input_size
        pos=torch.tensor([k for k in range(T)],device=self.out.weight.device)
        i=torch.tensor([k for k in range(T//2)],device=self.out.weight.device)
        pos=pos.reshape(pos.shape[0],1)
        pos=pos.repeat_interleave(i.shape[0],dim=1)
        i=i.repeat(pos.shape[0],1)
        AP=torch.zeros(T,T,device=self.out.weight.device)
        res=pos.long()
        res1=2*i
        AP[res,res1.long()]=torch.sin(pos/freq ** ((2*i)/d))
        AP[res,res1.long()+1]=torch.cos(pos/freq ** ((2*i)/d))
        return AP


    def forward(self,x):
        
        outputs=[]
        for head in range(self.heads):
            K=self.Wk[head](x)
            Q=self.Wq[head](x)
            V=self.Wv[head](x)

            attention_weight=torch.matmul(Q,K.transpose(1,0))
            AP=self.getabsolute_position(T=attention_weight.shape[0])
            attention_weight=attention_weight+AP
            attention_weight=self.softmax(attention_weight)
            attention_weight=self.drop(attention_weight)
            y=torch.matmul(attention_weight,V)

            outputs.append(y)
        

        y=self.out(torch.cat(outputs,dim=1))

        return y,attention_weight.clone()
    

if __name__=='__main__':
    pass
        

    # model=SelfAttention(input_size=1024,output_size=1024,pos_enc='absolute')
    # input=torch.randn(2,1024)
    # output,weight=model(input)
    # print(output.shape)
