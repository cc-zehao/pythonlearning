import torch
import torch.nn as nn
class mymodule(nn.Module):
    def __init__(self):
        super(mymodule,self).__init__()
        self.linear=nn.Linear(2,3)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.linear(x)
        x=self.relu(x)
        return x
model=mymodule()
print("模型参数：",list((model.parameters())))
for param in model.parameters():
    print("参数类型：",type(param),"参数大小：",param.size())
