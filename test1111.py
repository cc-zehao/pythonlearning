import visdom
import torch
# 新建一个连接客户端
# 指定env = 'test1'，默认是'main',注意在浏览器界面做环境的切换
vis = visdom.Visdom(env='test1')
# 绘制正弦函数
x = torch.arange(1, 100, 0.01)
y = torch.sin(x)
vis.line(X=x,Y=y, win='sinx',opts={'title':'y=sin(x)'})
# 绘制36张图片随机的彩色图片
vis.images(torch.randn(36,3,64,64).numpy(),nrow=6, win='imgs',opts={'title':'imgs'})