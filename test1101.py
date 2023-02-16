import visdom
import numpy as np
vis = visdom.Visdom(env='my_windows')  # 设置环境窗口的名称,如果不设置名称就默认为main
x = list(range(10))
y = list(range(10))
# 使用line函数绘制直线 并选择显示坐标轴
vis.line(X=np.array(x), Y=np.array(y), opts=dict(showlegend=True))
