import numpy as np
import matplotlib.pyplot as plt

def Gradient_descent_thetal_0 (thetal,x,y):
    J=0
    m=len(y)
    h=thetal*x
    J=(h.sum()-y.sum())/m
    return J#Gradient_descent_thetal_0(thetal,x,y)  return J

def Gradient_descent_thetal(thetal,x,y,j):
    J=0
    m=len(y)
    h=thetal*x
    diff=(h.sum(axis=1)-y)*x[:,j]
    J=diff.sum()/m
    return J#Gradient_descent_thetal(thetal,x,y,j)    return J

file_address='./data/housing.data'
data=np.fromfile(file_address,dtype=float,count=-1,sep=' ')
data=data.reshape(-1,14)    #重整数组为14列数组
data_x=np.delete(data,13,1)
data_x=np.insert(data_x,0,1,axis=1)#增添值为1的一列
data_y=data[:,13]
#数据处理

thetal=np.random.random((14))    #thetal
update=np.ones((1,14))    #update
max_iterate=50000          #最大迭代次数
min_update=0.0001      #最小迭代更新大小
step=0.0000126              #步进长度
iterate=0               #迭代次数
max_update=1            #平均迭代更新大小

while(iterate<max_iterate)&(max_update>min_update):
    update[0,0]=Gradient_descent_thetal_0(thetal,data_x,data_y)
    thetal[0]-=(update[0,0]*step)
    #thetal 0 参数迭代
    for i in range(1,13):
        update[0,i]=Gradient_descent_thetal(thetal,data_x,data_y,i)
        thetal[i]-=(update[0,i]*step)
    #thetal 参数迭代
    #print(update)   #每一次迭代后update数组的值
    max_update=max(update[0])*step
    if max_update<0:    #迭代更新值取绝对值
        max_update=-max_update
    #print(max_update)
    h=thetal*data_x
    diff=data_y-h.sum(axis=1)
    loss=diff.sum()/506
    plt.scatter(iterate,loss)
    iterate+=1
plt.show()