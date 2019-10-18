import numpy as np
import matplotlib.pyplot as plt # 导入 matplotlib 命名为 plt，类似 matlab，集成了许多可视化命令


# mean_1 = [1, 5]
# cov_1 = [[1, 0], [0, 1]]
# # .T进行转置这样更加方便，因为每个子list中就是同一个变量的值
# data_1 = np.random.multivariate_normal(mean_1, cov_1, 1000)
# print(data_1)
# plt.plot(data_1[0],data_1[1],label = 'k1',color='black')
#
# save_data = pd.DataFrame(data=data_1)
# save_data.to_csv(r'D:\Codes\pycharm\LearnKeras\Experience\data_1.csv',header=False,index=False)

# mean_2 = [5.7, 4]
# cov_2 = [[1, 0], [0, 1]]
# # .T进行转置这样更加方便，因为每个子list中就是同一个变量的值
# data_2 = np.random.multivariate_normal(mean_2, cov_2, 1000)
# print(np.mean(data_2[0]))
# print(np.mean(data_2[1]))
# plt.plot(data_2[0],data_2[1],label = 'k2',color='red')
# save_data = pd.DataFrame(data=data_2)
# save_data.to_csv(r'D:\Codes\pycharm\LearnKeras\Experience\data_2.csv',header=False,index=False)
#
# mean_3 = [5, 8.9]
# cov_3 = [[1, 0], [0, 1]]
# # .T进行转置这样更加方便，因为每个子list中就是同一个变量的值
# data_3 = np.random.multivariate_normal(mean_3, cov_3, 1000)
# print(np.mean(data_3[0]))
# print(np.mean(data_3[1]))
# plt.plot(data_3[0],data_3[1],label = 'k3',color='blue')
# save_data = pd.DataFrame(data=data_3)
# # 不让数据有行名和列名
# save_data.to_csv(r'D:\Codes\pycharm\LearnKeras\Experience\data_3.csv',header=False,index=False)
# plt.show()


# data = np.loadtxt('data_1.csv',delimiter=',')
# print(np.mean(data.T[0]))
import math

def parzen(n,xi,h1):
    def f(x):
        re = 0
        for i in range(n):
            re = re + math.sqrt(n)*math.exp(-((x-xi[i]).dot((x-xi[i]))*n)/(2*h1**2))
        return re/(n*math.sqrt(2*math.pi))
    return f
# # test
# f = parzen(2,np.array([[2,3],[3,4]]),1)
# t = f(np.array([1,2]))
# print(t)
# q = (math.sqrt(2)*math.exp(-2)+math.sqrt(2)*math.exp(-8))/(2*math.sqrt(2*math.pi))
# print(q)

data_set = np.loadtxt('data_2.csv',delimiter=',')
data_100 = data_set[0:300]
# print(np.mean(data_100.T[0]))

f = parzen(len(data_100),data_100,10)
# x1 = np.random.randint(100,size=100)
# x2 = np.random.randint(100,size=100)
# 将 x1和x2合并成矩阵
# x = np.array(list(zip(x1,x2)))
z = []
for i in range(len(data_set)):
    z.append(f(data_set[i]))
z = np.array(z)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('n=300 h1=10')
plt.xlabel('x')
plt.ylabel('y')
ax.plot_trisurf(data_set[:,0],data_set[:,1],z, cmap='rainbow')
plt.show()

# f = parzen(len(data_100),data_100,1)
# z = []
# for i in range(len(data_set)):
#     z.append(f(data_set[i]))
# z = np.array(z)
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_title('n=100 h1=1')
# plt.xlabel('x')
# plt.ylabel('y')
# ax.plot_trisurf(data_set[:,0],data_set[:,1],z, cmap='rainbow')
#
#
# f = parzen(len(data_100),data_100,5)
# z = []
# for i in range(len(data_set)):
#     z.append(f(data_set[i]))
# z = np.array(z)
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_title('n=100 h1=5')
# plt.xlabel('x')
# plt.ylabel('y')
# ax.plot_trisurf(data_set[:,0],data_set[:,1],z, cmap='rainbow')
# plt.show()


test = [0.3568334870227461,3.571159202472524]
print(f(test))



