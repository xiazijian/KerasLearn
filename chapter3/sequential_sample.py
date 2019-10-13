from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#设置随机种子
np.random.seed(7)
#导入数据
dataset = np.loadtxt('pima-indians-diabetes.csv',delimiter=',')
#print(dataset.shape)
#print(type(dataset))
#以下两种写法我发现效果是一样的，就是都截取了列,但是发现用列表的切片data[0:8]竟然没有切成功。。。
X = dataset[:,0:8]
Y = dataset[...,8]
# print(X[0])
# print(Y[0])
# 创建模型
model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
# 编译模型
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# 训练模型
model.fit(x=X,y=Y,epochs=150,batch_size=10,validation_split=0.2)
# 评估模型
scores = model.evaluate(x=X,y=Y)
print('\n%s : %.2f%%' % (model.metrics_names[1],scores[1]*100))



