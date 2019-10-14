from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# 导入数据
dataset = datasets.load_boston()

x = dataset.data
Y = dataset.target
# 设定随机种子，以便重复构建模型
seed = 7
np.random.seed(seed)
# 构建模型函数
def create_model(units_list=[13],optimizer='adam',init='normal'):
    model = Sequential()
    # 构建第一个隐藏层喝输出层
    units = units_list[0]
    model.add(Dense(units=units,activation='relu',input_dim=13,kernel_initializer=init))
    # 构造更多隐藏层
    for units in units_list[1:]:
        model.add(Dense(units=units,activation='relu',kernel_initializer=init))
    model.add(Dense(units=1,kernel_initializer=init))
    # 编译模型
    model.compile(loss='mean_squared_error',optimizer=optimizer,metrics=['accuracy','mse'])
    return model

model = KerasRegressor(build_fn=create_model,epochs=50,batch_size=5,verbose=1)
# 设置算法评估基准
kfold = KFold(n_splits=5,shuffle=True,random_state=seed)
results = cross_val_score(model,x,Y,cv=kfold)
print(results)
print('==='*20)
print('Baseline: %.2f (%.2f) MSE' % (results.mean(),results.std()))










