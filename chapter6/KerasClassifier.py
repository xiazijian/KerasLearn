from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier

# 构建模型
def create_model(optimizer='adam',init='glorot_uniform'):
    # 构建模型
    model = Sequential()
    model.add(Dense(units=12,kernel_initializer=init,input_dim=8,activation='relu'))
    model.add(Dense(units=8,kernel_initializer=init,activation='relu'))
    model.add(Dense(units=1,kernel_initializer=init,activation='sigmoid'))
    # 编译模型
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model
# 设置随机种子
seed = 7
np.random.seed(seed)
# 导入数据
dataset = np.loadtxt('pima-indians-diabetes.csv',delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]
# 创建模型 for scikit-learn
model = KerasClassifier(build_fn=create_model,verbose=0)
# 构造需要调参的参数
param_graid = {}
param_graid['optimizer'] = ['rmsprop','adam']
param_graid['init'] = ['glorot_uniform','normal','uniform']
param_graid['epochs'] = [50,100,150,200]
param_graid['batch_size'] = [5,10,20]
# 调参
from  sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=model,param_grid=param_graid)
results = grid.fit(x,y)
# 输出结果
print('Best: %f using %s' % (results.best_score_,results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']
for mean,std,param in zip(means,stds,params):
    print('%f (%f) with: %r' % (mean,std,param))


