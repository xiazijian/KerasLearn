from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from  keras.optimizers import SGD
import numpy as np

# 导入数据
dataset = datasets.load_iris()
x = dataset.data
Y = dataset.target
seed = 7
# 构建模型函数
def create_model(init='glorot_uniform'):
    # 构建模型
    model = Sequential()
    model.add(Dropout(rate=0.2,input_shape=(4,)))
    model.add(Dense(units=4,activation='relu',kernel_initializer=init,kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init,kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))

    # 模型优化
    learningRate = 0.1
    momentum = 0.9
    decay_rate = 0.005
    # 定义dropout
    sgd = SGD(lr=learningRate,momentum=momentum,decay=decay_rate,nesterov=False)
    # 编译模型
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    return model
model = KerasClassifier(build_fn=create_model,epochs=200,batch_size=5,verbose=1)
kfold = KFold(n_splits=10,shuffle=True,random_state=seed)
results = cross_val_score(model,x,Y,cv=kfold)
print('Accuracy:%.2f%% (%.2f)' % (results.mean()*100,results.std()))