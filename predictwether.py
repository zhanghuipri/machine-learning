import os

data_dir = './'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(header)
print(len(lines))
print(lines[0])

import numpy as np
#float_data存放excel的所有数据，去除了第一行的标题，去除了第一列的时间
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    # values表示一行数据，但是把第一列给去掉了
    values = [float(x) for x in line.split(',')[1:]]
    # 二维数组赋值，每一个rank之间用,分隔，“:”表示这个rank的所有
    # 下面是把每一行都用values来赋值
    float_data[i, :] = values

print('float_data\' shape is (%d %d) :'%(float_data.shape))

from matplotlib import pyplot as plt

# 这是取每一行的第一列数据，中间用,分隔
temp = float_data[:, 1]
print(temp[0])

# plt.plot(range(len(temp)), temp)
# plt.show()

# 测试时注释掉，用于观察具体的数据
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        print('data shape (%d %d ):', data.shape)
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            print('ite %d:[%s ~ %s] '% (j, rows[j] - lookback, rows[j]))
            #(240, 14)
            print(data[indices].shape)

            samples[j] = data[indices]
            #rows 为128,delay = 144
            print('row[%d] is (%d)'% (j, rows[j]))
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128

# (128, 240, 14)
'''
举个具体的例子
Our input array should then be something shaped as (N,5,2):

        [     Step1      Step2      Step3      Step4      Step5
Tank A:    [[Pa1,Ta1], [Pa2,Ta2], [Pa3,Ta3], [Pa4,Ta4], [Pa5,Ta5]],
Tank B:    [[Pb1,Tb1], [Pb2,Tb2], [Pb3,Tb3], [Pb4,Tb4], [Pb5,Tb5]],
  ....
Tank N:    [[Pn1,Tn1], [Pn2,Tn2], [Pn3,Tn3], [Pn4,Tn4], [Pn5,Tn5]],
        ]
'''

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,  # 测试时修改了，原始值为200000
                      shuffle=False,  # 测试时修改了，原始值为True
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (300000 - 200001 - lookback) // batch_size
def evaluate_naive_method():
    for step in range(val_steps):
        samples, targets = next(val_gen)

    return samples, targets

for step in range(2):
    samples, targets = next(train_gen)
    print( samples, targets)

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(float_data) - 300001 - lookback) // batch_size

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,TensorBoard

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

callback_early_stopping = EarlyStopping(
monitor='val_loss',
patience=5, verbose=1
)
callback_checkpoint = ModelCheckpoint(
filepath='./logs/my_model.h5',
monitor='val_loss',
save_best_only=True,
)
#tensorboard --logdir=D:/Java/PycharmProjects/test/com/huawei/video/logs
callback_tensorboard = TensorBoard(
log_dir='./logs/',
histogram_freq=1,
#embeddings_freq=1,
)
callback_reduce_lr = ReduceLROnPlateau(
monitor='val_loss',
factor=0.1,
patience=10,
)
callbacks_list = [callback_early_stopping,
                    callback_checkpoint,
                    callback_reduce_lr,
                    callback_reduce_lr]
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=evaluate_naive_method(),
                              validation_steps=val_steps,
                              verbose=1,
                              callbacks=callbacks_list)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
