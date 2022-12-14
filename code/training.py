import os
import csv
import numpy as np
import pandas as pd
from scipy import ndimage
import scipy
import matplotlib.pyplot as plt
import sys
import imageio.v3 as iio
from skimage.transform import resize
import NeuralNetMLP as nnm

train_data_dir = "../concerete_crack_images/training/"
test_data_dir = "../concerete_crack_images/test/"

train_imgs = os.listdir(train_data_dir)
test_imgs = os.listdir(test_data_dir)

print(len(train_imgs))
print(len(test_imgs))


# Prepare x and y
num_px = 64

def extrcatFeaturesAndLabels(dir, impg_dataset):
    X = np.zeros((len(impg_dataset), num_px * num_px * 3))
    y = np.zeros((len(impg_dataset)))
    for i in range(0, len(impg_dataset)):
        # Read an image from a file as an array.
        # The different colour bands/channels are stored
        # in the third dimension, such that a
        # grey-image is MxN, an RGB-image MxNx3
        image = np.array(iio.imread(dir + impg_dataset[i]))

        # Resize the image. Size of the output image (height, width)
        image = resize(image, (num_px, num_px))

        # Convert the matrix to a vector
        image = image.reshape((1, num_px * num_px * 3)).T

        # Note that RGB (Red, Green, Blue) are 8 bit each.
        # Hence, the range for each individual colour is 0-255 (as 2^8 = 256 possibilities).
        # By dividing by 255, Convert the 0-255 range to a 0.0-1.0 range where 0.0 means 0 (0x00) and 1.0 means 255 (0xFF).
        image = image / 255.0

        image = image.reshape((num_px * num_px * 3, 1))
        for j in range(0, num_px * num_px * 3):
            X[i][j] = image[j][0]

        if 'pos' in impg_dataset[i]:
            y[i] = 1
        else:
            y[i] = 0

    y = y.astype(int)

    return (X, y)


def get_acuuracy(model, X, y):
    y_pred = model.predict(X)

    if sys.version_info < (3, 0):
        accuracy = ((np.sum(y == y_pred, axis=0)).astype('float') /
                    X.shape[0])
    else:
        accuracy = np.sum(y == y_pred, axis=0) / X.shape[0]

    # print('Accuracy: %.2f%%' % (accuracy * 100))

    return (accuracy * 100)

# Training Data Set
X_train, y_train = extrcatFeaturesAndLabels(train_data_dir, train_imgs)
X_test, y_test = extrcatFeaturesAndLabels(test_data_dir, test_imgs)


print('Training Data set - Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
print('Test Data set - Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

# 10 different images of non-Crack
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == 0][i].reshape(num_px, num_px, 3)
    ax[i].imshow(img)

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# 10 different images of Crack
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == 1][i].reshape(num_px, num_px, 3)
    ax[i].imshow(img)

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()




nn = nnm.NeuralNetMLP(n_output=2,
                  n_features=X_train.shape[1],
                  n_hidden=50,
                  l2=0.1,
                  l1=0.1,
                  epochs=50,
                  eta=0.001,
                  alpha=0.01,
                  decrease_const=0.00001,
                  minibatches=100,
                  shuffle=True,
                  random_state=1)

nn.fit(X_train, y_train, print_progress=True)


batches = np.array_split(range(len(nn.cost_)), 100)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
plt.ylim([0, 6000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
#plt.savefig('./figures/cost2.png', dpi=300)
plt.show()


training_accuracy=get_acuuracy(nn, X_train, y_train)
print('Training accuracy: %.2f%%' %training_accuracy)

test_accuracy=get_acuuracy(nn, X_test, y_test)
print('Test accuracy: %.2f%%' %test_accuracy)

y_pred = nn.predict(X_test)
# fina_list = []
# fina_list.append(test_imgs)
# fina_list.append(y_pred)
# with open('output.csv', 'w') as cf:
#     csvfile = csv.writer(cf, delimiter=' ')
#     for column in zip(*[i for i in fina_list]):
#         csvfile.writerow(column)
# cf.close()

dataframe = pd.DataFrame({'Image':test_imgs,'Prediction':y_pred})
dataframe.to_csv("output.csv",sep=',',index=False)