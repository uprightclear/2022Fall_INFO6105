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
import analyze as an
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import binarize, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

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
        image = resize(image, (num_px, num_px))
        image = image.reshape((1, num_px * num_px * 3)).T
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

train_pred = nn.predict(X_train)
for i in range(len(train_pred)):
    if train_pred[i] != y_train[i]:
        print(train_imgs[i])

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



batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
plt.ylim([0, 6000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()



logreg = LogisticRegression()
an.model_training(logreg, X_train, y_train)

training_accuracy=get_acuuracy(logreg, X_train, y_train)
print('Training accuracy: %.2f%%' %training_accuracy)

test_accuracy=get_acuuracy(logreg, X_test, y_test)
print('Test accuracy: %.2f%%' %test_accuracy)

an.plot_auc_curve(logreg, X_train, y_train)


rf = RandomForestClassifier(n_estimators=1000,
                            criterion='gini',
                            max_features='sqrt',
                            n_jobs=-1)
an.model_training(rf, X_train, y_train)

an.plot_auc_curve(rf, X_test, y_test)
an.Find_Optimal_Cutoff(rf, X_test, y_test)

training_accuracy=get_acuuracy(rf, X_train, y_train)
print('Training accuracy: %.2f%%' %training_accuracy)

test_accuracy=get_acuuracy(rf, X_test, y_test)
print('Test accuracy: %.2f%%' %test_accuracy)


adaBoost = AdaBoostClassifier(n_estimators=150)
an.model_training(adaBoost, X_train, y_train)

an.plot_auc_curve(adaBoost, X_test, y_test)
an.Find_Optimal_Cutoff(adaBoost, X_test, y_test)
an.print_accurcay_metrics(adaBoost, X_test, y_test, 0.5)