#python main.py | qsub -d /home/u14035/Pneumonia
import os
import numpy as np
import pickle
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from datetime import datetime
from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


img_width = 250
img_height = 325
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory('chest_xray/val', target_size=(img_width, img_height), batch_size=1,
                                                        class_mode='binary', color_mode='grayscale')
main_path = 'ff512_2_2'
model = models.load_model(main_path + '.h5')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=116)
print(test_acc)
y_data = test_generator.next()
print(len(y_data[0]))
y_test, y_true = y_data[0], y_data[1]
for i in range(1, 116):
    y_data = test_generator.next()
    t0, t1  = y_data[0], y_data[1]
    y_test = np.concatenate((y_test, t0), axis = 0)
    y_true = np.concatenate((y_true, t1), axis = 0)

print(y_test.shape)
print(y_true.shape)

y_pred = model.predict_proba(y_test)
n_classes = 2

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(test_generator.class_indices)
class_names = ['PNEUMONIA', 'NORMAL']
y_pred = np.where(y_pred > 0.5, 1.0, 0.0)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plt.grid(False)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()




