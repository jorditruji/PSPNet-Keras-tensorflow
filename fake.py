import numpy as np
import matplotlib

import matplotlib.pyplot as plt
def plot_metrics(history):

    print(history.history.keys())

    fig = matplotlib.pyplot.figure(1)

    # summarize history for accuracy

    matplotlib.pyplot.subplot(211)
    acc=[]
    matplotlib.pyplot.plot(history.history['acc'])
    matplotlib.pyplot.plot(history.history['val_acc'])
    matplotlib.pyplot.title('model accuracy')
    matplotlib.pyplot.ylabel('accuracy')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    matplotlib.pyplot.subplot(212)
    matplotlib.pyplot.plot(history.history['loss'])
    matplotlib.pyplot.plot(history.history['val_loss'])
    matplotlib.pyplot.title('model loss')
    matplotlib.pyplot.ylabel('loss')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.legend(['train', 'test'], loc='upper left')
    fig.savefig('metrics_nocustom.png', dpi=fig.dpi)

# Load dataset

matplotlib.pyplot.subplot(211)
acc=[]
loss_train= np.array([4.81, 4.22, 3.91, 3.76, 3.73, 3.7, 3.68 ,3.66,3.64,3.62,3.61, 3.62 ,3.61, 3.59 ,3.59 ,3.58 ,3.57, 3.57,3.57 ,3.56, 3.55 ,3.55 ,3.55 ,3.55 ,3.55])
loss_val= np.array([4.61, 4.21, 3.85, 3.70, 3.69, 3.68, 3.68 ,3.66,3.66,3.66,3.66, 3.68 ,3.66, 3.65,3.61,3.63,3.61, 3.61,3.61,3.60, 3.60 ,3.60 ,3.60  ,3.60  ,3.60 ])

matplotlib.pyplot.plot(loss_train)
matplotlib.pyplot.plot(loss_val)
matplotlib.pyplot.title('model accuracy')
matplotlib.pyplot.ylabel('accuracy')
matplotlib.pyplot.xlabel('epoch')
matplotlib.pyplot.legend(['train', 'test'], loc='upper left')


matplotlib.pyplot.show()