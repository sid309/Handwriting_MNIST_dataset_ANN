import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import os

def prepare_data():
    mnist=tf.keras.datasets.mnist
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    x_valid,x_train=x_train[0:5000]/255.0,x_train[5000:]/255.0
    y_valid,y_train=y_train[0:5000],y_train[5000:]
    x_test=x_test/255.0
    return (x_train,y_train),(x_valid,y_valid),(x_test,y_test)

def save_plot(history):
    pd.DataFrame(history.history).plot(figsize=(10,7))
    plt.grid(True)
    figure=plt.gcf()
    figure.set_size_inches(10, 8)
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
    plotPath = os.path.join(plot_dir, "figure") # model/filename
    plt.savefig(plotPath)
def save_model(model,filename):
  model_dir='model'
  os.makedirs(model_dir,exist_ok=True)
  filepath=os.path.join(model_dir,filename)
  model.save('model/model.h5')
