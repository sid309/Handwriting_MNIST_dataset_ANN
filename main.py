import tensorflow as tf
from tensorflow import keras
from utils.model import *
from utils.all_utils import *

def main(loss_function,metrics,optimizer,EPOCHS):
    (x_train, y_train),(x_valid, y_valid),(x_test, y_test)=prepare_data()
    model.compile(loss=loss_function,optimizer=optimizer,metrics=[metrics])
    history=model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_valid,y_valid))
    model.evaluate(x_test,y_test)
    save_model(model,'mnist.model')
    save_plot(history)

if __name__ == '__main__':

    loss_function='sparse_categorical_crossentropy'
    metrics='accuracy'
    optimizer='SGD'
    EPOCHS = 30
try:
    main(loss_function=loss_function,metrics=metrics,optimizer=optimizer,EPOCHS=EPOCHS)
except Exception as e:
    raise(e)

