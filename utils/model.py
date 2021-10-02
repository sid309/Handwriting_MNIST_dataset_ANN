import tensorflow as tf
import keras

model=tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=[28,28],name='input_layer'),
                    tf.keras.layers.Dense(300,activation='relu',name='hidden_layer1'),
                    tf.keras.layers.Dense(100,activation='relu',name='hidden_layer2'),
                    tf.keras.layers.Dense(10,activation='softmax',name='output_layer')
                    ])