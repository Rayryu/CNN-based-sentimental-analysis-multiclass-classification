try:
    %tensorflow_version 2.x
except Exception:
    pass
import tensorflow as tf
from tensorflow.keras import layers

class DCNN(tf.keras.Model):
        def __init__(self,
                 vocab_size,  #Size of the vocabulary used. Will be given by the tokenizer
                 emb_dim=128,  #128 is just an intuitive default value, it's used so ofter for embedding 
                 nb_filters=50,  #Number of times we want to apply each filter
                 FFN_units=512,  #Number of units of the feedforward neural network at the end
                 nb_classes=2,  #Binary classification as default
                 dropout_rate=0.1,  #To turn off certain units/parameters to avoid overfitting
                 training=False,  #True if the network is in evaluation phase. Drop out will be applied only in training
                 name="dcnn"):  #A name for the network
        super(DCNN, self).__init__(name=name)

        self.embedding = layers.Embedding(vocab_size,
                                          emb_dim)
        self.bigram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=2,
                                    padding="valid",  #To add the zeros we need to performe the last convolutions
                                    activation="relu")
        self.trigram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=3,
                                    padding="valid",  #To add the zeros we need to performe the last convolutions
                                    activation="relu")
        self.fourgram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=4,
                                    padding="valid",  #To add the zeros we need to performe the last convolutions
                                    activation="relu")
        self.pool = layers.GlobalMaxPool1D()  #We'll be using this layer for all pooling steps
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)  #This is a good place to define dropout since dense_1 with create a lot of params
        if nb_classes == 2:  #Easy way to handle multiclasses
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)

        merged = tf.concat([x_1, x_2, x_3], axis=-1)  #(batch_size, 3 * nb_filters) 3 because we have 3 types of filters
        merged = self.dense_1(merged)  #Staring the feedforward process
        merged = self.dropout(merged, training)  #Applying dropout if training=True
        output = self.last_dense(merged)  
        
        return output