import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# Sigmoid and ReLU activations
class Dense_block(tf.keras.layers.Layer):
    def __init__(self, dense_count = [16,32], maxout_cout = [16,32], activation="relu", maxout=1, **kwargs):
        super(Dense_block, self).__init__(**kwargs)
        if not maxout:
            self.activation = tf.keras.activations.get(activation)
            self.layers = []
            for _, num in enumerate(dense_count):
                self.layers.append(tf.keras.layers.Dense(num, activation=self.activation, name=f"discriminator_dense{i}"))
        elif maxout:
            self.layers = []
            self.activation = tf.keras.activations.get(activation)
            for i, num in enumerate(dense_count):
                # MaxPooling will be performed at the last axis (axis=-1)
                self.layers.append(tf.keras.layers.Dense(num, activation=self.activation, name=f"discrim_dense{i}"))
                self.layers.append(tfa.layers.Maxout(num))  
        else:
            print("[Discriminator Error]: Invalid value for maxout")
    
    @tf.function
    def call(self, input):
        z = input
        for i, layer in enumerate(self.layers):
            z = layer(z)
        return z

class Sigmoid_layer(tf.keras.layers.Layer):
    def __init__(self, output_shape=(1,), activation="sigmoid", **kwargs):
        super(Sigmoid_layer, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.output_layer = tf.keras.layers.Dense(output_shape[0], activation=self.activation)
    
    @tf.function
    def call(self, input):
        return self.output_layer(input)


class Discriminator(tf.keras.Model):
    def __init__(self, disc_img_struc=[([240], "relu", 1)], disc_cond_struc=[([50], "relu", 1)], disc_body_struc=[([240], "relu", 1)], output_activation="sigmoid", output_shape=(1,), **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.output_activation = tf.keras.activations.get(output_activation)
        self.dense_img_blocks = []
        self.dense_cond_blocks = []
        self.dense_body_blocks = []
        self.flatten_layer = tf.keras.layers.Flatten()
        #self.discr_variables = []
        for i, (dense_count, activation, maxout) in enumerate(disc_img_struc):
            self.dense_img_blocks.append(Dense_block(dense_count=dense_count, activation=activation, maxout=maxout, name=f"Image_dense_block{i}"))
        for i, (dense_count, activation, maxout) in enumerate(disc_cond_struc):
            self.dense_cond_blocks.append(Dense_block(dense_count=dense_count, activation=activation, maxout=maxout, name=f"Conditional_dense_block{i}"))
        for i, (dense_count, activation, maxout) in enumerate(disc_body_struc):
            self.dense_body_blocks.append(Dense_block(dense_count=dense_count, activation=activation, maxout=maxout, name=f"Body_dense_block{i}"))
        self.out_sigmoid_layer = Sigmoid_layer(output_shape=output_shape, activation=output_activation, name=f"Sigmoid_Layer")

    #Input as a tensor only
    @tf.function
    def call(self, inputs):
        # Sample random vector
        #tf.random.set_seed(5)
        #input_vector = tf.random.normal(shape=[input_shape], mean=0.0, stddev=1.0)
        
        # Uprank the Input tensor. Need this after tensorflow 2.7
        # The first element in Input is number of samples. To flatten, this code provides 1 at axis 0
        input_img = inputs[0]
        input_cond = inputs[1]
        if input_img.shape.rank == 2:
            input_img = tf.expand_dims(input_img, axis=0)
        if input_cond.shape.rank == 1:
            input_cond = tf.expand_dims(input_cond, axis=0)
        
        Z = self.flatten_layer(input_img)
        Y = input_cond
        
        for block in self.dense_img_blocks:
            Z = block(Z)
        
        for block in self.dense_cond_blocks:
            Y = block(Y)
        
        concat_layer = tf.keras.layers.concatenate(inputs=[Z,Y], axis=-1)
        D = concat_layer
        
        for block in self.dense_body_blocks:
            D = block(D)
        return self.out_sigmoid_layer(D)
