import numpy as np
import tensorflow as tf

# Sigmoid and ReLU activations
class Dense_block(tf.keras.layers.Layer):
    def __init__(self, dense_count = [16,32], activation="relu", **kwargs):
        super(Dense_block, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.layers = []
        for i, num in enumerate(dense_count):
            self.layers.append(tf.keras.layers.Dense(num, activation=self.activation, name=f"generator_dense{i}"))
    
    @tf.function
    def call(self, input):
        z = input
        for i, layer in enumerate(self.layers):
            z = layer(z)
        return z

class Output_layer(tf.keras.layers.Layer):
    def __init__(self, output_shape=(28,28), activation="relu", **kwargs):
        super(Output_layer, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.output_layer = tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation=self.activation)
    
    @tf.function
    def call(self, input):
        #print("Input Type Generator: ", type(input))
        return self.output_layer(input)

# class Sigmoid_layer(tf.keras.layers.Layer):
#     def __init__(self, output_shape=(1,), activation="sigmoid", **kwargs):
#         super(Sigmoid_layer, self).__init__(**kwargs)
#         self.activation = tf.keras.activations.get(activation)
#         self.output_layer = tf.keras.layers.Dense(output_shape[0], activation=self.activation)
    
#     def call(self, input):
#         return self.output_layer(input)
class Generator(tf.keras.Model):
    def __init__(self, gen_struc=[([16,32], "relu"),([16,32], "sigmoid"),([16,32], "relu")], output_activation="relu", output_shape=(28,28), **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.dense_blocks = []
        self.output_activation = tf.keras.activations.get(output_activation)
        for dense_count, activation in gen_struc:
            self.dense_blocks.append(Dense_block(dense_count=dense_count, activation=activation))
        self.output_layer = Output_layer(output_shape=output_shape, activation=output_activation)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.out_shape = output_shape
    
    @tf.function
    def call(self, input):
        # Sample random vector
        #tf.random.set_seed(5)
        #input_vector = tf.random.normal(shape=[input_shape], mean=0.0, stddev=1.0)
        
        # Uprank the Input tensor. Need this after tensorflow 2.7
        if input.shape.rank == 1:
            input = tf.expand_dims(input, axis=0)

        Z = input
        for block in self.dense_blocks:
            Z = block(Z)
        return tf.reshape(tensor=self.output_layer(Z), shape=(self.out_shape[0], self.out_shape[1]))
