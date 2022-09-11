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
    def __init__(self, output_shape=(28,28), activation="sigmoid", **kwargs):
        super(Output_layer, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.output_layer = tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation=self.activation, name=f"gen_out_layer")
    
    @tf.function
    def call(self, input):
        #print("Input Type Generator: ", type(input))
        return self.output_layer(input)

class Generator(tf.keras.Model):
    def __init__(self, gen_noise_struc=[([200], "relu")], gen_cond_struc=[([1000], "relu")], gen_body_struc=[([1200], "relu")], output_activation="sigmoid", output_shape=(28,28), 
                  **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.dense_noise_blocks = []
        self.dense_cond_blocks = []
        self.dense_body_blocks = []
        self.output_activation = tf.keras.activations.get(output_activation)
        for i, (dense_count, activation) in enumerate(gen_noise_struc):
            self.dense_noise_blocks.append(Dense_block(dense_count=dense_count, activation=activation, name=f"Noise_dense_block{i}"))
        for i, (dense_count, activation) in enumerate(gen_cond_struc):
            self.dense_cond_blocks.append(Dense_block(dense_count=dense_count, activation=activation, name=f"Conditional_dense_block{i}"))
        for i, (dense_count, activation) in enumerate(gen_body_struc):
            self.dense_body_blocks.append(Dense_block(dense_count=dense_count, activation=activation, name=f"Body_dense_block{i}"))
        self.output_layer = Output_layer(output_shape=output_shape, activation=output_activation, name=f"Generator_output_layer")
        self.out_shape = output_shape
    
    @tf.function
    def call(self, inputs):
        # Sample random vector
        #tf.random.set_seed(5)
        #input_vector = tf.random.normal(shape=[input_shape], mean=0.0, stddev=1.0)
        
        # Uprank the Input tensor. Need this after tensorflow 2.7
        input_noise = inputs[0]
        input_cond = inputs[1]
        if input_noise.shape.rank == 1:
            input_noise = tf.expand_dims(input_noise, axis=0)
        if type(input_cond) == "list":
            input_cond = tf.convert_to_tensor(input_cond)
        input_cond = tf.expand_dims(input_cond, axis=0)       
        
        Z = input_noise
        Y = input_cond

        for block in self.dense_noise_blocks:
            Z = block(Z)
        
        for block in self.dense_cond_blocks:
            Y = block(Y)
        
        concat_layer = tf.keras.layers.concatenate(inputs=[Z,Y], axis=-1)
        G = concat_layer
        
        for block in self.dense_body_blocks:
            G = block(G)
        return tf.reshape(tensor=self.output_layer(G), shape=(self.out_shape[0], self.out_shape[1]))
