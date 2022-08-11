import numpy as np
import tensorflow as tf

# Sigmoid and ReLU activations
class Dense_block(tf.keras.layers.Layer):
    def __init__(self, dense_count = [16,32], activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.layers = []
        for _, num in enumerate(dense_count):
            self.layers.append(tf.keras.layers.Dense(num, activation=self.activation))
    
    def call(self, input):
        z = input
        for i, layer in enumerate(self.layers):
            z = layer(z)
        return z

class Output_layer(tf.keras.layers.Layer):
    def __init__(self, output_shape=(28,28), activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.output_layer = tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation=self.activation)
    
    def call(self, input):
        return self.output_layer(input)

class Generator(tf.keras.Model):
    def __init__(self, gen_struc=[([16,32], "relu"),([16,32], "sigmoid"),([16,32], "relu")], output_activation="relu", output_shape=(28,28), **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.dense_blocks = []
        self.output_activation = tf.keras.activations.get(output_activation)
        for dense_count, activation in gen_struc:
            self.dense_blocks.append(Dense_block(dense_count=dense_count, activation=activation))
        self.output_layer = Output_layer(output_shape=output_shape, activation=output_activation)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    
    def get_loss(self, noise_sample, discriminator_model):
        return tf.math.log(1 - discriminator(self.call(noise_sample)))
    
    def get_gradients(self, noise_sample, discriminator_model):
        with tf.GradientTape() as tape:
            self.variables_list = []
            for block in self.dense_block:
                for layer in block.layers:
                    tape.watch(layer.variables)
                    self.variables_list.append(layer.variables[0])
                    self.variables_list.append(layer.variables[1])
            tape.watch(self.output_layer.variables)
            self.variables_list.append(self.output_layer.variables[0])
            self.variables_list.append(self.output_layer.variables[1])
            L = self.get_loss(noise_sample=noise_sample, discriminator_model=discriminator_model)
            grads = tape.gradient(L, self.variables_list)
        return grads
    
    def learn(self, noise_sample, discriminator_model):
        grads = self.get_gradients(noise_sample=noise_sample, discriminator_model=discriminator_model)
        self.optimizer.apply_gradients(zip(grads, self.variables_list))

    def call(self, input):
        # Sample random vector
        #tf.random.set_seed(5)
        #input_vector = tf.random.normal(shape=[input_shape], mean=0.0, stddev=1.0)
        self.input_layer = tf.keras.Input(shape=(10,))
        Z = self.input_layer
        for block in self.dense_blocks:
            Z = block(Z)
        return self.output_layer(Z)
