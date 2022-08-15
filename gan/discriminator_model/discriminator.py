import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import keras.backend as K

# Sigmoid and ReLU activations
class Dense_block(tf.keras.layers.Layer):
    def __init__(self, dense_count = [16,32], maxout_cout = [16,32], activation="relu", maxout=1, **kwargs):
        super(Dense_block, self).__init__(**kwargs)
        if not maxout:
            self.activation = tf.keras.activations.get(activation)
            self.layers = []
            for _, num in enumerate(dense_count):
                self.layers.append(tf.keras.layers.Dense(num, activation=self.activation))
        elif maxout:
            self.layers = []
            self.activation = tf.keras.activations.get(activation)
            for i, num in enumerate(dense_count):
                # MaxPooling will be performed at the last axis (axis=-1)
                self.layers.append(tf.keras.layers.Dense(num, activation=self.activation, name=f"discrim_dense_{i}"))
                self.layers.append(tfa.layers.Maxout(num))  
        else:
            print("[Discriminator Error]: Invalid value for maxout")
    
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
    
    def call(self, input):
        return self.output_layer(input)


class Discriminator(tf.keras.Model):
    def __init__(self, gen_struc=[([16,32], "relu"),([64,64], "relu"),([32,16], "relu")], output_activation="sigmoid", output_shape=(1,), **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.output_activation = tf.keras.activations.get(output_activation)
        self.dense_blocks = []
        self.flatten_layer = tf.keras.layers.Flatten()
        #self.discr_variables = []
        for i, (dense_count, activation) in enumerate(gen_struc):
            self.dense_blocks.append(Dense_block(dense_count=dense_count, activation=activation, name=f"dense_block{i}"))
        self.out_sigmoid_layer = Sigmoid_layer(output_shape=output_shape, activation=output_activation)
    
    def get_loss(self, noise_sample, data_sample, generator_model):
        loss = K.log(self.call(data_sample)) + K.log(1 - self.call(generator_model(noise_sample)))
        return tf.losses.categorical_crossentropy(noise_sample, noise_sample)

    @tf.function
    def learn(self, noise_sample, data_sample, generator_model):
        discr_variables = []
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            # Capture Variables
            for block in self.dense_blocks:
                for layer in block.layers:
                    if len(layer.variables)!=0:
                        tape.watch(layer.variables)
                        discr_variables.append(layer.variables[0])
                        discr_variables.append(layer.variables[1])
            tape.watch(self.out_sigmoid_layer.variables)
            discr_variables.append(self.out_sigmoid_layer.variables[0])
            discr_variables.append(self.out_sigmoid_layer.variables[1])
            L = self.get_loss(noise_sample=noise_sample, data_sample=data_sample, generator_model=generator_model)
        grads = tape.gradient(L, discr_variables)
        del tape
        self.optimizer.apply_gradients(zip(grads, discr_variables))
        return grads
    

    #Input as a tensor only
    def call(self, input):
        # Sample random vector
        #tf.random.set_seed(5)
        #input_vector = tf.random.normal(shape=[input_shape], mean=0.0, stddev=1.0)
        
        # Uprank the Input tensor. Need this after tensorflow 2.7
        if input.shape.rank == 1:
            input = tf.expand_dims(input, axis=-1)
        
        self.input_layer = tf.keras.Input(shape=input.shape)
        Z = self.flatten_layer(self.input_layer)
        print("Flattened Shapes: ", Z.shape)
        for block in self.dense_blocks:
            Z = block(Z)
        return self.out_sigmoid_layer(Z)
