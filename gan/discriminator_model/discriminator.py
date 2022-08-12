import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# Sigmoid and ReLU activations
class Dense_block(tf.keras.layers.Layer):
    def __init__(self, dense_count = [16,32], maxout_cout = [16,32], activation="relu", maxout=1, **kwargs):
        super().__init__(**kwargs)
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
        print(input.shape)
        z = input
        for i, layer in enumerate(self.layers):
            z = layer(z)
        return z

class Sigmoid_layer(tf.keras.layers.Layer):
    def __init__(self, output_shape=(1,), activation="sigmoid", **kwargs):
        super().__init__(**kwargs)
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
        for dense_count, activation in gen_struc:
            self.dense_blocks.append(Dense_block(dense_count=dense_count, activation=activation))
        self.out_sigmoid_layer = Sigmoid_layer(output_shape=output_shape, activation=output_activation)
    
    def get_loss(self, noise_sample, data_sample, generator_model):
        return tf.math.log(self.call(data_sample)) + tf.math.log(1 - self.call(generator_model(noise_sample)))
    
    @tf.function
    def learn(self, noise_sample, data_sample, generator_model):
        with tf.GradientTape(persistent=True) as tape:
            
            # Capture Variables
            for block in self.dense_blocks:
                for layer in block.layers:
                    if len(layer.variables)!=0:
                        tape.watch(layer.variables)

            tape.watch(self.out_sigmoid_layer.variables)
            L = self.get_loss(noise_sample=noise_sample, data_sample=data_sample, generator_model=generator_model)
        grads = tape.gradient(L, self.trainable_variables)
        del tape
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
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
        for block in self.dense_blocks:
            Z = block(Z)
        return self.out_sigmoid_layer(Z)
