import numpy as np
import tensorflow as tf
from layers import conv_layers

class Discriminator(tf.keras.Model):
    def __init__(self, conv_blocks_struct, output_shape=(1,), **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.conv_blocks = []
        for i, conv_block_struct in enumerate(conv_blocks_struct):
            self.conv_blocks.append(Conv_block(conv_struct=conv_block_struct, name=f"conv_block{i}"))
        self.out_sigmoid_layer = Sigmoid_layer(output_shape=output_shape, name=f"Sigmoid_Output_Layer")

    @tf.function
    def call(self, input):
        # Sample random vector
        #tf.random.set_seed(5)
        #input_vector = tf.random.normal(shape=[input_shape], mean=0.0, stddev=1.0)
        
        # Uprank the Input tensor. Need this after tensorflow 2.7
        # The first element in Input is number of samples. To flatten, this code provides 1 at axis 0
        # put_img = inputs[0]
        # input_cond = inputs[1]
        # if input_img.shape.rank == 2:
        #     input_img = tf.expand_dims(input_img, axis=0)
        # if input_cond.shape.rank == 1:
        #     input_cond = tf.expand_dims(input_cond, axis=0)
        
        Z = input
        for block in self.conv_blocks:
            Z = block(input)
        
        return self.out_sigmoid_layer(Z)
