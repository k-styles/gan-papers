import numpy as np
import tensorflow as tf
from layers import conv_layers

class Generator(tf.keras.Model):
    def __init__(self, conv_blocks_struct=[], convT_blocks_struct, output_layer_struct={}, 
                 **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.convT_blocks = []
        for i, convT_block_struct in enumerate(convT_blocks_struct):
            self.convT_blocks.append(ConvT_block(convT_struct=convT_block_struct, name=f"convT_block{i}"))
        self.output_layer = Output_convT_layer(output_layer_struct=output_layer_struct, name=f"Generator_output_layer")
    
    @tf.function
    def call(self, input):
        # Sample random vector
        #tf.random.set_seed(5)
        #input_vector = tf.random.normal(shape=[input_shape], mean=0.0, stddev=1.0)
        
        # Uprank the Input tensor. Need this after tensorflow 2.7
        # if input_noise.shape.rank == 1:
        #     input_noise = tf.expand_dims(input_noise, axis=0)
        # if type(input_cond) == "list":
        #     input_cond = tf.convert_to_tensor(input_cond)
        #input_cond = tf.expand_dims(input_cond, axis=0)       
        
        # NOTE: Implementing only transpose convolution layers
        Z = input

        for block in self.convT_blocks :
            Z = block(Z)
        
        return Z
