import is_valid_conv_layer_struct, is_valid_convT_layer_struct
import tensorflow as tf

class Conv_block(tf.keras.layers.Layer):
    def __init__(self, conv_block_struct, **kwargs):
        super(Conv_block, self).__init__(**kwargs)
        self.layers = []
        if not isinstance(conv_block_struct, list):
            raise exception(f"[EXCEPTION]: Provided conv_block_struct should be a list, instead it is \"{type(conv_block_struct)}\"")
        for i, layer_struct in enumerate(conv_block_struct):
            is_valid_convT_layer_struct(layer_struct)
            self.layers.append(tf.keras.layers.Conv2D(
                filters=layer_struct["filters"],
                kernel_size=layer_struct["kernel_size"],
                strides=layer_struct["strides"],
                padding=layer_struct["padding"],
                data_format=layer_struct["data_format"],
                dilation_rate=layer_struct["dilation_rate"],
                groups=layer_struct["groups"],
                activation=layer_struct["activation"],
                use_bias=layer_struct["use_bias"],
                kernel_initializer=layer_struct["kernel_initializer"],
                bias_initializer=layer_struct["bias_initializer"],
                kernel_regularizer=layer_struct["kernel_regularizer"],
                bias_regularizer=layer_struct["bias_regularizer"],
                activity_regularizer=layer_struct["activity_regularizer"],
                kernel_constraint=layer_struct["kernel_constraint"],
                bias_constraint=layer_struct["bias_constraint"],
                **kwargs
            ))
    
    @tf.function
    def call(self, input):
        z = input
        for layer in self.layers:
            z = layer(z)
        return z

class ConvT_block(tf.keras.layers.Layer):
    def __init__(self, convT_block_struct, **kwargs):
        super(ConvT_block, self).__init__(**kwargs)
        self.layers = []
        if not isinstance(convT_block_struct, list):
            raise exception(f"[EXCEPTION]: Provided convT_block_struct should be a list, instead it is \"{type(convT_block_struct)}\"")
        for i, layer_struct in enumerate(convT_block_struct):
            is_valid_convT_layer_struct(layer_struct)
            self.layers.append(tf.keras.layers.Conv2DTranspose(
                filters=layer_struct["filters"],
                kernel_size=layer_struct["kernel_size"],
                strides=layer_struct["strides"],
                padding=layer_struct["padding"],
                output_padding=layer_struct["output_padding"],
                data_format=layer_struct["data_format"],
                dilation_rate=layer_struct["dilation_rate"],
                activation=layer_struct["activation"],
                use_bias=layer_struct["use_bias"],
                kernel_initializer=layer_struct["kernel_initializer"],
                bias_initializer=layer_struct["bias_initializer"],
                kernel_regularizer=layer_struct["kernel_regularizer"],
                bias_regularizer=layer_struct["bias_regularizer"],
                activity_regularizer=layer_struct["activity_regularizer"],
                kernel_constraint=layer_struct["kernel_constraint"],
                bias_constraint=layer_struct["bias_constraint"],
                **kwargs
            ))
    
    @tf.function
    def call(self, input):
        z = input
        for layer in self.layers:
            z = layer(z)
        return z

class Output_conv_layer(tf.keras.layers.Layer):
    def __init__(self, output_layer_struct, **kwargs):
        super(Output_conv_layer, self).__init__(**kwargs)
        is_valid_conv_layer_struct(layer_struct)
        self.output_layer = tf.keras.layers.Conv2D(
                filters=layer_struct["filters"],
                kernel_size=layer_struct["kernel_size"],
                strides=layer_struct["strides"],
                padding=layer_struct["padding"],
                data_format=layer_struct["data_format"],
                dilation_rate=layer_struct["dilation_rate"],
                groups=layer_struct["groups"],
                activation=layer_struct["activation"],
                use_bias=layer_struct["use_bias"],
                kernel_initializer=layer_struct["kernel_initializer"],
                bias_initializer=layer_struct["bias_initializer"],
                kernel_regularizer=layer_struct["kernel_regularizer"],
                bias_regularizer=layer_struct["bias_regularizer"],
                activity_regularizer=layer_struct["activity_regularizer"],
                kernel_constraint=layer_struct["kernel_constraint"],
                bias_constraint=layer_struct["bias_constraint"],
                **kwargs
        )
    
    @tf.function
    def call(self, input):
        #print("Input Type Generator: ", type(input))
        return self.output_layer(input)

class Output_convT_layer(tf.keras.layers.Layer):
    def __init__(self, output_layer_struct, **kwargs):
        super(Output_conv_layer, self).__init__(**kwargs)
        is_valid_conv_layer_struct(layer_struct)
        self.output_layer = tf.keras.layers.Conv2DTranspose(
                filters=layer_struct["filters"],
                kernel_size=layer_struct["kernel_size"],
                strides=layer_struct["strides"],
                padding=layer_struct["padding"],
                output_padding=layer_struct["output_padding"],
                data_format=layer_struct["data_format"],
                dilation_rate=layer_struct["dilation_rate"],
                activation=layer_struct["activation"],
                use_bias=layer_struct["use_bias"],
                kernel_initializer=layer_struct["kernel_initializer"],
                bias_initializer=layer_struct["bias_initializer"],
                kernel_regularizer=layer_struct["kernel_regularizer"],
                bias_regularizer=layer_struct["bias_regularizer"],
                activity_regularizer=layer_struct["activity_regularizer"],
                kernel_constraint=layer_struct["kernel_constraint"],
                bias_constraint=layer_struct["bias_constraint"],
                **kwargs
        )
    
    @tf.function
    def call(self, input):
        #print("Input Type Generator: ", type(input))
        return self.output_layer(input)

class Sigmoid_layer(tf.keras.layers.Layer):
    def __init__(self, output_shape=(1,), **kwargs):
        super(Sigmoid_layer, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get("sigmoid")
        self.output_layer = tf.keras.layers.Dense(output_shape[0], activation=self.activation)
    
    @tf.function
    def call(self, input):
        return self.output_layer(input)

class Dense_block(tf.keras.layers.Layer):
    def __init__(self, dense_struct, **kwargs):
        super(Dense_block, self).__init__(**kwargs)
        self.layers = []
        for i, num in enumerate(dense_count):
            self.layers.append(tf.keras.layers.Dense(
                units=dense_struct["dense"],
                activation=dense_struct["activation"],
                use_bias=dense_struct["use_bias"],
                kernel_initializer=dense_struct["kernel_initializer"],
                bias_initializer=dense_struct["bias_initializer"],
                kernel_regularizer=dense_struct["kernel_regularizer"],
                bias_regularizer=dense_struct["bias_regularizer"],
                activity_regularizer=dense_struct["activity_regularizer"],
                kernel_constraint=dense_struct["kernel_constraint"],
                bias_constraint=dense_struct["bias_constraint"],
                name=f"dense_layer{i}",
                **kwargs
            ))
    
    @tf.function
    def call(self, input):
        z = input
        for i, layer in enumerate(self.layers):
            z = layer(z)
        return z
