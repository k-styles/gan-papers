def is_valid_conv_layer_struct(conv_layer_struct):
    expected_keys = ["filters", "kernel_size", "strides", "padding", "data_format", "dilation_rate", "groups", 
                     "activation", "use_bias", "kernel_initializer", "bias_initializer", "kernel_regularizer", "bias_regularizer", 
                     "activity_regularizer", "kernel_constraint", "bias_constraint"]
    for expected_key in expected_keys:
        if expected_key not in conv_gen_struct.keys():
                raise exception("[EXCEPTION]: Expected key \"{expected_key}\" not passed in the layer's dictionary sructure.")

def is_valid_convT_layer_struct(conv_layer_struct):
    expected_keys = ["filters", "kernel_size", "strides", "padding", "output_padding", "data_format", "dilation_rate",
                     "activation", "use_bias", "kernel_initializer", "bias_initializer", "kernel_regularizer", "bias_regularizer", 
                     "activity_regularizer", "kernel_constraint", "bias_constraint"]
    for expected_key in expected_keys:
        if expected_key not in conv_gen_struct.keys():
                raise exception("[EXCEPTION]: Expected key \"{expected_key}\" not passed in the layer's dictionary sructure.")