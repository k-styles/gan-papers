# This file contains helping functions for layer structs
def is_valid_conv_layer_struct(conv_layer_struct):
	if not isinstance(conv_layer_struct, dict):
        	raise exception ("[EXCEPTION]: The structure should be of type dict")
    expected_keys = ["filters", "kernel_size", "strides", "padding", "data_format", "dilation_rate", "groups", 
                     "activation", "use_bias", "kernel_initializer", "bias_initializer", "kernel_regularizer", "bias_regularizer", 
                     "activity_regularizer", "kernel_constraint", "bias_constraint"]
    
	for available_key in conv_layer_struct.keys():
		if available_key not in expected_keys:
                	raise exception(f"[EXCEPTION]: Provided key \"{available_key}\" is not expected.")

	for expected_key in expected_keys:
        	if expected_key not in conv_layer_struct.keys():
			if expected_key == "filters":
				raise exception(f"\"filters\" is a mandatory entry.")
			if expected_key == "kernel_size":
				raise exception(f"\"kernel_size\" is a mandatory entry.")
			if expected_key == "strides":
				conv_layer_struct["strides"] = (1,1)
			if expected_key == "padding":
				conv_layer_struct["padding"] = "valid"
			if expected_key == "data_format":
				conv_layer_struct["data_format"] = None
			if expected_key == "dilation_rate":
				conv_layer_struct["dilation_rate"] = (1,1)
			if expected_key == "groups":
				conv_layer_struct["groups"] = 1
			if expected_key == "activation":
				conv_layer_struct["activation"] = None
			if expected_key == "use_bias":
				conv_layer_struct["use_bias"] = True
			if expected_key == "kernel_initializer":
				conv_layer_struct["kernel_initializer"] = 'glorot_uniform'
			if expected_key == "bias_initializer":
				conv_layer_struct["bias_initializer"] = 'zeros'
			if expected_key == "kernel_regularizer":
				conv_layer_struct["kernel_regularizer"] = None
			if expected_key == "bias_regularizer":
				conv_layer_struct["bias_regularizer"] = None
			if expected_key == "activity_regularizer":
				conv_layer_struct["activity_regularizer"] = None
			if expected_key == "kernel_constraint":
				conv_layer_struct["kernel_constraint"] = None
			if expected_key == "bias_constraint":
				conv_layer_struct["bias_constraint"] = None

def is_valid_convT_layer_struct(convT_layer_struct):
    	if not isinstance(convT_layer_struct, dict):
        	raise exception ("[EXCEPTION]: The structure should be of type dict")
    expected_keys = ["filters", "kernel_size", "strides", "padding", "output_padding", "data_format", "dilation_rate",
                     "activation", "use_bias", "kernel_initializer", "bias_initializer", "kernel_regularizer", "bias_regularizer", 
                     "activity_regularizer", "kernel_constraint", "bias_constraint"]
	
	for available_key in convT_layer_struct.keys():
		if available_key not in expected_keys:
                	raise exception(f"[EXCEPTION]: Provided key \"{available_key}\" is not expected.")

	for expected_key in expected_keys:
        	if expected_key not in convT_layer_struct.keys():
			if expected_key == "filters":
				raise exception(f"\"filters\" is a mandatory entry.")
			if expected_key == "kernel_size":
				raise exception(f"\"kernel_size\" is a mandatory entry.")
			if expected_key == "strides":
				convT_layer_struct["strides"] = (1,1)
			if expected_key == "padding":
				convT_layer_struct["padding"] = "valid"
			if expected_key == "output_padding":
				convT_layer_struct["output_padding"] = None
			if expected_key == "data_format":
				convT_layer_struct["data_format"] = None
			if expected_key == "dilation_rate":
				convT_layer_struct["dilation_rate"] = (1,1)
			if expected_key == "activation":
				convT_layer_struct["activation"] = None
			if expected_key == "use_bias":
				convT_layer_struct["use_bias"] = True
			if expected_key == "kernel_initializer":
				convT_layer_struct["kernel_initializer"] = 'glorot_uniform'
			if expected_key == "bias_initializer":
				convT_layer_struct["bias_initializer"] = 'zeros'
			if expected_key == "kernel_regularizer":
				convT_layer_struct["kernel_regularizer"] = None
			if expected_key == "bias_regularizer":
				convT_layer_struct["bias_regularizer"] = None
			if expected_key == "activity_regularizer":
				convT_layer_struct["activity_regularizer"] = None
			if expected_key == "kernel_constraint":
				convT_layer_struct["kernel_constraint"] = None
			if expected_key == "bias_constraint":
				convT_layer_struct["bias_constraint"] = None
