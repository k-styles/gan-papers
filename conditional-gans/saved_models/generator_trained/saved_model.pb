??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
?
generator_dense0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?*(
shared_namegenerator_dense0/kernel
?
+generator_dense0/kernel/Read/ReadVariableOpReadVariableOpgenerator_dense0/kernel*
_output_shapes
:	d?*
dtype0
?
generator_dense0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_namegenerator_dense0/bias
|
)generator_dense0/bias/Read/ReadVariableOpReadVariableOpgenerator_dense0/bias*
_output_shapes	
:?*
dtype0
?
generator_dense0/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	
?**
shared_namegenerator_dense0/kernel_1
?
-generator_dense0/kernel_1/Read/ReadVariableOpReadVariableOpgenerator_dense0/kernel_1*
_output_shapes
:	
?*
dtype0
?
generator_dense0/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namegenerator_dense0/bias_1
?
+generator_dense0/bias_1/Read/ReadVariableOpReadVariableOpgenerator_dense0/bias_1*
_output_shapes	
:?*
dtype0
?
generator_dense0/kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?	**
shared_namegenerator_dense0/kernel_2
?
-generator_dense0/kernel_2/Read/ReadVariableOpReadVariableOpgenerator_dense0/kernel_2* 
_output_shapes
:
?	?	*
dtype0
?
generator_dense0/bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:?	*(
shared_namegenerator_dense0/bias_2
?
+generator_dense0/bias_2/Read/ReadVariableOpReadVariableOpgenerator_dense0/bias_2*
_output_shapes	
:?	*
dtype0
?
gen_out_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*%
shared_namegen_out_layer/kernel

(gen_out_layer/kernel/Read/ReadVariableOpReadVariableOpgen_out_layer/kernel* 
_output_shapes
:
?	?*
dtype0
}
gen_out_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_namegen_out_layer/bias
v
&gen_out_layer/bias/Read/ReadVariableOpReadVariableOpgen_out_layer/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?-
value?-B?- B?-
?
dense_noise_blocks
dense_cond_blocks
dense_body_blocks
output_layer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
call

signatures*

0*

0*

0*
?
output_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
call*
<
0
1
2
3
4
5
6
 7*
<
0
1
2
3
4
5
6
 7*
* 
?
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 
* 

&serving_default* 
?

'layers
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
.call*
?

/layers
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6call*
?

7layers
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>call*
?

kernel
 bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses*

0
 1*

0
 1*
* 
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
WQ
VARIABLE_VALUEgenerator_dense0/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEgenerator_dense0/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEgenerator_dense0/kernel_1&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgenerator_dense0/bias_1&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEgenerator_dense0/kernel_2&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgenerator_dense0/bias_2&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEgen_out_layer/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEgen_out_layer/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*
* 
* 
* 
* 

J0*

0
1*

0
1*
* 
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
* 

P0*

0
1*

0
1*
* 
?
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
* 

V0*

0
1*

0
1*
* 
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*
* 
* 
* 

0
 1*

0
 1*
* 
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*
* 
* 
* 

0*
* 
* 
* 
?

kernel
bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses*
* 

J0*
* 
* 
* 
?

kernel
bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses*
* 

P0*
* 
* 
* 
?

kernel
bias
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses*
* 

V0*
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
?
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
r
serving_default_input_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
r
serving_default_input_2Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2generator_dense0/kernelgenerator_dense0/biasgenerator_dense0/kernel_1generator_dense0/bias_1generator_dense0/kernel_2generator_dense0/bias_2gen_out_layer/kernelgen_out_layer/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_6255867
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+generator_dense0/kernel/Read/ReadVariableOp)generator_dense0/bias/Read/ReadVariableOp-generator_dense0/kernel_1/Read/ReadVariableOp+generator_dense0/bias_1/Read/ReadVariableOp-generator_dense0/kernel_2/Read/ReadVariableOp+generator_dense0/bias_2/Read/ReadVariableOp(gen_out_layer/kernel/Read/ReadVariableOp&gen_out_layer/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_6255995
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegenerator_dense0/kernelgenerator_dense0/biasgenerator_dense0/kernel_1generator_dense0/bias_1generator_dense0/kernel_2generator_dense0/bias_2gen_out_layer/kernelgen_out_layer/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_6256029??
?"
?
F__inference_generator_layer_call_and_return_conditional_losses_6255788
input_1
input_2-
noise_dense_block0_6255763:	d?)
noise_dense_block0_6255765:	?3
 conditional_dense_block0_6255768:	
?/
 conditional_dense_block0_6255770:	?-
body_dense_block0_6255775:
?	?	(
body_dense_block0_6255777:	?	2
generator_output_layer_6255780:
?	?-
generator_output_layer_6255782:	?
identity??)Body_dense_block0/StatefulPartitionedCall?0Conditional_dense_block0/StatefulPartitionedCall?.Generator_output_layer/StatefulPartitionedCall?*Noise_dense_block0/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : l

ExpandDims
ExpandDimsinput_1ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : p
ExpandDims_1
ExpandDimsinput_2ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:??????????
*Noise_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0noise_dense_block0_6255763noise_dense_block0_6255765*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_Noise_dense_block0_layer_call_and_return_conditional_losses_6255612?
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0 conditional_dense_block0_6255768 conditional_dense_block0_6255770*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6255629Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV23Noise_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	?	?
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_6255775body_dense_block0_6255777*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6255648?
.Generator_output_layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0generator_output_layer_6255780generator_output_layer_6255782*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_Generator_output_layer_layer_call_and_return_conditional_losses_6255665^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
ReshapeReshape7Generator_output_layer/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*
_output_shapes

:V
IdentityIdentityReshape:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall/^Generator_output_layer/StatefulPartitionedCall+^Noise_dense_block0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2`
.Generator_output_layer/StatefulPartitionedCall.Generator_output_layer/StatefulPartitionedCall2X
*Noise_dense_block0/StatefulPartitionedCall*Noise_dense_block0/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
"__inference__wrapped_model_6255588
input_1
input_2$
generator_6255570:	d? 
generator_6255572:	?$
generator_6255574:	
? 
generator_6255576:	?%
generator_6255578:
?	?	 
generator_6255580:	?	%
generator_6255582:
?	? 
generator_6255584:	?
identity??!generator/StatefulPartitionedCall?
!generator/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2generator_6255570generator_6255572generator_6255574generator_6255576generator_6255578generator_6255580generator_6255582generator_6255584*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321124p
IdentityIdentity*generator/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:j
NoOpNoOp"^generator/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????: : : : : : : : 2F
!generator/StatefulPartitionedCall!generator/StatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_2
?

?
+__inference_generator_layer_call_fn_6255810
inputs_0
inputs_1
unknown:	d?
	unknown_0:	?
	unknown_1:	
?
	unknown_2:	?
	unknown_3:
?	?	
	unknown_4:	?	
	unknown_5:
?	?
	unknown_6:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_generator_layer_call_and_return_conditional_losses_6255674f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
__inference_call_321115	
input@
,gen_out_layer_matmul_readvariableop_resource:
?	?<
-gen_out_layer_biasadd_readvariableop_resource:	?
identity??$gen_out_layer/BiasAdd/ReadVariableOp?#gen_out_layer/MatMul/ReadVariableOp?
#gen_out_layer/MatMul/ReadVariableOpReadVariableOp,gen_out_layer_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype0|
gen_out_layer/MatMulMatMulinput+gen_out_layer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
$gen_out_layer/BiasAdd/ReadVariableOpReadVariableOp-gen_out_layer_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
gen_out_layer/BiasAddBiasAddgen_out_layer/MatMul:product:0,gen_out_layer/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
gen_out_layer/SigmoidSigmoidgen_out_layer/BiasAdd:output:0*
T0*
_output_shapes
:	?`
IdentityIdentitygen_out_layer/Sigmoid:y:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp%^gen_out_layer/BiasAdd/ReadVariableOp$^gen_out_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?	: : 2L
$gen_out_layer/BiasAdd/ReadVariableOp$gen_out_layer/BiasAdd/ReadVariableOp2J
#gen_out_layer/MatMul/ReadVariableOp#gen_out_layer/MatMul/ReadVariableOp:F B

_output_shapes
:	?	

_user_specified_nameinput
?

?
%__inference_signature_wrapper_6255867
input_1
input_2
unknown:	d?
	unknown_0:	?
	unknown_1:	
?
	unknown_2:	?
	unknown_3:
?	?	
	unknown_4:	?	
	unknown_5:
?	?
	unknown_6:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_6255588f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
4__inference_Noise_dense_block0_layer_call_fn_6255896	
input
unknown:	d?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_Noise_dense_block0_layer_call_and_return_conditional_losses_6255612g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
? 
?
__inference_call_321509
inputs_0
inputs_1,
noise_dense_block0_321484:	d?(
noise_dense_block0_321486:	?2
conditional_dense_block0_321489:	
?.
conditional_dense_block0_321491:	?,
body_dense_block0_321496:
?	?	'
body_dense_block0_321498:	?	1
generator_output_layer_321501:
?	?,
generator_output_layer_321503:	?
identity??)Body_dense_block0/StatefulPartitionedCall?0Conditional_dense_block0/StatefulPartitionedCall?.Generator_output_layer/StatefulPartitionedCall?*Noise_dense_block0/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : m

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : q
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:??????????
*Noise_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0noise_dense_block0_321484noise_dense_block0_321486*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321065?
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0conditional_dense_block0_321489conditional_dense_block0_321491*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321081Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV23Noise_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	?	?
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_321496body_dense_block0_321498*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321099?
.Generator_output_layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0generator_output_layer_321501generator_output_layer_321503*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321115^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
ReshapeReshape7Generator_output_layer/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*
_output_shapes

:V
IdentityIdentityReshape:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall/^Generator_output_layer/StatefulPartitionedCall+^Noise_dense_block0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2`
.Generator_output_layer/StatefulPartitionedCall.Generator_output_layer/StatefulPartitionedCall2X
*Noise_dense_block0/StatefulPartitionedCall*Noise_dense_block0/StatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
O__inference_Noise_dense_block0_layer_call_and_return_conditional_losses_6255907	
inputB
/generator_dense0_matmul_readvariableop_resource:	d??
0generator_dense0_biasadd_readvariableop_resource:	?
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
S__inference_Generator_output_layer_layer_call_and_return_conditional_losses_6255665	
input@
,gen_out_layer_matmul_readvariableop_resource:
?	?<
-gen_out_layer_biasadd_readvariableop_resource:	?
identity??$gen_out_layer/BiasAdd/ReadVariableOp?#gen_out_layer/MatMul/ReadVariableOp?
#gen_out_layer/MatMul/ReadVariableOpReadVariableOp,gen_out_layer_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype0|
gen_out_layer/MatMulMatMulinput+gen_out_layer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
$gen_out_layer/BiasAdd/ReadVariableOpReadVariableOp-gen_out_layer_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
gen_out_layer/BiasAddBiasAddgen_out_layer/MatMul:product:0,gen_out_layer/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
gen_out_layer/SigmoidSigmoidgen_out_layer/BiasAdd:output:0*
T0*
_output_shapes
:	?`
IdentityIdentitygen_out_layer/Sigmoid:y:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp%^gen_out_layer/BiasAdd/ReadVariableOp$^gen_out_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?	: : 2L
$gen_out_layer/BiasAdd/ReadVariableOp$gen_out_layer/BiasAdd/ReadVariableOp2J
#gen_out_layer/MatMul/ReadVariableOp#gen_out_layer/MatMul/ReadVariableOp:F B

_output_shapes
:	?	

_user_specified_nameinput
?$
?
#__inference__traced_restore_6256029
file_prefix;
(assignvariableop_generator_dense0_kernel:	d?7
(assignvariableop_1_generator_dense0_bias:	??
,assignvariableop_2_generator_dense0_kernel_1:	
?9
*assignvariableop_3_generator_dense0_bias_1:	?@
,assignvariableop_4_generator_dense0_kernel_2:
?	?	9
*assignvariableop_5_generator_dense0_bias_2:	?	;
'assignvariableop_6_gen_out_layer_kernel:
?	?4
%assignvariableop_7_gen_out_layer_bias:	?

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp(assignvariableop_generator_dense0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp(assignvariableop_1_generator_dense0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_generator_dense0_kernel_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_generator_dense0_bias_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_generator_dense0_kernel_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp*assignvariableop_5_generator_dense0_bias_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp'assignvariableop_6_gen_out_layer_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp%assignvariableop_7_gen_out_layer_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_9IdentityIdentity_8:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*"
_acd_function_control_output(*
_output_shapes
 "!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
__inference_call_321595	
inputB
/generator_dense0_matmul_readvariableop_resource:	d??
0generator_dense0_biasadd_readvariableop_resource:	?
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:d: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:E A

_output_shapes

:d

_user_specified_nameinput
?
?
__inference_call_321637	
inputB
/generator_dense0_matmul_readvariableop_resource:	
??
0generator_dense0_biasadd_readvariableop_resource:	?
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:
: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:E A

_output_shapes

:


_user_specified_nameinput
?
?
__inference_call_321099	
inputC
/generator_dense0_matmul_readvariableop_resource:
?	?	?
0generator_dense0_biasadd_readvariableop_resource:	?	
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?	?
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?	j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?	j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	?	?
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?	: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:F B

_output_shapes
:	?	

_user_specified_nameinput
?
?
:__inference_Conditional_dense_block0_layer_call_fn_6255916	
input
unknown:	
?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6255629g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
__inference_call_321649	
inputB
/generator_dense0_matmul_readvariableop_resource:	
??
0generator_dense0_biasadd_readvariableop_resource:	?
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp\
generator_dense0/CastCastinput*

DstT0*

SrcT0*
_output_shapes

:
?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0?
generator_dense0/MatMulMatMulgenerator_dense0/Cast:y:0.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:
: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:E A

_output_shapes

:


_user_specified_nameinput
?

?
+__inference_generator_layer_call_fn_6255693
input_1
input_2
unknown:	d?
	unknown_0:	?
	unknown_1:	
?
	unknown_2:	?
	unknown_3:
?	?	
	unknown_4:	?	
	unknown_5:
?	?
	unknown_6:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_generator_layer_call_and_return_conditional_losses_6255674f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:?????????
!
_user_specified_name	input_1:LH
#
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
S__inference_Generator_output_layer_layer_call_and_return_conditional_losses_6255887	
input@
,gen_out_layer_matmul_readvariableop_resource:
?	?<
-gen_out_layer_biasadd_readvariableop_resource:	?
identity??$gen_out_layer/BiasAdd/ReadVariableOp?#gen_out_layer/MatMul/ReadVariableOp?
#gen_out_layer/MatMul/ReadVariableOpReadVariableOp,gen_out_layer_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype0|
gen_out_layer/MatMulMatMulinput+gen_out_layer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
$gen_out_layer/BiasAdd/ReadVariableOpReadVariableOp-gen_out_layer_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
gen_out_layer/BiasAddBiasAddgen_out_layer/MatMul:product:0,gen_out_layer/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
gen_out_layer/SigmoidSigmoidgen_out_layer/BiasAdd:output:0*
T0*
_output_shapes
:	?`
IdentityIdentitygen_out_layer/Sigmoid:y:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp%^gen_out_layer/BiasAdd/ReadVariableOp$^gen_out_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?	: : 2L
$gen_out_layer/BiasAdd/ReadVariableOp$gen_out_layer/BiasAdd/ReadVariableOp2J
#gen_out_layer/MatMul/ReadVariableOp#gen_out_layer/MatMul/ReadVariableOp:F B

_output_shapes
:	?	

_user_specified_nameinput
?
?
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6255947	
inputC
/generator_dense0_matmul_readvariableop_resource:
?	?	?
0generator_dense0_biasadd_readvariableop_resource:	?	
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?	?
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?	j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?	j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	?	?
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?	: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:F B

_output_shapes
:	?	

_user_specified_nameinput
?
?
__inference_call_321691	
inputC
/generator_dense0_matmul_readvariableop_resource:
?	?	?
0generator_dense0_biasadd_readvariableop_resource:	?	
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?	?
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?	j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?	j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	?	?
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?	: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:F B

_output_shapes
:	?	

_user_specified_nameinput
?
?
8__inference_Generator_output_layer_layer_call_fn_6255876	
input
unknown:
?	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_Generator_output_layer_layer_call_and_return_conditional_losses_6255665g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?	: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes
:	?	

_user_specified_nameinput
?"
?
F__inference_generator_layer_call_and_return_conditional_losses_6255674

inputs
inputs_1-
noise_dense_block0_6255613:	d?)
noise_dense_block0_6255615:	?3
 conditional_dense_block0_6255630:	
?/
 conditional_dense_block0_6255632:	?-
body_dense_block0_6255649:
?	?	(
body_dense_block0_6255651:	?	2
generator_output_layer_6255666:
?	?-
generator_output_layer_6255668:	?
identity??)Body_dense_block0/StatefulPartitionedCall?0Conditional_dense_block0/StatefulPartitionedCall?.Generator_output_layer/StatefulPartitionedCall?*Noise_dense_block0/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : k

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : q
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:??????????
*Noise_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0noise_dense_block0_6255613noise_dense_block0_6255615*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_Noise_dense_block0_layer_call_and_return_conditional_losses_6255612?
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0 conditional_dense_block0_6255630 conditional_dense_block0_6255632*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6255629Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV23Noise_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	?	?
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_6255649body_dense_block0_6255651*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6255648?
.Generator_output_layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0generator_output_layer_6255666generator_output_layer_6255668*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_Generator_output_layer_layer_call_and_return_conditional_losses_6255665^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
ReshapeReshape7Generator_output_layer/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*
_output_shapes

:V
IdentityIdentityReshape:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall/^Generator_output_layer/StatefulPartitionedCall+^Noise_dense_block0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2`
.Generator_output_layer/StatefulPartitionedCall.Generator_output_layer/StatefulPartitionedCall2X
*Noise_dense_block0/StatefulPartitionedCall*Noise_dense_block0/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6255927	
inputB
/generator_dense0_matmul_readvariableop_resource:	
??
0generator_dense0_biasadd_readvariableop_resource:	?
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
O__inference_Noise_dense_block0_layer_call_and_return_conditional_losses_6255612	
inputB
/generator_dense0_matmul_readvariableop_resource:	d??
0generator_dense0_biasadd_readvariableop_resource:	?
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
__inference_call_321660	
inputB
/generator_dense0_matmul_readvariableop_resource:	
??
0generator_dense0_biasadd_readvariableop_resource:	?
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_nameinput
? 
?
F__inference_generator_layer_call_and_return_conditional_losses_6255843
inputs_0
inputs_1-
noise_dense_block0_6255818:	d?)
noise_dense_block0_6255820:	?3
 conditional_dense_block0_6255823:	
?/
 conditional_dense_block0_6255825:	?-
body_dense_block0_6255830:
?	?	(
body_dense_block0_6255832:	?	2
generator_output_layer_6255835:
?	?-
generator_output_layer_6255837:	?
identity??)Body_dense_block0/StatefulPartitionedCall?0Conditional_dense_block0/StatefulPartitionedCall?.Generator_output_layer/StatefulPartitionedCall?*Noise_dense_block0/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : m

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : q
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:??????????
*Noise_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0noise_dense_block0_6255818noise_dense_block0_6255820*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321065?
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0 conditional_dense_block0_6255823 conditional_dense_block0_6255825*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321081Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV23Noise_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	?	?
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_6255830body_dense_block0_6255832*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321099?
.Generator_output_layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0generator_output_layer_6255835generator_output_layer_6255837*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321115^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
ReshapeReshape7Generator_output_layer/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*
_output_shapes

:V
IdentityIdentityReshape:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall/^Generator_output_layer/StatefulPartitionedCall+^Noise_dense_block0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2`
.Generator_output_layer/StatefulPartitionedCall.Generator_output_layer/StatefulPartitionedCall2X
*Noise_dense_block0/StatefulPartitionedCall*Noise_dense_block0/StatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6255648	
inputC
/generator_dense0_matmul_readvariableop_resource:
?	?	?
0generator_dense0_biasadd_readvariableop_resource:	?	
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource* 
_output_shapes
:
?	?	*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?	?
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?	*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?	j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?	j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	?	?
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?	: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:F B

_output_shapes
:	?	

_user_specified_nameinput
? 
?
__inference_call_321124

inputs
inputs_1,
noise_dense_block0_321066:	d?(
noise_dense_block0_321068:	?2
conditional_dense_block0_321082:	
?.
conditional_dense_block0_321084:	?,
body_dense_block0_321100:
?	?	'
body_dense_block0_321102:	?	1
generator_output_layer_321116:
?	?,
generator_output_layer_321118:	?
identity??)Body_dense_block0/StatefulPartitionedCall?0Conditional_dense_block0/StatefulPartitionedCall?.Generator_output_layer/StatefulPartitionedCall?*Noise_dense_block0/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : k

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : q
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:??????????
*Noise_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0noise_dense_block0_321066noise_dense_block0_321068*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321065?
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0conditional_dense_block0_321082conditional_dense_block0_321084*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321081Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV23Noise_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	?	?
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_321100body_dense_block0_321102*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321099?
.Generator_output_layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0generator_output_layer_321116generator_output_layer_321118*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321115^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
ReshapeReshape7Generator_output_layer/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*
_output_shapes

:V
IdentityIdentityReshape:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall/^Generator_output_layer/StatefulPartitionedCall+^Noise_dense_block0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2`
.Generator_output_layer/StatefulPartitionedCall.Generator_output_layer/StatefulPartitionedCall2X
*Noise_dense_block0/StatefulPartitionedCall*Noise_dense_block0/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_call_321065	
inputB
/generator_dense0_matmul_readvariableop_resource:	d??
0generator_dense0_biasadd_readvariableop_resource:	?
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
__inference_call_321476
inputs_0
inputs_1,
noise_dense_block0_321439:	d?(
noise_dense_block0_321441:	?2
conditional_dense_block0_321456:	
?.
conditional_dense_block0_321458:	?,
body_dense_block0_321463:
?	?	'
body_dense_block0_321465:	?	1
generator_output_layer_321468:
?	?,
generator_output_layer_321470:	?
identity??)Body_dense_block0/StatefulPartitionedCall?0Conditional_dense_block0/StatefulPartitionedCall?.Generator_output_layer/StatefulPartitionedCall?*Noise_dense_block0/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : d

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*
_output_shapes

:dR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : h
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*
_output_shapes

:
?
*Noise_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0noise_dense_block0_321439noise_dense_block0_321441*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321065?
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0conditional_dense_block0_321456conditional_dense_block0_321458*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321455Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV23Noise_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	?	?
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_321463body_dense_block0_321465*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321099?
.Generator_output_layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0generator_output_layer_321468generator_output_layer_321470*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321115^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
ReshapeReshape7Generator_output_layer/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*
_output_shapes

:V
IdentityIdentityReshape:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall/^Generator_output_layer/StatefulPartitionedCall+^Noise_dense_block0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:d:
: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2`
.Generator_output_layer/StatefulPartitionedCall.Generator_output_layer/StatefulPartitionedCall2X
*Noise_dense_block0/StatefulPartitionedCall*Noise_dense_block0/StatefulPartitionedCall:D @

_output_shapes
:d
"
_user_specified_name
inputs/0:D@

_output_shapes
:

"
_user_specified_name
inputs/1
?
?
3__inference_Body_dense_block0_layer_call_fn_6255936	
input
unknown:
?	?	
	unknown_0:	?	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6255648g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	?	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?	: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes
:	?	

_user_specified_nameinput
?
?
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6255629	
inputB
/generator_dense0_matmul_readvariableop_resource:	
??
0generator_dense0_biasadd_readvariableop_resource:	?
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
 __inference__traced_save_6255995
file_prefix6
2savev2_generator_dense0_kernel_read_readvariableop4
0savev2_generator_dense0_bias_read_readvariableop8
4savev2_generator_dense0_kernel_1_read_readvariableop6
2savev2_generator_dense0_bias_1_read_readvariableop8
4savev2_generator_dense0_kernel_2_read_readvariableop6
2savev2_generator_dense0_bias_2_read_readvariableop3
/savev2_gen_out_layer_kernel_read_readvariableop1
-savev2_gen_out_layer_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_generator_dense0_kernel_read_readvariableop0savev2_generator_dense0_bias_read_readvariableop4savev2_generator_dense0_kernel_1_read_readvariableop2savev2_generator_dense0_bias_1_read_readvariableop4savev2_generator_dense0_kernel_2_read_readvariableop2savev2_generator_dense0_bias_2_read_readvariableop/savev2_gen_out_layer_kernel_read_readvariableop-savev2_gen_out_layer_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*a
_input_shapesP
N: :	d?:?:	
?:?:
?	?	:?	:
?	?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	d?:!

_output_shapes	
:?:%!

_output_shapes
:	
?:!

_output_shapes	
:?:&"
 
_output_shapes
:
?	?	:!

_output_shapes	
:?	:&"
 
_output_shapes
:
?	?:!

_output_shapes	
:?:	

_output_shapes
: 
?
?
__inference_call_321606	
inputB
/generator_dense0_matmul_readvariableop_resource:	d??
0generator_dense0_biasadd_readvariableop_resource:	?
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource*
_output_shapes
:	d?*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
__inference_call_321431
inputs_0
inputs_1,
noise_dense_block0_321406:	d?(
noise_dense_block0_321408:	?2
conditional_dense_block0_321411:	
?.
conditional_dense_block0_321413:	?,
body_dense_block0_321418:
?	?	'
body_dense_block0_321420:	?	1
generator_output_layer_321423:
?	?,
generator_output_layer_321425:	?
identity??)Body_dense_block0/StatefulPartitionedCall?0Conditional_dense_block0/StatefulPartitionedCall?.Generator_output_layer/StatefulPartitionedCall?*Noise_dense_block0/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : d

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*
_output_shapes

:dR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : h
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*
_output_shapes

:
?
*Noise_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims:output:0noise_dense_block0_321406noise_dense_block0_321408*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321065?
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0conditional_dense_block0_321411conditional_dense_block0_321413*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321081Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV23Noise_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	?	?
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_321418body_dense_block0_321420*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321099?
.Generator_output_layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0generator_output_layer_321423generator_output_layer_321425*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? * 
fR
__inference_call_321115^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
ReshapeReshape7Generator_output_layer/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*
_output_shapes

:V
IdentityIdentityReshape:output:0^NoOp*
T0*
_output_shapes

:?
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall/^Generator_output_layer/StatefulPartitionedCall+^Noise_dense_block0/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:d:
: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2`
.Generator_output_layer/StatefulPartitionedCall.Generator_output_layer/StatefulPartitionedCall2X
*Noise_dense_block0/StatefulPartitionedCall*Noise_dense_block0/StatefulPartitionedCall:D @

_output_shapes
:d
"
_user_specified_name
inputs/0:D@

_output_shapes
:

"
_user_specified_name
inputs/1
?
?
__inference_call_321564	
input@
,gen_out_layer_matmul_readvariableop_resource:
?	?<
-gen_out_layer_biasadd_readvariableop_resource:	?
identity??$gen_out_layer/BiasAdd/ReadVariableOp?#gen_out_layer/MatMul/ReadVariableOp?
#gen_out_layer/MatMul/ReadVariableOpReadVariableOp,gen_out_layer_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype0|
gen_out_layer/MatMulMatMulinput+gen_out_layer/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
$gen_out_layer/BiasAdd/ReadVariableOpReadVariableOp-gen_out_layer_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
gen_out_layer/BiasAddBiasAddgen_out_layer/MatMul:product:0,gen_out_layer/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
gen_out_layer/SigmoidSigmoidgen_out_layer/BiasAdd:output:0*
T0*
_output_shapes
:	?`
IdentityIdentitygen_out_layer/Sigmoid:y:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp%^gen_out_layer/BiasAdd/ReadVariableOp$^gen_out_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	?	: : 2L
$gen_out_layer/BiasAdd/ReadVariableOp$gen_out_layer/BiasAdd/ReadVariableOp2J
#gen_out_layer/MatMul/ReadVariableOp#gen_out_layer/MatMul/ReadVariableOp:F B

_output_shapes
:	?	

_user_specified_nameinput
?
?
__inference_call_321081	
inputB
/generator_dense0_matmul_readvariableop_resource:	
??
0generator_dense0_biasadd_readvariableop_resource:	?
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0?
generator_dense0/MatMulMatMulinput.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????

_user_specified_nameinput
?
?
__inference_call_321455	
inputB
/generator_dense0_matmul_readvariableop_resource:	
??
0generator_dense0_biasadd_readvariableop_resource:	?
identity??'generator_dense0/BiasAdd/ReadVariableOp?&generator_dense0/MatMul/ReadVariableOp\
generator_dense0/CastCastinput*

DstT0*

SrcT0*
_output_shapes

:
?
&generator_dense0/MatMul/ReadVariableOpReadVariableOp/generator_dense0_matmul_readvariableop_resource*
_output_shapes
:	
?*
dtype0?
generator_dense0/MatMulMatMulgenerator_dense0/Cast:y:0.generator_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	??
'generator_dense0/BiasAdd/ReadVariableOpReadVariableOp0generator_dense0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
generator_dense0/BiasAddBiasAdd!generator_dense0/MatMul:product:0/generator_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?j
generator_dense0/ReluRelu!generator_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	?j
IdentityIdentity#generator_dense0/Relu:activations:0^NoOp*
T0*
_output_shapes
:	??
NoOpNoOp(^generator_dense0/BiasAdd/ReadVariableOp'^generator_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:
: : 2R
'generator_dense0/BiasAdd/ReadVariableOp'generator_dense0/BiasAdd/ReadVariableOp2P
&generator_dense0/MatMul/ReadVariableOp&generator_dense0/MatMul/ReadVariableOp:E A

_output_shapes

:


_user_specified_nameinput"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
input_1,
serving_default_input_1:0?????????
7
input_2,
serving_default_input_2:0?????????3
output_1'
StatefulPartitionedCall:0tensorflow/serving/predict:??
?
dense_noise_blocks
dense_cond_blocks
dense_body_blocks
output_layer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
call

signatures"
_tf_keras_model
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
output_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
call"
_tf_keras_layer
X
0
1
2
3
4
5
6
 7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
 7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_generator_layer_call_fn_6255693
+__inference_generator_layer_call_fn_6255810?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_generator_layer_call_and_return_conditional_losses_6255843
F__inference_generator_layer_call_and_return_conditional_losses_6255788?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference__wrapped_model_6255588input_1input_2"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_call_321431
__inference_call_321476
__inference_call_321509?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
&serving_default"
signature_map
?

'layers
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses
.call"
_tf_keras_layer
?

/layers
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6call"
_tf_keras_layer
?

7layers
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses
>call"
_tf_keras_layer
?

kernel
 bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_Generator_output_layer_layer_call_fn_6255876?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_Generator_output_layer_layer_call_and_return_conditional_losses_6255887?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_call_321564?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
*:(	d?2generator_dense0/kernel
$:"?2generator_dense0/bias
*:(	
?2generator_dense0/kernel
$:"?2generator_dense0/bias
+:)
?	?	2generator_dense0/kernel
$:"?	2generator_dense0/bias
(:&
?	?2gen_out_layer/kernel
!:?2gen_out_layer/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_signature_wrapper_6255867input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
J0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
?2?
4__inference_Noise_dense_block0_layer_call_fn_6255896?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_Noise_dense_block0_layer_call_and_return_conditional_losses_6255907?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_call_321595
__inference_call_321606?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
P0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
?2?
:__inference_Conditional_dense_block0_layer_call_fn_6255916?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6255927?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_call_321637
__inference_call_321649
__inference_call_321660?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
'
V0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
?2?
3__inference_Body_dense_block0_layer_call_fn_6255936?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6255947?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_call_321691?
???
FullArgSpec
args?
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

kernel
bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
'
J0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

kernel
bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

kernel
bias
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
'
V0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper?
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6255947K&?#
?
?
input	?	
? "?
?
0	?	
? u
3__inference_Body_dense_block0_layer_call_fn_6255936>&?#
?
?
input	?	
? "?	?	?
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6255927S.?+
$?!
?
input?????????
? "?
?
0	?
? ?
:__inference_Conditional_dense_block0_layer_call_fn_6255916F.?+
$?!
?
input?????????
? "?	??
S__inference_Generator_output_layer_layer_call_and_return_conditional_losses_6255887K &?#
?
?
input	?	
? "?
?
0	?
? z
8__inference_Generator_output_layer_layer_call_fn_6255876> &?#
?
?
input	?	
? "?	??
O__inference_Noise_dense_block0_layer_call_and_return_conditional_losses_6255907S.?+
$?!
?
input?????????
? "?
?
0	?
? ~
4__inference_Noise_dense_block0_layer_call_fn_6255896F.?+
$?!
?
input?????????
? "?	??
"__inference__wrapped_model_6255588? P?M
F?C
A?>
?
input_1?????????
?
input_2?????????
? "*?'
%
output_1?
output_1x
__inference_call_321431] @?=
6?3
1?.
?
inputs/0d
?
inputs/1

? "?x
__inference_call_321476] @?=
6?3
1?.
?
inputs/0d
?
inputs/1

? "??
__inference_call_321509o R?O
H?E
C?@
?
inputs/0?????????
?
inputs/1?????????
? "?Y
__inference_call_321564> &?#
?
?
input	?	
? "?	?X
__inference_call_321595=%?"
?
?
inputd
? "?	?a
__inference_call_321606F.?+
$?!
?
input?????????
? "?	?X
__inference_call_321637=%?"
?
?
input

? "?	?X
__inference_call_321649=%?"
?
?
input

? "?	?a
__inference_call_321660F.?+
$?!
?
input?????????
? "?	?Y
__inference_call_321691>&?#
?
?
input	?	
? "?	?	?
F__inference_generator_layer_call_and_return_conditional_losses_6255788z P?M
F?C
A?>
?
input_1?????????
?
input_2?????????
? "?
?
0
? ?
F__inference_generator_layer_call_and_return_conditional_losses_6255843| R?O
H?E
C?@
?
inputs/0?????????
?
inputs/1?????????
? "?
?
0
? ?
+__inference_generator_layer_call_fn_6255693m P?M
F?C
A?>
?
input_1?????????
?
input_2?????????
? "??
+__inference_generator_layer_call_fn_6255810o R?O
H?E
C?@
?
inputs/0?????????
?
inputs/1?????????
? "??
%__inference_signature_wrapper_6255867? a?^
? 
W?T
(
input_1?
input_1?????????
(
input_2?
input_2?????????"*?'
%
output_1?
output_1