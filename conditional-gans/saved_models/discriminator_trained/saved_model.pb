ÀÃ
²
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68é

discrim_dense0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ð*&
shared_namediscrim_dense0/kernel

)discrim_dense0/kernel/Read/ReadVariableOpReadVariableOpdiscrim_dense0/kernel* 
_output_shapes
:
ð*
dtype0

discrim_dense0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*$
shared_namediscrim_dense0/bias
x
'discrim_dense0/bias/Read/ReadVariableOpReadVariableOpdiscrim_dense0/bias*
_output_shapes	
:ð*
dtype0

discrim_dense0/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:
2*(
shared_namediscrim_dense0/kernel_1

+discrim_dense0/kernel_1/Read/ReadVariableOpReadVariableOpdiscrim_dense0/kernel_1*
_output_shapes

:
2*
dtype0

discrim_dense0/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:2*&
shared_namediscrim_dense0/bias_1
{
)discrim_dense0/bias_1/Read/ReadVariableOpReadVariableOpdiscrim_dense0/bias_1*
_output_shapes
:2*
dtype0

discrim_dense0/kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:
¢ð*(
shared_namediscrim_dense0/kernel_2

+discrim_dense0/kernel_2/Read/ReadVariableOpReadVariableOpdiscrim_dense0/kernel_2* 
_output_shapes
:
¢ð*
dtype0

discrim_dense0/bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:ð*&
shared_namediscrim_dense0/bias_2
|
)discrim_dense0/bias_2/Read/ReadVariableOpReadVariableOpdiscrim_dense0/bias_2*
_output_shapes	
:ð*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ð*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	ð*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0

NoOpNoOp
µ=
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ð<
valueæ<Bã< BÜ<
µ
dense_img_blocks
dense_cond_blocks
dense_body_blocks
flatten_layer
out_sigmoid_layer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
call

signatures*

0*

0*

0*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¬
output_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
call*
<
 0
!1
"2
#3
$4
%5
&6
'7*
<
 0
!1
"2
#3
$4
%5
&6
'7*
* 
°
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 

-serving_default* 
¦

.layers
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5call*
¦

6layers
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=call*
¦

>layers
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
Ecall*
* 
* 
* 

Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
¦

&kernel
'bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses*

&0
'1*

&0
'1*
* 

Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
UO
VARIABLE_VALUEdiscrim_dense0/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdiscrim_dense0/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdiscrim_dense0/kernel_1&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdiscrim_dense0/bias_1&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdiscrim_dense0/kernel_2&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdiscrim_dense0/bias_2&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*
* 
* 
* 
* 

V0
W1*

 0
!1*

 0
!1*
* 

Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
* 
* 
* 

]0
^1*

"0
#1*

"0
#1*
* 

_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
* 
* 
* 

d0
e1*

$0
%1*

$0
%1*
* 

fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 

&0
'1*

&0
'1*
* 

knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*
* 
* 
* 

0*
* 
* 
* 
¦

 kernel
!bias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses*

v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 
* 

V0
W1*
* 
* 
* 
¨

"kernel
#bias
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
* 

]0
^1*
* 
* 
* 
¬

$kernel
%bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
* 

d0
e1*
* 
* 
* 
* 
* 
* 
* 
* 

 0
!1*

 0
!1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 
* 
* 

"0
#1*

"0
#1*
* 

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 

$0
%1*

$0
%1*
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
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
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
r
serving_default_input_2Placeholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ý
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2discrim_dense0/kerneldiscrim_dense0/biasdiscrim_dense0/kernel_1discrim_dense0/bias_1discrim_dense0/kernel_2discrim_dense0/bias_2dense/kernel
dense/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_6256401
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
í
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)discrim_dense0/kernel/Read/ReadVariableOp'discrim_dense0/bias/Read/ReadVariableOp+discrim_dense0/kernel_1/Read/ReadVariableOp)discrim_dense0/bias_1/Read/ReadVariableOp+discrim_dense0/kernel_2/Read/ReadVariableOp)discrim_dense0/bias_2/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_6256552
È
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamediscrim_dense0/kerneldiscrim_dense0/biasdiscrim_dense0/kernel_1discrim_dense0/bias_1discrim_dense0/kernel_2discrim_dense0/bias_2dense/kernel
dense/bias*
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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_6256586¦©
¨
Á
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6256156	
input?
-discrim_dense0_matmul_readvariableop_resource:
2<
.discrim_dense0_biasadd_readvariableop_resource:2
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource*
_output_shapes

:
2*
dtype0}
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2e
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes

:2k
maxout_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      2   
maxout_1/ReshapeReshape!discrim_dense0/Relu:activations:0maxout_1/Reshape/shape:output:0*
T0*"
_output_shapes
:2`
maxout_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
maxout_1/MaxMaxmaxout_1/Reshape:output:0'maxout_1/Max/reduction_indices:output:0*
T0*
_output_shapes

:2[
IdentityIdentitymaxout_1/Max:output:0^NoOp*
T0*
_output_shapes

:2
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
¾

à
__inference_call_322391	
input7
$dense_matmul_readvariableop_resource:	ð3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ð*
dtype0k
dense/MatMulMatMulinput#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*
_output_shapes

:W
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	ð: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:F B

_output_shapes
:	ð

_user_specified_nameinput

½
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6256504	
inputA
-discrim_dense0_matmul_readvariableop_resource:
¢ð=
.discrim_dense0_biasadd_readvariableop_resource:	ð
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource* 
_output_shapes
:
¢ð*
dtype0~
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ð
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ðf
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	ðk
maxout_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ð   
maxout_2/ReshapeReshape!discrim_dense0/Relu:activations:0maxout_2/Reshape/shape:output:0*
T0*#
_output_shapes
:ð`
maxout_2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
maxout_2/MaxMaxmaxout_2/Reshape:output:0'maxout_2/Max/reduction_indices:output:0*
T0*
_output_shapes
:	ð\
IdentityIdentitymaxout_2/Max:output:0^NoOp*
T0*
_output_shapes
:	ð
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	¢: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:F B

_output_shapes
:	¢

_user_specified_nameinput
­

/__inference_Sigmoid_Layer_layer_call_fn_6256421	
input
unknown:	ð
	unknown_0:
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Sigmoid_Layer_layer_call_and_return_conditional_losses_6256196f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	ð: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes
:	ð

_user_specified_nameinput


É
%__inference_signature_wrapper_6256401
input_1
input_2
unknown:
ð
	unknown_0:	ð
	unknown_1:
2
	unknown_2:2
	unknown_3:
¢ð
	unknown_4:	ð
	unknown_5:	ð
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_6256099f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:LH
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
Õ

__inference_call_322430	
inputA
-discrim_dense0_matmul_readvariableop_resource:
ð=
.discrim_dense0_biasadd_readvariableop_resource:	ð
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource* 
_output_shapes
:
ð*
dtype0~
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ð
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ðf
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	ði
maxout/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ð   
maxout/ReshapeReshape!discrim_dense0/Relu:activations:0maxout/Reshape/shape:output:0*
T0*#
_output_shapes
:ð^
maxout/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :{

maxout/MaxMaxmaxout/Reshape:output:0%maxout/Max/reduction_indices:output:0*
T0*
_output_shapes
:	ðZ
IdentityIdentitymaxout/Max:output:0^NoOp*
T0*
_output_shapes
:	ð
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:F B

_output_shapes
:	

_user_specified_nameinput
í
Ù
__inference_call_322325
inputs_0
inputs_1-
image_dense_block0_322302:
ð(
image_dense_block0_322304:	ð1
conditional_dense_block0_322307:
2-
conditional_dense_block0_322309:2,
body_dense_block0_322314:
¢ð'
body_dense_block0_322316:	ð'
sigmoid_layer_322319:	ð"
sigmoid_layer_322321:
identity¢)Body_dense_block0/StatefulPartitionedCall¢0Conditional_dense_block0/StatefulPartitionedCall¢*Image_dense_block0/StatefulPartitionedCall¢%Sigmoid_Layer/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : q
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿy
flatten/ReshapeReshapeExpandDims:output:0flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿî
*Image_dense_block0/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0image_dense_block0_322302image_dense_block0_322304*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321848ü
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0conditional_dense_block0_322307conditional_dense_block0_322309*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321868Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ã
concatenate/concatConcatV23Image_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	¢î
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_322314body_dense_block0_322316*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321890ø
%Sigmoid_Layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0sigmoid_layer_322319sigmoid_layer_322321*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321906t
IdentityIdentity.Sigmoid_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall+^Image_dense_block0/StatefulPartitionedCall&^Sigmoid_Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2X
*Image_dense_block0/StatefulPartitionedCall*Image_dense_block0/StatefulPartitionedCall2N
%Sigmoid_Layer/StatefulPartitionedCall%Sigmoid_Layer/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ñ


J__inference_Sigmoid_Layer_layer_call_and_return_conditional_losses_6256432	
input7
$dense_matmul_readvariableop_resource:	ð3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ð*
dtype0k
dense/MatMulMatMulinput#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*
_output_shapes

:W
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	ð: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:F B

_output_shapes
:	ð

_user_specified_nameinput
å

__inference_call_322445	
inputA
-discrim_dense0_matmul_readvariableop_resource:
ð=
.discrim_dense0_biasadd_readvariableop_resource:	ð
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource* 
_output_shapes
:
ð*
dtype0~
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ð
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ðf
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	ði
maxout/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ð   
maxout/ReshapeReshape!discrim_dense0/Relu:activations:0maxout/Reshape/shape:output:0*
T0*#
_output_shapes
:ð^
maxout/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :{

maxout/MaxMaxmaxout/Reshape:output:0%maxout/Max/reduction_indices:output:0*
T0*
_output_shapes
:	ðZ
IdentityIdentitymaxout/Max:output:0^NoOp*
T0*
_output_shapes
:	ð
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
ß#
²
#__inference__traced_restore_6256586
file_prefix:
&assignvariableop_discrim_dense0_kernel:
ð5
&assignvariableop_1_discrim_dense0_bias:	ð<
*assignvariableop_2_discrim_dense0_kernel_1:
26
(assignvariableop_3_discrim_dense0_bias_1:2>
*assignvariableop_4_discrim_dense0_kernel_2:
¢ð7
(assignvariableop_5_discrim_dense0_bias_2:	ð2
assignvariableop_6_dense_kernel:	ð+
assignvariableop_7_dense_bias:

identity_9¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7Í
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ó
valueéBæ	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B Ë
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp&assignvariableop_discrim_dense0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp&assignvariableop_1_discrim_dense0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp*assignvariableop_2_discrim_dense0_kernel_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp(assignvariableop_3_discrim_dense0_bias_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp*assignvariableop_4_discrim_dense0_kernel_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp(assignvariableop_5_discrim_dense0_bias_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_9IdentityIdentity_8:output:0^NoOp_1*
T0*
_output_shapes
: î
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

¾
O__inference_Image_dense_block0_layer_call_and_return_conditional_losses_6256456	
inputA
-discrim_dense0_matmul_readvariableop_resource:
ð=
.discrim_dense0_biasadd_readvariableop_resource:	ð
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource* 
_output_shapes
:
ð*
dtype0~
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ð
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ðf
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	ði
maxout/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ð   
maxout/ReshapeReshape!discrim_dense0/Relu:activations:0maxout/Reshape/shape:output:0*
T0*#
_output_shapes
:ð^
maxout/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :{

maxout/MaxMaxmaxout/Reshape:output:0%maxout/Max/reduction_indices:output:0*
T0*
_output_shapes
:	ðZ
IdentityIdentitymaxout/Max:output:0^NoOp*
T0*
_output_shapes
:	ð
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
¯
Ù
__inference_call_322243
inputs_0
inputs_1-
image_dense_block0_322220:
ð(
image_dense_block0_322222:	ð1
conditional_dense_block0_322225:
2-
conditional_dense_block0_322227:2,
body_dense_block0_322232:
¢ð'
body_dense_block0_322234:	ð'
sigmoid_layer_322237:	ð"
sigmoid_layer_322239:
identity¢)Body_dense_block0/StatefulPartitionedCall¢0Conditional_dense_block0/StatefulPartitionedCall¢*Image_dense_block0/StatefulPartitionedCall¢%Sigmoid_Layer/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : h

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*"
_output_shapes
:R
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
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  q
flatten/ReshapeReshapeExpandDims:output:0flatten/Const:output:0*
T0*
_output_shapes
:	î
*Image_dense_block0/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0image_dense_block0_322220image_dense_block0_322222*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321848ü
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0conditional_dense_block0_322225conditional_dense_block0_322227*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321868Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ã
concatenate/concatConcatV23Image_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	¢î
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_322232body_dense_block0_322234*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321890ø
%Sigmoid_Layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0sigmoid_layer_322237sigmoid_layer_322239*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321906t
IdentityIdentity.Sigmoid_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall+^Image_dense_block0/StatefulPartitionedCall&^Sigmoid_Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 ::
: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2X
*Image_dense_block0/StatefulPartitionedCall*Image_dense_block0/StatefulPartitionedCall2N
%Sigmoid_Layer/StatefulPartitionedCall%Sigmoid_Layer/StatefulPartitionedCall:H D

_output_shapes

:
"
_user_specified_name
inputs/0:D@

_output_shapes
:

"
_user_specified_name
inputs/1
¼
`
D__inference_flatten_layer_call_and_return_conditional_losses_6256118

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ\
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
¢
3__inference_Body_dense_block0_layer_call_fn_6256489	
input
unknown:
¢ð
	unknown_0:	ð
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6256179g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	ð`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	¢: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes
:	¢

_user_specified_nameinput
Ç

__inference_call_322273	
input?
-discrim_dense0_matmul_readvariableop_resource:
2<
.discrim_dense0_biasadd_readvariableop_resource:2
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOpZ
discrim_dense0/CastCastinput*

DstT0*

SrcT0*
_output_shapes

:

$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource*
_output_shapes

:
2*
dtype0
discrim_dense0/MatMulMatMuldiscrim_dense0/Cast:y:0,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2e
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes

:2k
maxout_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      2   
maxout_1/ReshapeReshape!discrim_dense0/Relu:activations:0maxout_1/Reshape/shape:output:0*
T0*"
_output_shapes
:2`
maxout_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
maxout_1/MaxMaxmaxout_1/Reshape:output:0'maxout_1/Max/reduction_indices:output:0*
T0*
_output_shapes

:2[
IdentityIdentitymaxout_1/Max:output:0^NoOp*
T0*
_output_shapes

:2
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:
: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:E A

_output_shapes

:


_user_specified_nameinput
¼
`
D__inference_flatten_layer_call_and_return_conditional_losses_6256412

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿ\
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
¦
:__inference_Conditional_dense_block0_layer_call_fn_6256465	
input
unknown:
2
	unknown_0:2
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6256156f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
Ç

__inference_call_322500	
input?
-discrim_dense0_matmul_readvariableop_resource:
2<
.discrim_dense0_biasadd_readvariableop_resource:2
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOpZ
discrim_dense0/CastCastinput*

DstT0*

SrcT0*
_output_shapes

:

$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource*
_output_shapes

:
2*
dtype0
discrim_dense0/MatMulMatMuldiscrim_dense0/Cast:y:0,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2e
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes

:2k
maxout_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      2   
maxout_1/ReshapeReshape!discrim_dense0/Relu:activations:0maxout_1/Reshape/shape:output:0*
T0*"
_output_shapes
:2`
maxout_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
maxout_1/MaxMaxmaxout_1/Reshape:output:0'maxout_1/Max/reduction_indices:output:0*
T0*
_output_shapes

:2[
IdentityIdentitymaxout_1/Max:output:0^NoOp*
T0*
_output_shapes

:2
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:
: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:E A

_output_shapes

:


_user_specified_nameinput
Ø

__inference_call_322484	
input?
-discrim_dense0_matmul_readvariableop_resource:
2<
.discrim_dense0_biasadd_readvariableop_resource:2
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource*
_output_shapes

:
2*
dtype0}
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2e
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes

:2k
maxout_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      2   
maxout_1/ReshapeReshape!discrim_dense0/Relu:activations:0maxout_1/Reshape/shape:output:0*
T0*"
_output_shapes
:2`
maxout_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
maxout_1/MaxMaxmaxout_1/Reshape:output:0'maxout_1/Max/reduction_indices:output:0*
T0*
_output_shapes

:2[
IdentityIdentitymaxout_1/Max:output:0^NoOp*
T0*
_output_shapes

:2
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:
: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:E A

_output_shapes

:


_user_specified_nameinput
¾

à
__inference_call_321906	
input7
$dense_matmul_readvariableop_resource:	ð3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ð*
dtype0k
dense/MatMulMatMulinput#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*
_output_shapes

:W
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	ð: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:F B

_output_shapes
:	ð

_user_specified_nameinput
õ

 __inference__traced_save_6256552
file_prefix4
0savev2_discrim_dense0_kernel_read_readvariableop2
.savev2_discrim_dense0_bias_read_readvariableop6
2savev2_discrim_dense0_kernel_1_read_readvariableop4
0savev2_discrim_dense0_bias_1_read_readvariableop6
2savev2_discrim_dense0_kernel_2_read_readvariableop4
0savev2_discrim_dense0_bias_2_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ê
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ó
valueéBæ	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B ¶
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_discrim_dense0_kernel_read_readvariableop.savev2_discrim_dense0_bias_read_readvariableop2savev2_discrim_dense0_kernel_1_read_readvariableop0savev2_discrim_dense0_bias_1_read_readvariableop2savev2_discrim_dense0_kernel_2_read_readvariableop0savev2_discrim_dense0_bias_2_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*^
_input_shapesM
K: :
ð:ð:
2:2:
¢ð:ð:	ð:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
ð:!

_output_shapes	
:ð:$ 

_output_shapes

:
2: 

_output_shapes
:2:&"
 
_output_shapes
:
¢ð:!

_output_shapes	
:ð:%!

_output_shapes
:	ð: 

_output_shapes
::	

_output_shapes
: 
å
×
__inference_call_321913

inputs
inputs_1-
image_dense_block0_321849:
ð(
image_dense_block0_321851:	ð1
conditional_dense_block0_321869:
2-
conditional_dense_block0_321871:2,
body_dense_block0_321891:
¢ð'
body_dense_block0_321893:	ð'
sigmoid_layer_321907:	ð"
sigmoid_layer_321909:
identity¢)Body_dense_block0/StatefulPartitionedCall¢0Conditional_dense_block0/StatefulPartitionedCall¢*Image_dense_block0/StatefulPartitionedCall¢%Sigmoid_Layer/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : q
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿy
flatten/ReshapeReshapeExpandDims:output:0flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿî
*Image_dense_block0/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0image_dense_block0_321849image_dense_block0_321851*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321848ü
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0conditional_dense_block0_321869conditional_dense_block0_321871*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321868Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ã
concatenate/concatConcatV23Image_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	¢î
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_321891body_dense_block0_321893*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321890ø
%Sigmoid_Layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0sigmoid_layer_321907sigmoid_layer_321909*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321906t
IdentityIdentity.Sigmoid_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall+^Image_dense_block0/StatefulPartitionedCall&^Sigmoid_Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2X
*Image_dense_block0/StatefulPartitionedCall*Image_dense_block0/StatefulPartitionedCall2N
%Sigmoid_Layer/StatefulPartitionedCall%Sigmoid_Layer/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ

__inference_call_322554	
inputA
-discrim_dense0_matmul_readvariableop_resource:
¢ð=
.discrim_dense0_biasadd_readvariableop_resource:	ð
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource* 
_output_shapes
:
¢ð*
dtype0~
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ð
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ðf
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	ðk
maxout_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ð   
maxout_2/ReshapeReshape!discrim_dense0/Relu:activations:0maxout_2/Reshape/shape:output:0*
T0*#
_output_shapes
:ð`
maxout_2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
maxout_2/MaxMaxmaxout_2/Reshape:output:0'maxout_2/Max/reduction_indices:output:0*
T0*
_output_shapes
:	ð\
IdentityIdentitymaxout_2/Max:output:0^NoOp*
T0*
_output_shapes
:	ð
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	¢: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:F B

_output_shapes
:	¢

_user_specified_nameinput
æ

__inference_call_321890	
inputA
-discrim_dense0_matmul_readvariableop_resource:
¢ð=
.discrim_dense0_biasadd_readvariableop_resource:	ð
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource* 
_output_shapes
:
¢ð*
dtype0~
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ð
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ðf
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	ðk
maxout_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ð   
maxout_2/ReshapeReshape!discrim_dense0/Relu:activations:0maxout_2/Reshape/shape:output:0*
T0*#
_output_shapes
:ð`
maxout_2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
maxout_2/MaxMaxmaxout_2/Reshape:output:0'maxout_2/Max/reduction_indices:output:0*
T0*
_output_shapes
:	ð\
IdentityIdentitymaxout_2/Max:output:0^NoOp*
T0*
_output_shapes
:	ð
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	¢: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:F B

_output_shapes
:	¢

_user_specified_nameinput

½
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6256179	
inputA
-discrim_dense0_matmul_readvariableop_resource:
¢ð=
.discrim_dense0_biasadd_readvariableop_resource:	ð
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource* 
_output_shapes
:
¢ð*
dtype0~
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ð
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ðf
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	ðk
maxout_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ð   
maxout_2/ReshapeReshape!discrim_dense0/Relu:activations:0maxout_2/Reshape/shape:output:0*
T0*#
_output_shapes
:ð`
maxout_2/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
maxout_2/MaxMaxmaxout_2/Reshape:output:0'maxout_2/Max/reduction_indices:output:0*
T0*
_output_shapes
:	ð\
IdentityIdentitymaxout_2/Max:output:0^NoOp*
T0*
_output_shapes
:	ð
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	¢: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:F B

_output_shapes
:	¢

_user_specified_nameinput
#

J__inference_discriminator_layer_call_and_return_conditional_losses_6256322
input_1
input_2.
image_dense_block0_6256299:
ð)
image_dense_block0_6256301:	ð2
 conditional_dense_block0_6256304:
2.
 conditional_dense_block0_6256306:2-
body_dense_block0_6256311:
¢ð(
body_dense_block0_6256313:	ð(
sigmoid_layer_6256316:	ð#
sigmoid_layer_6256318:
identity¢)Body_dense_block0/StatefulPartitionedCall¢0Conditional_dense_block0/StatefulPartitionedCall¢*Image_dense_block0/StatefulPartitionedCall¢%Sigmoid_Layer/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : p

ExpandDims
ExpandDimsinput_1ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : p
ExpandDims_1
ExpandDimsinput_2ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
flatten/PartitionedCallPartitionedCallExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_6256118°
*Image_dense_block0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0image_dense_block0_6256299image_dense_block0_6256301*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_Image_dense_block0_layer_call_and_return_conditional_losses_6256135¼
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0 conditional_dense_block0_6256304 conditional_dense_block0_6256306*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6256156Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ã
concatenate/concatConcatV23Image_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	¢§
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_6256311body_dense_block0_6256313*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6256179­
%Sigmoid_Layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0sigmoid_layer_6256316sigmoid_layer_6256318*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Sigmoid_Layer_layer_call_and_return_conditional_losses_6256196t
IdentityIdentity.Sigmoid_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall+^Image_dense_block0/StatefulPartitionedCall&^Sigmoid_Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2X
*Image_dense_block0/StatefulPartitionedCall*Image_dense_block0/StatefulPartitionedCall2N
%Sigmoid_Layer/StatefulPartitionedCall%Sigmoid_Layer/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:LH
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
Ë
£
4__inference_Image_dense_block0_layer_call_fn_6256441	
input
unknown:
ð
	unknown_0:	ð
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_Image_dense_block0_layer_call_and_return_conditional_losses_6256135g
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:	ð`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
ñ


J__inference_Sigmoid_Layer_layer_call_and_return_conditional_losses_6256196	
input7
$dense_matmul_readvariableop_resource:	ð3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ð*
dtype0k
dense/MatMulMatMulinput#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:Y
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*
_output_shapes

:W
IdentityIdentitydense/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	ð: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:F B

_output_shapes
:	ð

_user_specified_nameinput
° 

J__inference_discriminator_layer_call_and_return_conditional_losses_6256377
inputs_0
inputs_1.
image_dense_block0_6256354:
ð)
image_dense_block0_6256356:	ð2
 conditional_dense_block0_6256359:
2.
 conditional_dense_block0_6256361:2-
body_dense_block0_6256366:
¢ð(
body_dense_block0_6256368:	ð(
sigmoid_layer_6256371:	ð#
sigmoid_layer_6256373:
identity¢)Body_dense_block0/StatefulPartitionedCall¢0Conditional_dense_block0/StatefulPartitionedCall¢*Image_dense_block0/StatefulPartitionedCall¢%Sigmoid_Layer/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : q
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"   ÿÿÿÿy
flatten/ReshapeReshapeExpandDims:output:0flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
*Image_dense_block0/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0image_dense_block0_6256354image_dense_block0_6256356*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321848þ
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0 conditional_dense_block0_6256359 conditional_dense_block0_6256361*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321868Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ã
concatenate/concatConcatV23Image_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	¢ð
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_6256366body_dense_block0_6256368*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321890ú
%Sigmoid_Layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0sigmoid_layer_6256371sigmoid_layer_6256373*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321906t
IdentityIdentity.Sigmoid_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall+^Image_dense_block0/StatefulPartitionedCall&^Sigmoid_Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2X
*Image_dense_block0/StatefulPartitionedCall*Image_dense_block0/StatefulPartitionedCall2N
%Sigmoid_Layer/StatefulPartitionedCall%Sigmoid_Layer/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ê

__inference_call_322515	
input?
-discrim_dense0_matmul_readvariableop_resource:
2<
.discrim_dense0_biasadd_readvariableop_resource:2
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource*
_output_shapes

:
2*
dtype0}
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2e
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes

:2k
maxout_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      2   
maxout_1/ReshapeReshape!discrim_dense0/Relu:activations:0maxout_1/Reshape/shape:output:0*
T0*"
_output_shapes
:2`
maxout_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
maxout_1/MaxMaxmaxout_1/Reshape:output:0'maxout_1/Max/reduction_indices:output:0*
T0*
_output_shapes

:2[
IdentityIdentitymaxout_1/Max:output:0^NoOp*
T0*
_output_shapes

:2
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput

¶
"__inference__wrapped_model_6256099
input_1
input_2)
discriminator_6256081:
ð$
discriminator_6256083:	ð'
discriminator_6256085:
2#
discriminator_6256087:2)
discriminator_6256089:
¢ð$
discriminator_6256091:	ð(
discriminator_6256093:	ð#
discriminator_6256095:
identity¢%discriminator/StatefulPartitionedCallï
%discriminator/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2discriminator_6256081discriminator_6256083discriminator_6256085discriminator_6256087discriminator_6256089discriminator_6256091discriminator_6256093discriminator_6256095*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321913t
IdentityIdentity.discriminator/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:n
NoOpNoOp&^discriminator/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2N
%discriminator/StatefulPartitionedCall%discriminator/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:LH
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
¨
Á
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6256480	
input?
-discrim_dense0_matmul_readvariableop_resource:
2<
.discrim_dense0_biasadd_readvariableop_resource:2
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource*
_output_shapes

:
2*
dtype0}
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2e
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes

:2k
maxout_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      2   
maxout_1/ReshapeReshape!discrim_dense0/Relu:activations:0maxout_1/Reshape/shape:output:0*
T0*"
_output_shapes
:2`
maxout_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
maxout_1/MaxMaxmaxout_1/Reshape:output:0'maxout_1/Max/reduction_indices:output:0*
T0*
_output_shapes

:2[
IdentityIdentitymaxout_1/Max:output:0^NoOp*
T0*
_output_shapes

:2
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
¯
Ù
__inference_call_322292
inputs_0
inputs_1-
image_dense_block0_322253:
ð(
image_dense_block0_322255:	ð1
conditional_dense_block0_322274:
2-
conditional_dense_block0_322276:2,
body_dense_block0_322281:
¢ð'
body_dense_block0_322283:	ð'
sigmoid_layer_322286:	ð"
sigmoid_layer_322288:
identity¢)Body_dense_block0/StatefulPartitionedCall¢0Conditional_dense_block0/StatefulPartitionedCall¢*Image_dense_block0/StatefulPartitionedCall¢%Sigmoid_Layer/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : h

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*"
_output_shapes
:R
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
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  q
flatten/ReshapeReshapeExpandDims:output:0flatten/Const:output:0*
T0*
_output_shapes
:	î
*Image_dense_block0/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0image_dense_block0_322253image_dense_block0_322255*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321848ü
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0conditional_dense_block0_322274conditional_dense_block0_322276*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_322273Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ã
concatenate/concatConcatV23Image_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	¢î
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_322281body_dense_block0_322283*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321890ø
%Sigmoid_Layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0sigmoid_layer_322286sigmoid_layer_322288*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 * 
fR
__inference_call_321906t
IdentityIdentity.Sigmoid_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall+^Image_dense_block0/StatefulPartitionedCall&^Sigmoid_Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 ::
: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2X
*Image_dense_block0/StatefulPartitionedCall*Image_dense_block0/StatefulPartitionedCall2N
%Sigmoid_Layer/StatefulPartitionedCall%Sigmoid_Layer/StatefulPartitionedCall:H D

_output_shapes

:
"
_user_specified_name
inputs/0:D@

_output_shapes
:

"
_user_specified_name
inputs/1

¾
O__inference_Image_dense_block0_layer_call_and_return_conditional_losses_6256135	
inputA
-discrim_dense0_matmul_readvariableop_resource:
ð=
.discrim_dense0_biasadd_readvariableop_resource:	ð
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource* 
_output_shapes
:
ð*
dtype0~
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ð
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ðf
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	ði
maxout/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ð   
maxout/ReshapeReshape!discrim_dense0/Relu:activations:0maxout/Reshape/shape:output:0*
T0*#
_output_shapes
:ð^
maxout/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :{

maxout/MaxMaxmaxout/Reshape:output:0%maxout/Max/reduction_indices:output:0*
T0*
_output_shapes
:	ðZ
IdentityIdentitymaxout/Max:output:0^NoOp*
T0*
_output_shapes
:	ð
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
¨
E
)__inference_flatten_layer_call_fn_6256406

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_6256118`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê

__inference_call_321868	
input?
-discrim_dense0_matmul_readvariableop_resource:
2<
.discrim_dense0_biasadd_readvariableop_resource:2
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource*
_output_shapes

:
2*
dtype0}
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2e
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes

:2k
maxout_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      2   
maxout_1/ReshapeReshape!discrim_dense0/Relu:activations:0maxout_1/Reshape/shape:output:0*
T0*"
_output_shapes
:2`
maxout_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
maxout_1/MaxMaxmaxout_1/Reshape:output:0'maxout_1/Max/reduction_indices:output:0*
T0*
_output_shapes

:2[
IdentityIdentitymaxout_1/Max:output:0^NoOp*
T0*
_output_shapes

:2
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
ÿ"

J__inference_discriminator_layer_call_and_return_conditional_losses_6256203

inputs
inputs_1.
image_dense_block0_6256136:
ð)
image_dense_block0_6256138:	ð2
 conditional_dense_block0_6256157:
2.
 conditional_dense_block0_6256159:2-
body_dense_block0_6256180:
¢ð(
body_dense_block0_6256182:	ð(
sigmoid_layer_6256197:	ð#
sigmoid_layer_6256199:
identity¢)Body_dense_block0/StatefulPartitionedCall¢0Conditional_dense_block0/StatefulPartitionedCall¢*Image_dense_block0/StatefulPartitionedCall¢%Sigmoid_Layer/StatefulPartitionedCallP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : q
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
flatten/PartitionedCallPartitionedCallExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_6256118°
*Image_dense_block0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0image_dense_block0_6256136image_dense_block0_6256138*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_Image_dense_block0_layer_call_and_return_conditional_losses_6256135¼
0Conditional_dense_block0/StatefulPartitionedCallStatefulPartitionedCallExpandDims_1:output:0 conditional_dense_block0_6256157 conditional_dense_block0_6256159*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *^
fYRW
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6256156Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ã
concatenate/concatConcatV23Image_dense_block0/StatefulPartitionedCall:output:09Conditional_dense_block0/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*
_output_shapes
:	¢§
)Body_dense_block0/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0body_dense_block0_6256180body_dense_block0_6256182*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ð*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6256179­
%Sigmoid_Layer/StatefulPartitionedCallStatefulPartitionedCall2Body_dense_block0/StatefulPartitionedCall:output:0sigmoid_layer_6256197sigmoid_layer_6256199*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_Sigmoid_Layer_layer_call_and_return_conditional_losses_6256196t
IdentityIdentity.Sigmoid_Layer/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:ú
NoOpNoOp*^Body_dense_block0/StatefulPartitionedCall1^Conditional_dense_block0/StatefulPartitionedCall+^Image_dense_block0/StatefulPartitionedCall&^Sigmoid_Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2V
)Body_dense_block0/StatefulPartitionedCall)Body_dense_block0/StatefulPartitionedCall2d
0Conditional_dense_block0/StatefulPartitionedCall0Conditional_dense_block0/StatefulPartitionedCall2X
*Image_dense_block0/StatefulPartitionedCall*Image_dense_block0/StatefulPartitionedCall2N
%Sigmoid_Layer/StatefulPartitionedCall%Sigmoid_Layer/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å

__inference_call_321848	
inputA
-discrim_dense0_matmul_readvariableop_resource:
ð=
.discrim_dense0_biasadd_readvariableop_resource:	ð
identity¢%discrim_dense0/BiasAdd/ReadVariableOp¢$discrim_dense0/MatMul/ReadVariableOp
$discrim_dense0/MatMul/ReadVariableOpReadVariableOp-discrim_dense0_matmul_readvariableop_resource* 
_output_shapes
:
ð*
dtype0~
discrim_dense0/MatMulMatMulinput,discrim_dense0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ð
%discrim_dense0/BiasAdd/ReadVariableOpReadVariableOp.discrim_dense0_biasadd_readvariableop_resource*
_output_shapes	
:ð*
dtype0
discrim_dense0/BiasAddBiasAdddiscrim_dense0/MatMul:product:0-discrim_dense0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	ðf
discrim_dense0/ReluReludiscrim_dense0/BiasAdd:output:0*
T0*
_output_shapes
:	ði
maxout/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      ð   
maxout/ReshapeReshape!discrim_dense0/Relu:activations:0maxout/Reshape/shape:output:0*
T0*#
_output_shapes
:ð^
maxout/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :{

maxout/MaxMaxmaxout/Reshape:output:0%maxout/Max/reduction_indices:output:0*
T0*
_output_shapes
:	ðZ
IdentityIdentitymaxout/Max:output:0^NoOp*
T0*
_output_shapes
:	ð
NoOpNoOp&^discrim_dense0/BiasAdd/ReadVariableOp%^discrim_dense0/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2N
%discrim_dense0/BiasAdd/ReadVariableOp%discrim_dense0/BiasAdd/ReadVariableOp2L
$discrim_dense0/MatMul/ReadVariableOp$discrim_dense0/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
¿

Õ
/__inference_discriminator_layer_call_fn_6256344
inputs_0
inputs_1
unknown:
ð
	unknown_0:	ð
	unknown_1:
2
	unknown_2:2
	unknown_3:
¢ð
	unknown_4:	ð
	unknown_5:	ð
	unknown_6:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_discriminator_layer_call_and_return_conditional_losses_6256203f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¹

Ó
/__inference_discriminator_layer_call_fn_6256222
input_1
input_2
unknown:
ð
	unknown_0:	ð
	unknown_1:
2
	unknown_2:2
	unknown_3:
¢ð
	unknown_4:	ð
	unknown_5:	ð
	unknown_6:
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_discriminator_layer_call_and_return_conditional_losses_6256203f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:LH
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Û
serving_defaultÇ
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ
7
input_2,
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿ3
output_1'
StatefulPartitionedCall:0tensorflow/serving/predict:©
Ê
dense_img_blocks
dense_cond_blocks
dense_body_blocks
flatten_layer
out_sigmoid_layer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
call

signatures"
_tf_keras_model
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Á
output_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
call"
_tf_keras_layer
X
 0
!1
"2
#3
$4
%5
&6
'7"
trackable_list_wrapper
X
 0
!1
"2
#3
$4
%5
&6
'7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
/__inference_discriminator_layer_call_fn_6256222
/__inference_discriminator_layer_call_fn_6256344¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
J__inference_discriminator_layer_call_and_return_conditional_losses_6256377
J__inference_discriminator_layer_call_and_return_conditional_losses_6256322¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÖBÓ
"__inference__wrapped_model_6256099input_1input_2"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
__inference_call_322243
__inference_call_322292
__inference_call_322325¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
-serving_default"
signature_map
»

.layers
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5call"
_tf_keras_layer
»

6layers
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=call"
_tf_keras_layer
»

>layers
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
Ecall"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_flatten_layer_call_fn_6256406¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_flatten_layer_call_and_return_conditional_losses_6256412¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
»

&kernel
'bias
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
/__inference_Sigmoid_Layer_layer_call_fn_6256421¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
J__inference_Sigmoid_Layer_layer_call_and_return_conditional_losses_6256432¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
__inference_call_322391¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
):'
ð2discrim_dense0/kernel
": ð2discrim_dense0/bias
':%
22discrim_dense0/kernel
!:22discrim_dense0/bias
):'
¢ð2discrim_dense0/kernel
": ð2discrim_dense0/bias
:	ð2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÓBÐ
%__inference_signature_wrapper_6256401input_1input_2"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
V0
W1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
Ý2Ú
4__inference_Image_dense_block0_layer_call_fn_6256441¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
O__inference_Image_dense_block0_layer_call_and_return_conditional_losses_6256456¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ù2Ö
__inference_call_322430
__inference_call_322445¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
]0
^1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
ã2à
:__inference_Conditional_dense_block0_layer_call_fn_6256465¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þ2û
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6256480¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
__inference_call_322484
__inference_call_322500
__inference_call_322515¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
d0
e1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
Ü2Ù
3__inference_Body_dense_block0_layer_call_fn_6256489¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6256504¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
__inference_call_322554¡
²
FullArgSpec
args
jself
jinput
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
»

 kernel
!bias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
½

"kernel
#bias
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Á

$kernel
%bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
.
d0
e1"
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
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
trackable_dict_wrapper
N__inference_Body_dense_block0_layer_call_and_return_conditional_losses_6256504K$%&¢#
¢

input	¢
ª "¢

0	ð
 u
3__inference_Body_dense_block0_layer_call_fn_6256489>$%&¢#
¢

input	¢
ª "	ð«
U__inference_Conditional_dense_block0_layer_call_and_return_conditional_losses_6256480R"#.¢+
$¢!

inputÿÿÿÿÿÿÿÿÿ
ª "¢

02
 
:__inference_Conditional_dense_block0_layer_call_fn_6256465E"#.¢+
$¢!

inputÿÿÿÿÿÿÿÿÿ
ª "2¦
O__inference_Image_dense_block0_layer_call_and_return_conditional_losses_6256456S !.¢+
$¢!

inputÿÿÿÿÿÿÿÿÿ
ª "¢

0	ð
 ~
4__inference_Image_dense_block0_layer_call_fn_6256441F !.¢+
$¢!

inputÿÿÿÿÿÿÿÿÿ
ª "	ð
J__inference_Sigmoid_Layer_layer_call_and_return_conditional_losses_6256432J&'&¢#
¢

input	ð
ª "¢

0
 p
/__inference_Sigmoid_Layer_layer_call_fn_6256421=&'&¢#
¢

input	ð
ª "³
"__inference__wrapped_model_6256099 !"#$%&'T¢Q
J¢G
EB
!
input_1ÿÿÿÿÿÿÿÿÿ

input_2ÿÿÿÿÿÿÿÿÿ
ª "*ª'
%
output_1
output_1|
__inference_call_322243a !"#$%&'D¢A
:¢7
52

inputs/0

inputs/1

ª "|
__inference_call_322292a !"#$%&'D¢A
:¢7
52

inputs/0

inputs/1

ª "
__inference_call_322325s !"#$%&'V¢S
L¢I
GD
"
inputs/0ÿÿÿÿÿÿÿÿÿ

inputs/1ÿÿÿÿÿÿÿÿÿ
ª "X
__inference_call_322391=&'&¢#
¢

input	ð
ª "Y
__inference_call_322430> !&¢#
¢

input	
ª "	ða
__inference_call_322445F !.¢+
$¢!

inputÿÿÿÿÿÿÿÿÿ
ª "	ðW
__inference_call_322484<"#%¢"
¢

input

ª "2W
__inference_call_322500<"#%¢"
¢

input

ª "2`
__inference_call_322515E"#.¢+
$¢!

inputÿÿÿÿÿÿÿÿÿ
ª "2Y
__inference_call_322554>$%&¢#
¢

input	¢
ª "	ðÌ
J__inference_discriminator_layer_call_and_return_conditional_losses_6256322~ !"#$%&'T¢Q
J¢G
EB
!
input_1ÿÿÿÿÿÿÿÿÿ

input_2ÿÿÿÿÿÿÿÿÿ
ª "¢

0
 Ï
J__inference_discriminator_layer_call_and_return_conditional_losses_6256377 !"#$%&'V¢S
L¢I
GD
"
inputs/0ÿÿÿÿÿÿÿÿÿ

inputs/1ÿÿÿÿÿÿÿÿÿ
ª "¢

0
 ¤
/__inference_discriminator_layer_call_fn_6256222q !"#$%&'T¢Q
J¢G
EB
!
input_1ÿÿÿÿÿÿÿÿÿ

input_2ÿÿÿÿÿÿÿÿÿ
ª "¦
/__inference_discriminator_layer_call_fn_6256344s !"#$%&'V¢S
L¢I
GD
"
inputs/0ÿÿÿÿÿÿÿÿÿ

inputs/1ÿÿÿÿÿÿÿÿÿ
ª "¤
D__inference_flatten_layer_call_and_return_conditional_losses_6256412\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_flatten_layer_call_fn_6256406O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÇ
%__inference_signature_wrapper_6256401 !"#$%&'e¢b
¢ 
[ªX
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ
(
input_2
input_2ÿÿÿÿÿÿÿÿÿ"*ª'
%
output_1
output_1