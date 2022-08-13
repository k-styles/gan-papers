import tensorflow as tf
from generator_model import generator
from discriminator_model import discriminator
from matplotlib import pyplot as plt
import logging

#tf.compat.v1.disable_eager_execution()

data = input("What data you want to train on? (mnist, tfd): ")
if data == "mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
    logging.error(f"Not yet implemented for {data}. This should be either reported or can be contributed.")

fig = plt.figure(figsize=(10, 7))

do_you = input("Do you want to visualize data?: (yes, no)")

if(do_you == "yes"):
    rows_in = input("Number of rows you want: ")
    columns_in = input("Number of columns you want: ")
    rows = int(rows_in)
    columns = int(columns_in)
    for i in range(rows * columns):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(x_train[i])
        plt.axis('off')

    plt.show()


# # Random Input Tensor to generator
# tf.random.set_seed(5)
# random_tensor = tf.random.normal(shape=[10], mean=0.0, stddev=1.0)
# # Build generator
# generator = generator.Generator()
# generator.build(input_shape=random_tensor.shape)
# print(generator.summary())

# # Random Input Tensor to discriminator
# tf.random.set_seed(5)
# random_tensor_image = tf.random.normal(shape=[28,28], mean=0.0, stddev=1.0)
# # Build discriminator
# discriminator = discriminator.Discriminator()
# discriminator.build(input_shape=random_tensor_image.shape)
# print(discriminator.summary())

# inputs = tf.keras.Input(tensor=random_tensor)
# gen_outputs = generator(inputs)
# outputs = discriminator(gen_outputs)

# GAN = tf.keras.Model(inputs=inputs, outputs=outputs)
# print(GAN.summary())

# Training
generator = generator.Generator()
discriminator = discriminator.Discriminator()

#GAN = tf.keras.Model(inputs=input, outputs=outputs)

noise_input_shape = 10
k = 1
iterations=2
m = 32
train_ds = [data_sample for data_sample in x_train]#tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(m)
for iter in range(iterations):
    for discriminator_step in range(k):
        # m training samples have already been sampled
        noise_ds = [tf.random.normal(shape=[noise_input_shape], mean=0.0, stddev=1.0) for _ in range(m)]
        
        for noise_sample, train_sample in zip(noise_ds, train_ds):
            discriminator.learn(noise_sample=noise_sample, data_sample=train_sample, generator_model=generator)
        
       noise_ds = [tf.random.normal(shape=[noise_input_shape], mean=0.0, stddev=1.0) for _ in range(m)]
        for noise_sample, train_sample in zip(noise_ds, train_ds):
            generator.learn(noise_sample=noise_sample, discriminator_model=discriminator)