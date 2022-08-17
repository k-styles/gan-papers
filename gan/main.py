import tensorflow as tf
from generator_model import generator
from discriminator_model import discriminator
from matplotlib import pyplot as plt
import logging
from training_toolkit import train_tools
#tf.config.run_functions_eagerly(True)

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
generator = generator.Generator(learning_rate=1e-3, epsilon=0.1)
discriminator = discriminator.Discriminator(learning_rate=1e-5, epsilon=0.1)

#GAN = tf.keras.Model(inputs=input, outputs=outputs)

noise_input_shape = 10
k = 1
iterations=1000
m = 64

generator.build(input_shape=[10])
discriminator.build(input_shape=(28,28))

train_ds = [data_sample for data_sample in x_train]#tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(m)

# Get train tools
gen_learn = train_tools.Generator_Learn(generator_model=generator, discriminator_model=discriminator)
disc_learn = train_tools.Discriminator_Learn(generator_model=generator, discriminator_model=discriminator)

# noise_sample = tf.random.normal(shape=[noise_input_shape], mean=0.0, stddev=1.0)
# # print(generator(noise_sample))
# plt.imshow(generator(noise_sample))
# plt.show()
# print(discriminator(generator(noise_sample)))
gen_loss_fn = train_tools.Generator_Loss()
disc_loss_fn = train_tools.Discriminator_Loss()

data_sample = train_ds[0]
noise_sample = tf.random.normal(shape=[noise_input_shape], mean=0.0, stddev=1.0)
print("Disc_Loss: ", disc_loss_fn(data_sample=data_sample, noise_sample=noise_sample, discriminator_model=discriminator, generator_model=generator))
print(f"GAN Training has started.\nDataSet: {data}\nIterations:{iterations} | k: {k} | Batch_size: {k}")
for iter in range(iterations):
    print(f"[Epoch {iter}]: ")
    for discriminator_step in range(k):
        # m training samples have already been sampled
        print(f"\tDiscriminator Epoch {discriminator_step + 1}:")
        noise_ds = [tf.random.normal(shape=[noise_input_shape], mean=0.0, stddev=1.0) for _ in range(m)]
        for noise_sample, train_sample in zip(noise_ds, train_ds):
            #print("\tNoise_Sample:", noise_sample.shape, "Train_sample:", train_sample.shape)
            disc_learn.learn(noise_sample=noise_sample, data_sample=train_sample)
        #print("Disciminator Loss:", disc_learn.loss)
        #tf.print(disc_learn.loss)
        
    noise_ds = [tf.random.normal(shape=[noise_input_shape], mean=0.0, stddev=1.0) for _ in range(m)]
    for noise_sample, train_sample in zip(noise_ds, train_ds):
        gen_learn.learn(noise_sample=noise_sample)
    #print("Generator Loss:", gen_learn.loss)
    #tf.print(gen_learn.loss)

test_noise_sample = tf.random.normal(shape=[noise_input_shape], mean=0.0, stddev=1.0)

plt.imshow(generator(test_noise_sample))
#plt.imshow(train_ds[0])
plt.show()