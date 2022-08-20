import tensorflow as tf
from generator_model import generator
from discriminator_model import discriminator
from matplotlib import pyplot as plt
import logging
from training_toolkit import train_tools
tf.config.run_functions_eagerly(False)

print("Using mnist dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

fig = plt.figure(figsize=(10, 7))

do_you = input("Do you want to visualize data?: (yes, no): ")

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

y_train_hot_encode = tf.one_hot(indices=y_train, depth=10)
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
generator = generator.Generator(gen_struc=[([250,250], "relu")], output_activation="relu")
discriminator = discriminator.Discriminator(gen_struc=[([100,100], "relu"),([400,400], "relu")])
#generator = tf.keras.models.load_model("saved_models/generator_trained")
#discriminator = tf.keras.models.load_model("saved_models/discriminator_trained")
#GAN = tf.keras.Model(inputs=input, outputs=outputs)

noise_input_shape = 100
k = 1
iterations=10000
m = 100

generator.build(input_shape=[noise_input_shape])
discriminator.build(input_shape=(28,28))

train_ds = [data_sample for data_sample in x_train]#tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(m)
train_labels_ds = [y for y in y_train_hot_encode]

# Get train tools
gen_learn = train_tools.Generator_Learn(generator_model=generator, discriminator_model=discriminator, learning_rate=1e-5, epsilon=1e-1, name='Adam')
disc_learn = train_tools.Discriminator_Learn(generator_model=generator, discriminator_model=discriminator, learning_rate=1e-5, epsilon=1e-1, name='Adam')

# noise_sample = tf.random.normal(shape=[noise_input_shape], mean=0.0, stddev=1.0)
# # print(generator(noise_sample))
# plt.imshow(generator(noise_sample))
# plt.show()        self.optimizer = tf.keras.optimizers.get(identifier=name)
# print(discriminator(generator(noise_sample)))
gen_loss_fn = train_tools.Generator_Loss()
disc_loss_fn = train_tools.Discriminator_Loss()

data_sample = train_ds[0]
noise_sample = tf.random.uniform(shape=[noise_input_shape], minval=0.0, maxval=1.0)

gen_rows = 10
gen_columns = 10
test_noise_data = [tf.random.uniform(shape=[noise_input_shape], minval=0.0, maxval=1.0) for _ in range(gen_rows * gen_columns)]

print("Disc_Loss: ", disc_loss_fn(data_sample=data_sample, noise_sample=noise_sample, discriminator_model=discriminator, generator_model=generator))
print(f"\nGAN Training has started...\nDataSet: {data}\nIterations:{iterations} | k: {k} | Batch_size: {m}")
for iter in range(iterations):
    print(f"[Epoch {iter + 1}]: ")
    for discriminator_step in range(k):
        # m training samples have already been sampled
        print(f"\tDiscriminator Epoch {discriminator_step + 1}:")
        noise_ds = [tf.random.normal(shape=[noise_input_shape], mean=0.0, stddev=1.0) for _ in range(m)]
        disc_learn.learn(noise_batch=noise_ds, data_batch=train_ds)
        # for noise_sample, train_sample in zip(noise_ds, train_ds):
        #     #print("\tNoise_Sample:", noise_sample.shape, "Train_sample:", train_sample.shape)
        #     disc_learn.learn(noise_sample=noise_sample, data_sample=train_sample)
        # #print("Disciminator Loss:", disc_learn.loss)
        # #tf.print(disc_learn.loss)
        
    noise_ds = [tf.random.normal(shape=[noise_input_shape], mean=0.0, stddev=1.0) for _ in range(m)]
    gen_learn.learn(noise_batch=noise_ds)
    # for noise_sample, train_sample in zip(noise_ds, train_ds):
    #     gen_learn.learn(noise_sample=noise_sample)
    # #print("Generator Loss:", gen_learn.loss)
    # #tf.print(gen_learn.loss)
    fig_generated = plt.figure(figsize=(10,7))
    for i in range(gen_rows * gen_columns):
        fig_generated.add_subplot(gen_rows, gen_columns, i + 1)
        plt.imshow(generator(test_noise_data[i]), cmap="gray")
        plt.xticks([])
        plt.yticks([])
    fig_generated.savefig(f"generated_images/mnist_{iter}.png")

    print(generator(test_noise_data[0]))

    # Save Models after every 10 iterations
    if iter % 10 == 0:
        generator.save("saved_models/generator_trained", save_format="tf")
        discriminator.save("saved_models/discriminator_trained", save_format="tf")
    
    plt.close(fig_generated) # ALWAYS CLOSE


# generator = tf.keras.models.load_model('saved_models/generator_trained')
# discriminator = tf.keras.models.load_model('saved_models/discriminator_trained')

