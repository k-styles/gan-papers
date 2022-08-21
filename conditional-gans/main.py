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

# Training
generator = generator.Generator(gen_noise_struc=[([200], "relu")], gen_cond_struc=[([1000], "relu")], gen_body_struc=[([1200], "relu")], output_activation="relu", output_shape=(28,28))
discriminator = discriminator.Discriminator(disc_img_struc=[([240], "relu", 1)], disc_cond_struc=[([50], "relu", 1)], disc_body_struc=[([240], "relu", 1)], output_activation="sigmoid", output_shape=(1,))

noise_input_shape = 100
k = 1
iterations=10000
m = 100

generator.build(input_shape=[noise_input_shape])
discriminator.build(input_shape=(28,28))

train_ds = tf.convert_to_tensor([data_sample for data_sample in x_train])
train_labels_ds = tf.convert_to_tensor([y for y in y_train_hot_encode])

indices = tf.range(start=0, limit=tf.shape(train_ds)[0], dtype=tf.int32)


# Get train tools
gen_learn = train_tools.Generator_Learn(generator_model=generator, discriminator_model=discriminator, learning_rate=1e-5, epsilon=1e-1, name='Adam')
disc_learn = train_tools.Discriminator_Learn(generator_model=generator, discriminator_model=discriminator, learning_rate=1e-5, epsilon=1e-1, name='Adam')

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
        noise_batch = [tf.random.uniform(shape=[noise_input_shape], minval=0.0, maxval=1.0) for _ in range(m)]
        shuffled_batch_indices = tf.random.shuffle(indices)[:m]
        data_batch = tf.gather(train_ds, shuffled_batch_indices)
        gen_cond_batch = tf.gather(train_labels_ds, shuffled_batch_indices)
        discr_cond_batch = gen_cond_batch

        disc_learn.learn(noise_batch=noise_batch, data_batch=data_batch, gen_cond_batch=gen_cond_batch, discr_cond_batch=discr_cond_batch)
    
    # NOTE: SAMPLING NOISE AGAIN    
    noise_batch = [tf.random.normal(shape=[noise_input_shape], minval=0.0, maxval=1.0) for _ in range(m)]
    gen_learn.learn(noise_batch=noise_batch, gen_cond_batch=gen_cond_batch, discr_cond_batch=discr_cond_batch)
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

