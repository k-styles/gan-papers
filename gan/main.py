import tensorflow as tf
from generator_model import generator
from discriminator_model import discriminator
from matplotlib import pyplot as plt
import logging
from training_toolkit import train_tools
import random
tf.config.run_functions_eagerly(False)

data = input("What data you want to train on? (mnist, tfd): ")
if data == "mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
    logging.error(f"Not yet implemented for {data}. This should be either reported or can be contributed.")

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



# Training
generator = generator.Generator(learning_rate=2e-5, epsilon=1e-8, gen_struc=[([500,500], "relu")], output_activation="relu")
discriminator = discriminator.Discriminator(learning_rate=2e-5, epsilon=1e-8, gen_struc=[([400,400], "relu")])
#generator = tf.keras.models.load_model("saved_models/generator_trained")
#discriminator = tf.keras.models.load_model("saved_models/discriminator_trained")
#GAN = tf.keras.Model(inputs=input, outputs=outputs)

noise_input_shape = 100
k = 2
iterations=10000
m = 32

generator.build(input_shape=[noise_input_shape])
discriminator.build(input_shape=(28,28))

# Adding some noise to the training data
train_ds = [data_sample + 10 * tf.random.normal(shape=[data_sample.shape[0], data_sample.shape[1]], mean=0.0, stddev=1.0) for data_sample in x_train]#tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(m)

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

gen_rows = 10
gen_columns = 10

print("Disc_Loss: ", disc_loss_fn(data_sample=data_sample, noise_sample=noise_sample, discriminator_model=discriminator, generator_model=generator))
print(f"\nGAN Training has started...\nDataSet: {data}\nIterations:{iterations} | k: {k} | Batch_size: {m}")
for iter in range(iterations):
    print(f"[Epoch {iter + 1}]: ")
    for discriminator_step in range(k):
        # m training samples have already been sampled
        print(f"\tDiscriminator Epoch {discriminator_step + 1}:")
        noise_ds = [tf.random.normal(shape=[noise_input_shape], mean=0.0, stddev=1.0) for _ in range(m)]
        random.shuffle(train_ds)
        train_ds = train_ds[:m]
        disc_learn.learn(noise_batch=noise_ds, data_batch=train_ds)
    
    noise_ds = [tf.random.normal(shape=[noise_input_shape], mean=0.0, stddev=1.0) for _ in range(m)]
    gen_learn.learn(noise_batch=noise_ds)
    # for noise_sample, train_sample in zip(noise_ds, train_ds):
    #     gen_learn.learn(noise_sample=noise_sample)
    # #print("Generator Loss:", gen_learn.loss)
    # #tf.print(gen_learn.loss)
    fig_generated = plt.figure(figsize=(10,7))
    test_noise_data = [tf.random.normal(shape=[noise_input_shape], mean=0.0, stddev=1.0) for _ in range(gen_rows * gen_columns)]
    for i in range(gen_rows * gen_columns):
        fig_generated.add_subplot(gen_rows, gen_columns, i + 1)
        plt.imshow(generator(test_noise_data[i]), cmap="gray")
        plt.xticks([])
        plt.yticks([])
    fig_generated.savefig(f"generated_images/mnist_{iter}.png")
    
    test_fake = generator(test_noise_data[0])
    print(test_fake)
    print("Discriminator Prediction on generator sample: ", discriminator(test_fake))
    print("Discriminator Prediction on random fake noise samples: ", discriminator(tf.random.normal(shape=[28,28], mean=0.0, stddev=1.0)))
    print("DIscriminator Prediction on real sample: ", discriminator(train_ds[0]))

    # Save Models after every 10 iterations
    if iter % 10 == 0:
        generator.save("saved_models/generator_trained", save_format="tf")
        discriminator.save("saved_models/discriminator_trained", save_format="tf")
    
    plt.close(fig_generated) # ALWAYS CLOSE


# generator = tf.keras.models.load_model('saved_models/generator_trained')
# discriminator = tf.keras.models.load_model('saved_models/discriminator_trained')

