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
generator = generator.Generator(learning_rate=1e-4, epsilon=1e-8, beta_1=0.0, beta_2=0.9, gen_struc=[([1200,1200], "relu")], output_activation="sigmoid")
discriminator = discriminator.Discriminator(learning_rate=4e-4, epsilon=1e-8, beta_1=0.0, beta_2=0.9, gen_struc=[([240,240], "relu")])
#generator = tf.keras.models.load_model("saved_models/generator_trained")
#discriminator = tf.keras.models.load_model("saved_models/discriminator_trained")
#GAN = tf.keras.Model(inputs=input, outputs=outputs)

noise_input_shape = 100
k = 1
iterations=50000
m = 100

generator.build(input_shape=[noise_input_shape])
discriminator.build(input_shape=(28,28))

# Adding some noise to the training data
train_ds = [data_sample for data_sample in x_train]# + 10 * tf.random.normal(shape=[data_sample.shape[0], data_sample.shape[1]], mean=0.0, stddev=1.0) for data_sample in x_train]#tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(m)

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

gen_rows = 10
gen_columns = 10

lr_decay_factor = 1.000004

print(f"\nGAN Training has started...\nDataSet: {data}\nIterations:{iterations} | k: {k} | Batch_size: {m}")
for iter in range(iterations):
    print(f"[Epoch {iter + 1}]: generator_lr: {generator.gen_learning_rate} | discriminator_lr: {discriminator.discr_learning_rate}")
    for discriminator_step in range(k):
        # m training samples have already been sampled
        print(f"\tDiscriminator Epoch {discriminator_step + 1}:")
        noise_ds = [tf.random.uniform(shape=[noise_input_shape], minval=0.0, maxval=1.0) for _ in range(m)]
        train_data = train_ds
        random.shuffle(train_data)
        train_data = train_data[:m]
        disc_learn.learn(noise_batch=noise_ds, data_batch=train_data)
    
    noise_ds = [tf.random.uniform(shape=[noise_input_shape], minval=0.0, maxval=1.0) for _ in range(m)]
    gen_learn.learn(noise_batch=noise_ds)
    # for noise_sample, train_sample in zip(noise_ds, train_ds):
    #     gen_learn.learn(noise_sample=noise_sample)
    # #print("Generator Loss:", gen_learn.loss)
    # #tf.print(gen_learn.loss)
    fig_generated = plt.figure(figsize=(10,7))
    test_noise_data = [tf.random.uniform(shape=[noise_input_shape], minval=0.0, maxval=1.0) for _ in range(gen_rows * gen_columns)]
    for i in range(gen_rows * gen_columns):
        fig_generated.add_subplot(gen_rows, gen_columns, i + 1)
        plt.imshow(generator(test_noise_data[i]), cmap="gray")
        plt.xticks([])
        plt.yticks([])
    fig_generated.savefig(f"generated_images/mnist_{iter}.png")
    
    test_fake = generator(test_noise_data[0])
    print(test_fake)
    print("Discriminator Prediction on generator sample: ", discriminator(test_fake))
    print("Discriminator Prediction on random fake noise samples: ", discriminator(tf.random.uniform(shape=[28,28], minval=0.0, maxval=1.0)))
    print("DIscriminator Prediction on real sample: ", discriminator(train_data[0]))

    # Decay learning rate
    if discriminator.discr_learning_rate >= 0.00000004:
        discriminator.discr_learning_rate = discriminator.discr_learning_rate / lr_decay_factor
    if generator.gen_learning_rate >= 0.0000001:
        generator.gen_learning_rate = generator.gen_learning_rate / lr_decay_factor


    # Save Models after every 10 iterations
    if iter % 10 == 0:
        generator.save_weights("saved_models/generator_trained", save_format="tf")
        discriminator.save_weights("saved_models/discriminator_trained", save_format="tf")
    print("Hey: ", discriminator(tf.random.uniform(shape=(28,28))))
    
    plt.close(fig_generated) # ALWAYS CLOSE


# generator = tf.keras.models.load_model('saved_models/generator_trained')
# discriminator = tf.keras.models.load_model('saved_models/discriminator_trained')

