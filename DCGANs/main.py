import tensorflow as tf
from generator_model import generator
from discriminator_model import discriminator
from matplotlib import pyplot as plt
import logging
from training_toolkit import train_tools
import tensorflow_datasets.public_api as tfds

tf.config.run_functions_eagerly(False)

data = input("What data you want to train on? (LSUN, Imagenet-1k): ")
if data == "LSUN":
    #TODO: Load LSUN Dataset
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
    logging.error(f"Not yet implemented for {data}. This should be either reported or feel free to make a pull request.")

fig = plt.figure(figsize=(10, 7))

do_you = input("Do you want to visualize data?: (yes, no): ")

if(do_you == "yes"):
    rows_in = input("Number of rows you want: ")
    columns_in = input("Number of columns you want: ")
    rows = int(rows_in)
    columns = int(columns_in)
    for i in range(rows * columns):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(x_train[i], cmap='gray')
        plt.axis('off')

    plt.show()


# Training
generator = generator.Generator()
discriminator = discriminator.Discriminator(disc_img_struc=[([240], "relu", 1)], disc_cond_struc=[([50], "relu", 1)], disc_body_struc=[([240], "relu", 1)], output_activation="sigmoid", output_shape=(1,))

noise_input_shape = 100
k = 1
iterations=50000
m = 100

# Normalize pixels
x_train_normalized = [tf.keras.utils.normalize(data_sample) for data_sample in x_train]

y_train_hot_encode = tf.one_hot(indices=y_train, depth=10)
train_ds = tf.convert_to_tensor([data_sample for data_sample in x_train_normalized])
train_labels_ds = tf.convert_to_tensor([y for y in y_train_hot_encode])

indices = tf.range(start=0, limit=tf.shape(train_ds)[0], dtype=tf.int32)


#generator.build(input_shape=[noise_input_shape, len(y_train_hot_encode[1])])
#discriminator.build(input_shape=[(uniform,28), len(y_train_hot_encode[0])])

# Get train tools
gen_learn = train_tools.Generator_Learn(generator_model=generator, discriminator_model=discriminator, learning_rate=4e-4, epsilon=1e-8, beta_1=0.0, beta_2=0.9)
disc_learn = train_tools.Discriminator_Learn(generator_model=generator, discriminator_model=discriminator, learning_rate=8e-4, epsilon=1e-8, beta_1=0.0, beta_2=0.9)

gen_loss_fn = train_tools.Generator_Loss()
disc_loss_fn = train_tools.Discriminator_Loss()

data_sample = train_ds[0]
noise_sample = tf.random.uniform(shape=[noise_input_shape], minval=0., maxval=1.0)

gen_rows = 10
gen_columns = 10
test_noise_data = [tf.random.uniform(shape=[noise_input_shape], minval=0.0, maxval=1.0) for _ in range(gen_rows * gen_columns)]
############# NOTE NOTE NOTE NOTE NOTE: THE FOLLOWING STATEMENT WAS IMPORTANT FOR GRADIENTS TO BE NO LONGER NONE ##############################
print("Before discriminator: ", discriminator([train_ds[0], train_labels_ds[0]]))
print(f"\nGAN Training has started...\nDataSet: mnist\nIterations:{iterations} | k: {k} | Batch_size: {m}")
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
    noise_batch = [tf.random.uniform(shape=[noise_input_shape], minval=0.0, maxval=1.0) for _ in range(m)]
    gen_learn.learn(noise_batch=noise_batch, gen_cond_batch=gen_cond_batch, discr_cond_batch=discr_cond_batch)
    fig_generated = plt.figure(figsize=(10,7))
    for i in range(gen_rows * gen_columns):
        fig_generated.add_subplot(gen_rows, gen_columns, i + 1)
        plt.imshow(generator([test_noise_data[i], tf.one_hot(indices=i%10, depth=10)]), cmap="gray")
        plt.xticks([])
        plt.yticks([])
    fig_generated.savefig(f"generated_images/mnist_{iter}.png")

    fake_generated_sample = generator(inputs=[test_noise_data[0], tf.convert_to_tensor([1,0,0,0,0,0,0,0,0,0])])
    print("Discriminator Prediction on generator sample: ", discriminator([fake_generated_sample, tf.convert_to_tensor([1,0,0,0,0,0,0,0,0,0])]))
    print("Discriminator Prediction on random fake noise samples: ", discriminator([tf.random.normal(shape=[28,28], mean=0.0, stddev=1.0), tf.convert_to_tensor([1,0,0,0,0,0,0,0,0,0])]))
    print("DIscriminator Prediction on real sample with correct condition: ", discriminator([train_ds[0], tf.convert_to_tensor([0,0,0,0,0,1,0,0,0,0])]))
    print("DIscriminator Prediction on real sample with incorrect condition: ", discriminator([train_ds[0], tf.convert_to_tensor([1,0,0,0,0,0,0,0,0,0])]))
    # Save Models after every 10 iterations
    if iter % 10 == 0:
        generator.save("saved_models/generator_trained", save_format="tf")
        discriminator.save("saved_models/discriminator_trained", save_format="tf")
    
    plt.close(fig_generated) # ALWAYS CLOSE


# generator = tf.keras.models.load_model('saved_models/generator_trained')
# discriminator = tf.keras.models.load_model('saved_models/discriminator_trained')

