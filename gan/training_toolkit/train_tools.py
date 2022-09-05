import tensorflow as tf

class Discriminator_Loss(tf.keras.losses.Loss):
    @tf.function
    def __call__(self, noise_sample, data_sample, discriminator_model, generator_model):
        return (tf.math.log(discriminator_model(data_sample)) + tf.math.log(1 - discriminator_model(generator_model(noise_sample))))

class Generator_Loss(tf.keras.losses.Loss):
    @tf.function
    def __call__(self, noise_sample, discriminator_model, generator_model):
        return (tf.math.log(discriminator_model(generator_model(noise_sample))))

class Discriminator_Cost(tf.keras.losses.Loss): 
    @tf.function
    def __call__(self, noise_batch, data_batch, discriminator_model, generator_model):
        m = len(noise_batch)
        disc_cost = 0
        disc_loss = Discriminator_Loss()
        for noise_sample, data_sample in zip(noise_batch, data_batch):   # Not optimal
            disc_cost += disc_loss(noise_sample=noise_sample, data_sample=data_sample, 
                                   discriminator_model=discriminator_model, generator_model=generator_model)
        return -(1/m) * disc_cost

class Generator_Cost(tf.keras.losses.Loss): 
    @tf.function
    def __call__(self, noise_batch, discriminator_model, generator_model):
        m = len(noise_batch)
        gen_cost = 0
        gen_loss = Generator_Loss()
        for noise_sample in noise_batch:   # Not optimal
            gen_cost += gen_loss(noise_sample=noise_sample, discriminator_model=discriminator_model, 
                                 generator_model=generator_model)
        return -(1/m) * gen_cost

class Discriminator_Learn:
    def __init__(self, generator_model, discriminator_model):
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.batch_loss = None

    @tf.function
    def learn(self, noise_batch, data_batch):
        discr_variables = []
        discr_cost_fn = Discriminator_Cost()
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            # Capture Variables
            # for block in self.discriminator_model.dense_blocks:
            #     for layer in block.layers:
            #         if len(layer.variables)!=0:
            #             tape.watch(layer.variables)
            #             discr_variables.append(layer.variables[0])
            #             discr_variables.append(layer.variables[1])
            # tape.watch(self.discriminator_model.out_sigmoid_layer.variables[0])
            # tape.watch(self.discriminator_model.out_sigmoid_layer.variables[1])
            # discr_variables.append(self.discriminator_model.out_sigmoid_layer.variables[0])
            # discr_variables.append(self.discriminator_model.out_sigmoid_layer.variables[1])
            # self.batch_loss = discr_cost_fn(noise_batch=noise_batch, data_batch=data_batch, 
            #                   generator_model=self.generator_model, discriminator_model=self.discriminator_model)
            tape.watch(self.discriminator_model.trainable_variables)
        grads = tape.gradient(self.batch_loss, self.discriminator_model.trainable_variables)
        tf.print("Discriminator Batch Loss: ", self.batch_loss)
        del tape
        self.discriminator_model.optimizer.apply_gradients(zip(grads, self.discriminator_model.trainable_variables))
        return grads


class Generator_Learn:
    def __init__(self, generator_model, discriminator_model):
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.batch_loss = None

    @tf.function
    def learn(self, noise_batch):
        gen_variables = []
        gen_cost_fn = Generator_Cost()
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
        #     # Capture Variables
        #     for block in self.generator_model.dense_blocks:
        #         for layer in block.layers:
        #             if len(layer.variables)!=0:
        #                 tape.watch(layer.variables)
        #                 gen_variables.append(layer.variables[0])
        #                 gen_variables.append(layer.variables[1])
        #     tape.watch(self.generator_model.output_layer.variables[0])
        #     tape.watch(self.generator_model.output_layer.variables[1])
        #     gen_variables.append(self.generator_model.output_layer.variables[0])
        #     gen_variables.append(self.generator_model.output_layer.variables[1])
        #     self.batch_loss = gen_cost_fn(noise_batch=noise_batch, generator_model=self.generator_model, 
        #                                   discriminator_model=self.discriminator_model)
        # print("Variables: ", gen_variables, " || Number of Variables: ", len(gen_variables))
            tape.watch(self.generator_model.learnable_variables)
        grads = tape.gradient(self.batch_loss, self.generator_model.trainable_variables)
        tf.print("Generator Batch Loss: ", self.batch_loss)
        del tape
        self.generator_model.optimizer.apply_gradients(zip(grads, self.generator_model.trainable_variables))
        return grads