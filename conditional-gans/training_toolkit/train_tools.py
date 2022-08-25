import tensorflow as tf

class Discriminator_Loss(tf.keras.losses.Loss):
    @tf.function
    def __call__(self, noise_sample, data_sample, gen_cond_sample, discr_cond_sample, discriminator_model, generator_model):
        discr_inputs_real = [data_sample, discr_cond_sample]
        gen_inputs = [noise_sample, gen_cond_sample]
        discr_inputs_fake = [generator_model(gen_inputs), discr_cond_sample]

        return  tf.math.log(discriminator_model(inputs=discr_inputs)) + tf.math.log(1 - discriminator_model(inputs=[generator_model(gen_inputs), discr_cond_sample]))

class Generator_Loss(tf.keras.losses.Loss):
    @tf.function
    def __call__(self, noise_sample, gen_cond_sample, discr_cond_sample, discriminator_model, generator_model):
        gen_inputs = [noise_sample, gen_cond_sample]
        discr_inputs_fake = [generator_model(gen_inputs), discr_cond_sample]
        return tf.math.log(discriminator_model(inputs=[generator_model(gen_inputs), discr_cond_sample]))

class Discriminator_Cost(tf.keras.losses.Loss): 
    @tf.function
    def __call__(self, noise_batch, data_batch, gen_cond_batch, discr_cond_batch, discriminator_model, generator_model):
        m = len(noise_batch)
        disc_cost = 0
        disc_loss = Discriminator_Loss()
        # CONVERT TENSOR BATCHES TO LISTS
        noise_batch = tf.unstack(noise_batch)
        data_batch = tf.unstack(data_batch)
        gen_cond_batch = tf.unstack(gen_cond_batch)
        discr_cond_batch = tf.unstack(discr_cond_batch)
        for noise_sample, data_sample, gen_cond_sample, discr_cond_sample in zip(noise_batch, data_batch, gen_cond_batch, discr_cond_batch):   # Not optimal
            disc_cost += disc_loss(noise_sample=noise_sample, data_sample=data_sample, gen_cond_sample=gen_cond_sample, 
                                   discr_cond_sample=discr_cond_sample, discriminator_model=discriminator_model, generator_model=generator_model)
        return -(1/m) * disc_cost

class Generator_Cost(tf.keras.losses.Loss): 
    @tf.function
    def __call__(self, noise_batch, gen_cond_batch, discr_cond_batch, discriminator_model, generator_model):
        m = len(noise_batch)
        gen_cost = 0
        gen_loss = Generator_Loss()
        # CONVERT TENSOR BATCHES TO LISTS
        noise_batch = tf.unstack(noise_batch)
        gen_cond_batch = tf.unstack(gen_cond_batch)
        discr_cond_batch = tf.unstack(discr_cond_batch)
        for noise_sample, gen_cond_sample, discr_cond_sample in zip(noise_batch, gen_cond_batch, discr_cond_batch):   # Not optimal
            gen_cost += gen_loss(noise_sample=noise_sample, gen_cond_sample=gen_cond_sample, 
                                 discr_cond_batch=discr_cond_sample, discriminator_model=discriminator_model, generator_model=generator_model)
        return -(1/m) * gen_cost

class Discriminator_Learn:
    def __init__(self, generator_model, discriminator_model, learning_rate=5e-03, epsilon=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False):
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.batch_loss = None
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.amsgrad = amsgrad
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=self.epsilon, 
                                                  beta_1=self.beta_1, beta_2=self.beta_2, amsgrad=self.amsgrad)

    @tf.function
    def learn(self, noise_batch, data_batch, gen_cond_batch, discr_cond_batch):
        discr_variables = []
        discr_cost_fn = Discriminator_Cost()
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            # Capture Variables
            for block in self.discriminator_model.dense_img_blocks:
                for layer in block.layers:
                    if len(layer.variables)!=0:
                        tape.watch(layer.variables)
                        discr_variables.append(layer.variables[0])
                        discr_variables.append(layer.variables[1])
            
            for block in self.discriminator_model.dense_cond_blocks:
                for layer in block.layers:
                    if len(layer.variables)!=0:
                        tape.watch(layer.variables)
                        discr_variables.append(layer.variables[0])
                        discr_variables.append(layer.variables[1])
            
            for block in self.discriminator_model.dense_body_blocks:
                for layer in block.layers:
                    if len(layer.variables)!=0:
                        tape.watch(layer.variables)
                        discr_variables.append(layer.variables[0])
                        discr_variables.append(layer.variables[1])
            
            tape.watch(self.discriminator_model.out_sigmoid_layer.variables)
            discr_variables.append(self.discriminator_model.out_sigmoid_layer.variables[0])
            discr_variables.append(self.discriminator_model.out_sigmoid_layer.variables[1])
            self.batch_loss = discr_cost_fn(noise_batch=noise_batch, data_batch=data_batch, gen_cond_batch=gen_cond_batch, discr_cond_batch=discr_cond_batch, 
                                            generator_model=self.generator_model, discriminator_model=self.discriminator_model)
        grads = tape.gradient(self.batch_loss, discr_variables)
        tf.print("Discriminator Batch Loss: ", self.batch_loss)
        del tape
        self.optimizer.apply_gradients(zip(grads, discr_variables))
        return grads


class Generator_Learn:
    def __init__(self, generator_model, discriminator_model, learning_rate=5e-03, epsilon=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False):
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.batch_loss = None
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.amsgrad = amsgrad
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=self.epsilon, 
                                                  beta_1=self.beta_1, beta_2=self.beta_2, amsgrad=self.amsgrad)

    @tf.function
    def learn(self, noise_batch, gen_cond_batch, discr_cond_batch):
        gen_variables = []
        gen_cost_fn = Generator_Cost()
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            # Capture Variables
            for block in self.discriminator_model.dense_img_blocks:
                for layer in block.layers:
                    if len(layer.variables)!=0:
                        tape.watch(layer.variables)
                        discr_variables.append(layer.variables[0])
                        discr_variables.append(layer.variables[1])
            
            for block in self.discriminator_model.dense_cond_blocks:
                for layer in block.layers:
                    if len(layer.variables)!=0:
                        tape.watch(layer.variables)
                        discr_variables.append(layer.variables[0])
                        discr_variables.append(layer.variables[1])
            
            for block in self.discriminator_model.dense_body_blocks:
                for layer in block.layers:
                    if len(layer.variables)!=0:
                        tape.watch(layer.variables)
                        discr_variables.append(layer.variables[0])
                        discr_variables.append(layer.variables[1])
            
            tape.watch(self.generator_model.output_layer.variables)
            gen_variables.append(self.generator_model.output_layer.variables[0])
            gen_variables.append(self.generator_model.output_layer.variables[1])
            self.batch_loss = gen_cost_fn(noise_batch=noise_batch, generator_model=self.generator_model, gen_cond_batch=gen_cond_batch, 
                                          discr_cond_batch=discr_cond_batch, discriminator_model=self.discriminator_model)
        grads = tape.gradient(self.batch_loss, gen_variables)
        tf.print("Generator Batch Loss: ", self.batch_loss)
        del tape
        self.optimizer.apply_gradients(zip(grads, gen_variables))
        return grads