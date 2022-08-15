import tensorflow as tf

class Discriminator_Loss(tf.keras.losses.Loss):
    @tf.function
    def __call__(self, noise_sample, data_sample, discriminator_model, generator_model):
        return tf.math.log(discriminator_model(data_sample)) + tf.math.log(1 - discriminator_model(generator_model(noise_sample)))

class Generator_Loss(tf.keras.losses.Loss):
    @tf.function
    def __call__(self, noise_sample, discriminator_model, generator_model):
        return tf.math.log(1 - discriminator_model(generator_model(noise_sample)))

class Discriminator_Learn:
    def __init__(self, generator_model, discriminator_model):
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.loss = None

    @tf.function
    def learn(self, noise_sample, data_sample):
        discr_variables = []
        discr_loss_fn = Discriminator_Loss()
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            # Capture Variables
            for block in self.discriminator_model.dense_blocks:
                for layer in block.layers:
                    if len(layer.variables)!=0:
                        tape.watch(layer.variables)
                        discr_variables.append(layer.variables[0])
                        discr_variables.append(layer.variables[1])
            tape.watch(self.discriminator_model.out_sigmoid_layer.variables)
            discr_variables.append(self.discriminator_model.out_sigmoid_layer.variables[0])
            discr_variables.append(self.discriminator_model.out_sigmoid_layer.variables[1])
            self.loss = discr_loss_fn(noise_sample=noise_sample, data_sample=data_sample, 
                              generator_model=self.generator_model, discriminator_model=self.discriminator_model)
        grads = tape.gradient(self.loss, discr_variables)
        del tape
        self.discriminator_model.optimizer.apply_gradients(zip(grads, discr_variables))
        return grads


class Generator_Learn:
    def __init__(self, generator_model, discriminator_model):
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.loss = None

    @tf.function
    def learn(self, noise_sample):
        gen_variables = []
        gen_loss_fn = Generator_Loss()
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            # Capture Variables
            for block in self.generator_model.dense_blocks:
                for layer in block.layers:
                    if len(layer.variables)!=0:
                        tape.watch(layer.variables)
                        gen_variables.append(layer.variables[0])
                        gen_variables.append(layer.variables[1])
            tape.watch(self.generator_model.output_layer.variables)
            gen_variables.append(self.generator_model.output_layer.variables[0])
            gen_variables.append(self.generator_model.output_layer.variables[1])
            self.loss = gen_loss_fn(noise_sample=noise_sample, generator_model=self.generator_model, discriminator_model=self.discriminator_model)
        grads = tape.gradient(self.loss, gen_variables)
        del tape
        self.generator_model.optimizer.apply_gradients(zip(grads, gen_variables))
        return grads