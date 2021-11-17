class UNet(BaseModel):
    def __init__(self, config):
       super().__init__(config)
       self.base_model = tf.keras.applications.MobileNetV2(input_shape=self.config.model.input, include_top=False)
       
    def load_data(self):
        self.dataset, self.info = DataLoader().load_data(self.config.data )
        self._preprocess_data()

    def build(self):
        self.model = tf.keras.Model(inputs=inputs, outputs=x)

    def train(self):
        self.model.compile(optimizer=self.config.train.optimizer.type,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=self.config.train.metrics)

        model_history = self.model.fit(self.train_dataset, epochs=self.epoches,
                                       steps_per_epoch=self.steps_per_epoch,
                                       validation_steps=self.validation_steps,
                                       validation_data=self.test_dataset)

        return model_history.history['loss'], model_history.history['val_loss']

    def evaluate(self):
        predictions = []
        for image, mask in self.dataset.take(1):
            predictions.append(self.model.predict(image))

        return predictions
