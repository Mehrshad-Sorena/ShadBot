import tensorflow as tf

class ModelGenerator(tf.keras.Model):	

	def __init__(self, label_index=None):

		super().__init__()
		self.label_index = label_index

		self.MAX_EPOCHS = 2

	def call(self, inputs):

		print('*********************************************')

		if self.label_index is None: return inputs

		print('inputs =====> ', inputs)

		result = inputs[:, :, self.label_index]

		print('*********************************************')

		return result[:, :, tf.newaxis]

	def compile_and_fit(self, model, window, patience=0):

		early_stopping = tf.keras.callbacks.EarlyStopping(
															monitor = 'val_loss',
															patience = patience,
                                                    		mode = 'min'
                                                    		)

		model.compile(
						loss = tf.keras.losses.MeanSquaredError(),
						optimizer = tf.keras.optimizers.Adam(),
						metrics = [tf.keras.metrics.MeanAbsoluteError()]
						)

		print('window train = ', window.train)

		history = model.fit(
							window.train, 
							epochs = self.MAX_EPOCHS,
							validation_data = window.val,
							callbacks = [early_stopping]
							)

		return history