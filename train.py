

self.saver = tf.train.Saver(tf.trainable_variables(), save_relative_paths=True)
self.init = tf.global_variables_initializer()