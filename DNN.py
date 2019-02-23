import tensorflow as tf



#preparring the data for input
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def model1():
    #defining the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(40, 40)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.Dense(512, activatoin=tf.nn.relu), 
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(43, activation=tf.nn.softmax)
    ])
    return model

model = model
#execution
#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])
#
#model.fit(x_train, y_train, epochs=5)
#model.evaluate(x_test, y_test)

