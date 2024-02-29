from tensorflow import keras
import numpy as np
import json

# SETTINGS
INNER_NODES = [128]  # Both INNER_NODES and INNER_ACTIVATION must have the same amount of entries
INNER_ACTIVATION = ['tanh']
OPTIMIZER = 'adam'
LOSS = 'sparse_categorical_crossentropy'
EPOCHS = 20
MODEL_NAME = 'nexus'  # This is the name of the file the model will save too

# The training data was created from my minimax noughts and crosses AI,
# available at https://www.harryheywood.co.uk/ai-playground/noughts-and-crosses/index.html
train_data_file = open('data/train_data.json')
train_values_file = open('data/train_values.json')
train_data = np.asfarray(json.load(train_data_file))
train_values = np.asfarray(json.load(train_values_file))

model = keras.Sequential([
    keras.layers.Dense(9)
])

for i in range(len(INNER_NODES)):
    model.add(keras.layers.Dense(INNER_NODES[i], activation=INNER_ACTIVATION[i]))

model.add(keras.layers.Dense(9, activation='softmax'))
model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])
model.fit(train_data, train_values, epochs=EPOCHS)
model.save("models/" + MODEL_NAME + ".h5")
model.summary()
