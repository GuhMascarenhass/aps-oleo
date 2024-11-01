import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')



def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel(f"{predicted_label} ({true_label})", color=color)



# Load data
oil_training = r"C:\Users\gustavo\Desktop\oil-spill\test"


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    oil_training,
    image_size=(400, 400),  # Redimensionar todas as imagens para 224x224
    batch_size=32,          # Número de imagens por lote
    label_mode='int',       # Pode ser 'int', 'categorical' ou 'binary'
    shuffle=True,           # Misturar as imagens
    validation_split=0.2,   # Usar 20% para validação
    subset='training',
    seed=123
)

# Criar dataset de validação
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    oil_training,
    image_size=(400, 400),
    batch_size=32,
    label_mode='int',
    shuffle=True,
    validation_split=0.2,
    subset='validation',
    seed=123
)

# Normalizando as imagens
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
normalized_val_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# Criar o modelo ajustado para 224x224 imagens
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # Ajuste o número de classes conforme seu dataset
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
model.fit(normalized_dataset, validation_data=normalized_val_dataset, epochs=5)

# Avaliar o modelo
loss, accuracy = model.evaluate(normalized_val_dataset)
print(f"Validation Accuracy: {accuracy}")

# Prever os resultados
for images, labels in normalized_val_dataset.take(1):
    predictions = model.predict(images)

# Plotar algumas previsões
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, labels, images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, labels)
plt.show()
