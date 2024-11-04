import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import os
import filtros as fl

def preprocess_image_for_model(image):
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    detected_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    detected_tensor = tf.image.resize(detected_tensor, [255, 255])
    return detected_tensor / 255.0

def load_and_process_images(directory_path, label, is_validation=False):
    images, labels = [], []

    for filename in os.listdir(directory_path):
        image_path = os.path.join(directory_path, filename)
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            if is_validation:
                processed_image_tensor = preprocess_validation_image(image_path)
            else:
                image = cv2.imread(image_path)
                processed_image = fl.imagem_preprocessada(image)
                processed_image_tensor = preprocess_image_for_model(processed_image)
                
            images.append(processed_image_tensor)
            labels.append(label)
    
    images_tensor = tf.stack(images)
    labels_tensor = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((images_tensor, labels_tensor))
    dataset = dataset.batch(32)
    return dataset


# Modelo Simplificado
def build_model(input_shape=(255, 255, 1)):
    model = keras.Sequential([
        layers.Conv2D(16, (5, 5), activation="relu", kernel_regularizer=l2(0.01), input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        layers.Flatten(),
        layers.Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Previsão em lote
def predict_oil_spill_in_batch(model, directory_path):
    processed_images = load_and_process_images(directory_path, label=0)  # 'label' é um parâmetro obrigatório, mas irrelevante aqui
    predictions = model.predict(processed_images.map(lambda x, y: x))
    for i, prediction in enumerate(predictions):
        print(f"Imagem {i+1}: Probabilidade de mancha de óleo = {prediction[0]}")

def preprocess_validation_image(image_path):
    # Carregar a imagem
    image = cv2.imread(image_path)
    
    # Converter para o espaço de cor HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definir o intervalo para o tom de ciano
    lower_cyan = np.array([85, 100, 100])  # Ajuste conforme necessário
    upper_cyan = np.array([95, 255, 255])
    
    # Criar uma máscara para isolar a cor ciano
    cyan_mask = cv2.inRange(hsv_image, lower_cyan, upper_cyan)
    
    # Converter a máscara para escala de cinza
    gray_image = cyan_mask
    
    # Normalizar os valores dos pixels para [0, 1]
    normalized_image = gray_image / 255.0
    
    # Redimensionar para 255x255
    resized_image = cv2.resize(normalized_image, (255, 255))
    
    # Expandir dimensões para adicionar o canal
    final_image = np.expand_dims(resized_image, axis=-1)
    
    return tf.convert_to_tensor(final_image, dtype=tf.float32)


# Treinamento
if __name__ == "__main__":

    escolha = int(input("Quer carregar ou criar um modelo? 0/1 "))
    model = None

    if escolha == 0:
        model = load_model("model_ValoresAte1.h5")
    else:
        images_directory_path = r"C:\Users\gustavo\Desktop\oil-spill\train\images-train"
        drt_val_dataset = r"C:\Users\gustavo\Desktop\oil-spill\train\labels"
        train_dataset = load_and_process_images(images_directory_path, label=1)
        val_dataset = load_and_process_images(drt_val_dataset, label=0, is_validation=True)
        model = build_model()
        early_stopping = EarlyStopping(monitor="accuracy", patience=3, restore_best_weights=True)
        model.summary()

        model.fit(train_dataset, validation_data=val_dataset, epochs=6, callbacks=[early_stopping])
        model.save("model.h5")

    drt_test = r"C:\Users\gustavo\Desktop\oil-spill\test\images"    
        
    # Fazer previsões
    predict_oil_spill_in_batch(model, drt_test)
