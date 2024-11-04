import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import os
import filtros as fl  # Assume-se que este módulo contém funções para pré-processamento de imagens.


# Função para pré-processar uma imagem para ser usada pelo modelo
def preprocess_image_for_model(image):
    # Se a imagem for em escala de cinza (2D), adiciona uma dimensão extra
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    # Converte a imagem para um tensor e redimensiona para 255x255
    detected_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    detected_tensor = tf.image.resize(detected_tensor, [255, 255])
    # Normaliza os valores dos pixels para a faixa [0, 1]
    return detected_tensor / 255.0


# Função para carregar e processar imagens de um diretório
def load_and_process_images(directory_path, label, is_validation=False):
    images, labels = [], []

    # Itera sobre todos os arquivos no diretório
    for filename in os.listdir(directory_path):
        image_path = os.path.join(directory_path, filename)
        # Processa apenas arquivos de imagem
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            if is_validation:
                processed_image_tensor = preprocess_validation_image(image_path)
            else:
                image = cv2.imread(image_path)  # Lê a imagem
                processed_image = fl.imagem_preprocessada(
                    image
                )  # Aplica pré-processamento definido no módulo 'filtros'
                processed_image_tensor = preprocess_image_for_model(processed_image)

            images.append(
                processed_image_tensor
            )  # Adiciona a imagem processada à lista
            labels.append(label)  # Adiciona o rótulo correspondente à lista

    # Cria tensores a partir das listas de imagens e rótulos
    images_tensor = tf.stack(images)
    labels_tensor = tf.constant(labels)
    # Cria um conjunto de dados do TensorFlow com lotes de tamanho 32
    dataset = tf.data.Dataset.from_tensor_slices((images_tensor, labels_tensor))
    dataset = dataset.batch(32)
    return dataset


# Função para construir o modelo de rede neural
def build_model(input_shape=(255, 255, 1)):
    model = keras.Sequential(
        [
            layers.Conv2D(
                16,
                (5, 5),
                activation="relu",
                kernel_regularizer=l2(0.01),
                input_shape=input_shape,
            ),
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
            layers.Dense(1, activation="sigmoid"),  # Saída binária usando sigmoid
        ]
    )
    # Compila o modelo com o otimizador Adam e perda de entropia cruzada binária
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Função para prever se há derramamento de óleo em lote
def predict_oil_spill_in_batch(model, directory_path):
    processed_images = load_and_process_images(
        directory_path, label=0
    )  # 'label' é necessário, mas não relevante aqui
    predictions = model.predict(processed_images.map(lambda x, y: x))  # Faz previsões
    for i, prediction in enumerate(predictions):
        print(
            f"Imagem {i+1}: Probabilidade de mancha de óleo = {prediction[0]}"
        )  # Exibe a probabilidade


# Função para pré-processar uma imagem de validação
def preprocess_validation_image(image_path):
    image = cv2.imread(image_path)  # Lê a imagem
    hsv_image = cv2.cvtColor(
        image, cv2.COLOR_BGR2HSV
    )  # Converte para espaço de cor HSV

    # Define intervalo para detectar cor ciano (presumivelmente usada para identificar manchas de óleo)
    lower_cyan = np.array([85, 100, 100])
    upper_cyan = np.array([95, 255, 255])

    # Cria uma máscara para isolar a cor ciano
    cyan_mask = cv2.inRange(hsv_image, lower_cyan, upper_cyan)
    gray_image = cyan_mask  # A máscara é convertida em escala de cinza
    normalized_image = gray_image / 255.0  # Normaliza para [0, 1]
    resized_image = cv2.resize(
        normalized_image, (255, 255)
    )  # Redimensiona para 255x255
    final_image = np.expand_dims(resized_image, axis=-1)  # Adiciona dimensão do canal
    return tf.convert_to_tensor(final_image, dtype=tf.float32)


# Função principal de treinamento e previsão
if __name__ == "__main__":
    escolha = int(
        input("Quer carregar ou criar um modelo? 0/1 ")
    )  # Pergunta ao usuário se deseja carregar ou criar um modelo
    model = None

    if escolha == 0:
        model = load_model("model_ValoresAte1.h5")  # Carrega um modelo pré-treinado
    else:
        # Define os caminhos para os dados de treinamento e validação
        images_directory_path = r"C:\Users\gustavo\Desktop\oil-spill\train\images-train"
        drt_val_dataset = r"C:\Users\gustavo\Desktop\oil-spill\train\labels"

        # Carrega os datasets de treinamento e validação
        train_dataset = load_and_process_images(images_directory_path, label=1)
        val_dataset = load_and_process_images(
            drt_val_dataset, label=0, is_validation=True
        )

        # Constrói e treina o modelo
        model = build_model()
        early_stopping = EarlyStopping(
            monitor="accuracy", patience=3, restore_best_weights=True
        )  # Interrompe o treinamento se não houver melhora
        model.summary()

        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=6,
            callbacks=[early_stopping],
        )  # Treina o modelo por até 6 épocas
        model.save("model.h5")  # Salva o modelo treinado

    drt_test = r"C:\Users\gustavo\Desktop\oil-spill\test\images"  # Define o caminho para o conjunto de teste

    # Faz previsões em lote no conjunto de teste
    predict_oil_spill_in_batch(model, drt_test)
