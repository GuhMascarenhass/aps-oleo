import cv2
import numpy as np
import tensorflow as tf


# Pre processamento de imagens (Aplicação dos filtros mediano, e clahe)
def imagem_preprocessada(img_redimencionada):
    imagem_mediana = tratamento_imagem(img_redimencionada)
    imagem_clahe = apply_clahe_to_dark_areas(imagem_mediana)
    imagem_preti = prewitt(imagem_clahe)

    return imagem_preti


# Função para carregar e redimensionar uma imagem
def tratamento_imagem(img):
    img_resize = cv2.resize(img, (700, 700))
    img_resize = cv2.medianBlur(img_resize, 15)

    return img_resize


# Função para aplicar CLAHE (Equalização Adaptativa do Histograma)
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(20, 20)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


# Função para ajustar parâmetros dinamicamente com base na luminosidade da imagem
    # Calcular o histograma para análise de intensidade
def adjust_parameters(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    brightness = np.mean(image)

    # Ajustar os parâmetros de CLAHE e limiarização com base no brilho
    if brightness < 150:  # Imagem escura
        clip_limit = 4.0
        threshold_value = 60
    elif brightness >= 150:  # Imagem clara
        clip_limit = 6.0
        threshold_value = 165
    
    return clip_limit, threshold_value


# Função para aplicar CLAHE em áreas escuras da imagem
def apply_clahe_to_dark_areas(image):
    # Converter para escala de cinza se a imagem for colorida
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Ajustar parâmetros dinamicamente
    clip_limit, threshold_value = adjust_parameters(gray_image)

    # Criar uma máscara para regiões escuras
    _, mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Aplicar CLAHE com os parâmetros ajustados
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    clahe_result = clahe.apply(gray_image)

    # Combinar CLAHE nas áreas escuras e manter áreas claras originais
    combined = cv2.bitwise_and(clahe_result, clahe_result, mask=mask)
    combined = cv2.add(
        combined, cv2.bitwise_and(gray_image, gray_image, mask=cv2.bitwise_not(mask))
    )
    return combined


# Função para aplicar o filtro de bordas Prewitt
def prewitt(image):
    # Definindo kernels para detecção de bordas
    kernel_x = np.array([[1, 0, -1], [1, 1, -1], [1, 0, -1]])
    kernel_y = np.array([[1, -1, 1], [0, 1, 0], [-1, 1, -1]])

    # Aplicando os kernels na imagem
    prewitt_x = cv2.filter2D(image, -1, kernel_x)
    prewitt_y = cv2.filter2D(image, -1, kernel_y)

    # Combinando os resultados dos dois filtros
    prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
    return prewitt_combined


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

