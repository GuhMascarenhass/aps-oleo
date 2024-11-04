import cv2
import numpy as np
import tensorflow as tf


# Preprocessamento de imagens (aplicação de filtros mediano, CLAHE e Prewitt)
def imagem_preprocessada(img_redimencionada):
    # Aplica o filtro mediano
    imagem_mediana = tratamento_imagem(img_redimencionada)
    # Aplica CLAHE para melhorar o contraste em áreas escuras
    imagem_clahe = apply_clahe_to_dark_areas(imagem_mediana)
    # Aplica o filtro de bordas Prewitt
    imagem_preti = prewitt(imagem_clahe)

    return imagem_preti


# Função para redimensionar a imagem e aplicar um filtro mediano
def tratamento_imagem(img):
    # Redimensiona a imagem para 700x700 pixels
    img_resize = cv2.resize(img, (700, 700))
    # Aplica um filtro mediano para suavizar a imagem e reduzir o ruído
    img_resize = cv2.medianBlur(img_resize, 15)

    return img_resize


# Função para aplicar CLAHE (Equalização Adaptativa do Histograma)
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(20, 20)):
    # Cria o objeto CLAHE com os parâmetros especificados
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # Aplica CLAHE à imagem
    return clahe.apply(image)


# Função para ajustar parâmetros dinamicamente com base na luminosidade da imagem
def adjust_parameters(image):
    # Calcula o histograma da imagem
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Calcula o brilho médio da imagem
    brightness = np.mean(image)

    # Ajusta os parâmetros de CLAHE e limiarização com base no brilho
    if brightness < 150:  # Imagem considerada escura
        clip_limit = 4.0
        threshold_value = 60
    else:  # Imagem considerada clara
        clip_limit = 6.0
        threshold_value = 165

    return clip_limit, threshold_value


# Função para aplicar CLAHE em áreas escuras da imagem
def apply_clahe_to_dark_areas(image):
    # Converte a imagem para escala de cinza, se for colorida
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Ajusta dinamicamente os parâmetros de CLAHE e limiarização
    clip_limit, threshold_value = adjust_parameters(gray_image)

    # Cria uma máscara para as regiões escuras
    _, mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Aplica CLAHE com os parâmetros ajustados
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    clahe_result = clahe.apply(gray_image)

    # Combina as áreas processadas com CLAHE com as áreas claras originais
    combined = cv2.bitwise_and(clahe_result, clahe_result, mask=mask)
    combined = cv2.add(
        combined, cv2.bitwise_and(gray_image, gray_image, mask=cv2.bitwise_not(mask))
    )
    return combined


# Função para aplicar o filtro de bordas Prewitt
def prewitt(image):
    # Definindo kernels para detecção de bordas horizontais e verticais
    kernel_x = np.array([[1, 0, -1], [1, 1, -1], [1, 0, -1]])
    kernel_y = np.array([[1, -1, 1], [0, 1, 0], [-1, 1, -1]])

    # Aplicando os kernels na imagem
    prewitt_x = cv2.filter2D(image, -1, kernel_x)
    prewitt_y = cv2.filter2D(image, -1, kernel_y)

    # Combina os resultados dos dois filtros para detectar bordas em ambas as direções
    prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
    return prewitt_combined


# Função para pré-processar uma imagem de validação
def preprocess_validation_image(image_path):
    # Carrega a imagem do caminho especificado
    image = cv2.imread(image_path)

    # Converte a imagem para o espaço de cor HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define o intervalo para detectar o tom de ciano
    lower_cyan = np.array([85, 100, 100])
    upper_cyan = np.array([95, 255, 255])

    # Cria uma máscara para isolar a cor ciano
    cyan_mask = cv2.inRange(hsv_image, lower_cyan, upper_cyan)

    # Converte a máscara para escala de cinza
    gray_image = cyan_mask

    # Normaliza os valores dos pixels para a faixa [0, 1]
    normalized_image = gray_image / 255.0

    # Redimensiona a imagem normalizada para 255x255
    resized_image = cv2.resize(normalized_image, (255, 255))

    # Expande a dimensão para incluir o canal
    final_image = np.expand_dims(resized_image, axis=-1)

    return tf.convert_to_tensor(final_image, dtype=tf.float32)
