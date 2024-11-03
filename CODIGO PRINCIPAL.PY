import cv2
import numpy as np


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(20, 20)):
    """
    Aplica o CLAHE (equalização adaptativa) na imagem em escala de cinza,
    com parâmetros ajustáveis de limite de contraste e tamanho da grade.

    Parameters:
    - image: Imagem em escala de cinza.
    - clip_limit: Limite de contraste para o CLAHE.
    - tile_grid_size: Tamanho da grade local para equalizar o histograma.

    Returns:
    - Imagem com CLAHE aplicado.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def apply_clahe_to_dark_areas(image, clip_limit=4.0, tile_grid_size=(30, 30)):
    """
    Aplica o CLAHE apenas em áreas escuras da imagem para evitar aumento de contraste em áreas já iluminadas.
    """
    # Converter para escala de cinza, se ainda não estiver
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Aplicar uma máscara para regiões mais escuras
    # 45
    _, mask = cv2.threshold(gray_image, 140, 255, cv2.THRESH_BINARY_INV)

    # Aplicar CLAHE na imagem inteira
    clahe_result = apply_clahe(gray_image, clip_limit, tile_grid_size)

    # Combinar apenas as áreas de interesse (escuro) com o CLAHE aplicado
    combined = cv2.bitwise_and(clahe_result, clahe_result, mask=mask)

    # Adicionar de volta as regiões claras originais
    combined = cv2.add(
        combined, cv2.bitwise_and(gray_image, gray_image, mask=cv2.bitwise_not(mask))
    )

    return combined


def contornos(imagem):
    # Detecção de bordas usando o Canny
    cv2.imshow("im", imagem)
    bordas = cv2.Canny(imagem, 10, 10)

    # Realizar a segmentação usando limiarização
    _, limiar = cv2.threshold(bordas, 5, 5, cv2.THRESH_BINARY)

    # Detectar contornos
    contornos, _ = cv2.findContours(limiar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar contornos sobre a imagem original
    imagem_contornos = imagem.copy()
    cv2.drawContours(imagem_contornos, contornos, -1, (225, 0, 0), 3)

    return imagem_contornos


def prewitt(image):
    # Aplicar os operadores Prewitt (convolução com máscaras horizontais e verticais)
    kernel_x = np.array([[1, 0, -1], [1, 1, -1], [1, 0, -1]])

    kernel_y = np.array([[1, -1, 1], [0, 1, 0], [-1, 1, -1]])

    prewitt_x = cv2.filter2D(image, -1, kernel_x)
    prewitt_y = cv2.filter2D(image, -1, kernel_y)

    # Combinar os gradientes
    prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

    return prewitt_combined


def tratamento_imagem():
    imagem = cv2.imread(r"C:\Users\gustavo\Desktop\oil-spill\test\images\img_0033.jpg")

    # Redimensionar a imagem para uma visualização mais prática
    img_resize = cv2.resize(imagem, (700, 700))

    return img_resize



def paint_blue_contours(combined_image, original_image):
    """
    Pinta de azul os contornos destacados na imagem `combined` e os desenha na imagem `original`.
    """
    # Detectar contornos na imagem combinada
    contours, _ = cv2.findContours(combined_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Criar uma cópia da imagem original
    colored_image = original_image.copy()

    # Pintar os contornos de azul (BGR = (255, 0, 0))
    cv2.drawContours(colored_image, contours, -1, (255, 0, 0), 2)

    return colored_image


img_resize = tratamento_imagem()

mediana = cv2.medianBlur(img_resize, 15)


preto1 = apply_clahe_to_dark_areas(mediana)

segmento = paint_blue_contours(preto1, preto1)


cv2.imshow("mg-preto-mediana.jpg", preto1)
cv2.imshow("mg-preto-sgmento.jpg", segmento)


a = prewitt(mediana)

s2 = prewitt(img_resize)

s3 = cv2.medianBlur(a, 15)

preto = apply_clahe_to_dark_areas(s3)

cv2.imshow("mg-preto.jpg", preto)

cv2.waitKey(0)
cv2.destroyAllWindows()
