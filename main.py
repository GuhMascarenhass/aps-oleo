import cv2
import numpy as np

def sobel(image):
    # Aplicar a convolução com as máscaras Sobel
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel na direção X
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel na direção Y

    # Converter o resultado para uint8 (valores de 0 a 255) para visualização
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # Combinar os gradientes X e Y
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # Aplicar o desfoque Gaussiano na imagem resultante do Sobel
    blurred_sobel = cv2.GaussianBlur(sobel_combined, (5, 5), 0)

    return blurred_sobel

def laplaciano(imagem):
    # Aplicar o filtro Laplaciano
    laplaciano = cv2.Laplacian(imagem, cv2.CV_64F)

    # Converter a escala para positiva
    laplaciano = np.uint8(np.absolute(laplaciano))

    return laplaciano

def gagau(img_cinza, tamanho_kernel):
    # Aplicar o filtro Gaussiano
    gau_img = cv2.GaussianBlur(img_cinza, tamanho_kernel, 0)
    return gau_img

def contornos(imagem):
    # Detecção de bordas usando o Canny
    bordas = cv2.Canny(imagem, 30, 40)

    # Realizar a segmentação usando limiarização
    _, limiar = cv2.threshold(bordas, 20, 255, cv2.THRESH_BINARY)

    # Detectar contornos
    contornos, _ = cv2.findContours(limiar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar contornos sobre a imagem original
    imagem_contornos = imagem.copy()
    cv2.drawContours(imagem_contornos, contornos, -1, (0, 255, 0), 2)

    return imagem_contornos

def prewitt(image):
    # Aplicar os operadores Prewitt (convolução com máscaras horizontais e verticais)
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    prewitt_x = cv2.filter2D(image, -1, kernel_x)
    prewitt_y = cv2.filter2D(image, -1, kernel_y)

    # Combinar os gradientes
    prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

    return prewitt_combined

def tratamento_imagem():
    # Solicitar o caminho da imagem ao usuário
    drt = "2.webp"
    imagem = cv2.imread(drt)

    # Redimensionar a imagem para uma visualização mais prática
    img_resize = cv2.resize(imagem, (400, 400))

    # Converter para escala de cinza
    img_cinza = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    # Aplicar desfoque (filtro blur)
    blur = cv2.blur(img_resize, (5, 5))

    return imagem, img_resize, img_cinza, blur

# Exemplo de uso do pipeline de processamento
if __name__ == "__main__":
    # Processar a imagem
    imagem, img_resize, img_cinza, blur = tratamento_imagem()

    hsv = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
    blur1 = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV_FULL)
    
    cv2.imshow('hsv', hsv)
    cv2.imshow('hsv2', hsv2)
    cv2.imshow('blur', blur1)



    # Aplicar Sobel
    '''sobel_result = sobel(img_cinza)
    cv2.imshow('Sobel', sobel_result)

    # Aplicar Laplaciano
    laplacian_result = laplaciano(blur)
    cv2.imshow('Laplaciano', laplacian_result)

    # Aplicar Prewitt
    prewitt_result = prewitt(img_cinza)
    cv2.imshow('Prewitt', prewitt_result)

    # Detectar contornos
    contornos_result = contornos(img_resize)
    cv2.imshow('Contornos', contornos_result)
'''
    # Aguardar a tecla de encerramento
    cv2.waitKey(0)
    cv2.destroyAllWindows()
