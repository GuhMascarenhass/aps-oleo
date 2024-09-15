import cv2
import numpy as np

def detectarmanchas_oleo(imagem_caminho):
    # Carrega a imagem
    imagem = cv2.imread(imagem_caminho)
    imagem_redimensionada = cv2.resize(imagem, (600, 400))  # Redimensionar para facilitzar o processamento

    # Converter para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2GRAY)

    # Aplicar um filtro gaussiano para suavizar a imagem
    imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
    cv2.imshow("ads", imagem_suavizada)
    # Detectar bordas usando o Canny
    bordas = cv2.Canny(imagem_suavizada, 50, 150)

    # Realizar a segmentação usando uma limiarização
    _,limiar = cv2.threshold(bordas, 20, 30,  cv2.THRESH_BINARY)

    # Detectar contornos para as manchas
    contornos, _  = cv2.findContours(limiar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar contornos sobre a imagem original
    imagem_contornos = imagem_redimensionada.copy()
    cv2.drawContours(imagem_contornos, contornos, -1, (0, 255, 0), 2)
    laplaciano = cv2.Laplacian(imagem_cinza, cv2.CV_64F)

# Converter a escala para positiva
    laplaciano = np.uint8(np.absolute(laplaciano))


    # Mostrar as imagens (opcional, se você estiver rodando localmente)
    cv2.imshow('Imagem Original', imagem_redimensionada)
    cv2.imshow('Manchas Detectadas', imagem_contornos)
    cv2.imshow("sdf", bordas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imagem_contornos
imagem_resultado = detectarmanchas_oleo(r"C:\Users\gustavo\Downloads\img4.jpeg")
