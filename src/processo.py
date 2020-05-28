import numpy as np
import cv2

# VideoCapture(id da cam ou caminho do vídeo)
cap = cv2.VideoCapture(0)  # Acessa a webcam
while True:
    # .read() lê a webcam retornando um frame e um boolean que indica se teve sucesso ou não na requisição da webcam
    (x, frame) = cap.read()

    # Se for false sai do while
    if not x:
        break

    # Redimensiona o frame pela metade utilizando a técnica slicing
    frame = frame[::2, ::2]

    # Converte para um frame em tons de cinza
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # cv2.cvtColor(<imagem>, <Tipo a converter>)

    # Suaviza ou borra o frame e tira o ruído dele. Aumenta a precisão das bordas     Sempre ímpares
    frame_sv = cv2.GaussianBlur(frame_pb, (3, 3),
                                0)  # cv2.GaussianBlur(<imagem>, (<largura>, <altura>), <qtd de desvios no eixo X e Y>)

    # Binariza o frame fazendo um cálculo das intensidades dos frames, onde pixels proximos de 0 se tornam 0 e os pixels proximos ao 255 ficam 255
    (T, frame_bin1) = cv2.threshold(frame_sv, 160, 255, cv2.THRESH_BINARY)  # cv2.threshold(<imagem>, <valor de intensidade, nesse caso 160>, <valor máximo>, <Tipo>) retorna o frame binarizado e um retVal
    (T, frame_bin2) = cv2.threshold(frame_sv, 160, 255, cv2.THRESH_BINARY_INV)  # Inverso

    # Detecar bordas no frame usando a função Canny
    bordas = cv2.Canny(frame_bin1, 100, 300)  # cv2.Canny(<imagem>, <limiar 1>, <limiar2>) => Se x > limiar 2 = borda; se x < limiar 1 = não borda; Se  20 < x < 120 = depende

    # Cria uma pilha vertical contendo pilhas horizontais, tudo isso para mostrar todas as alterações de uma vez
    pilha = np.vstack([
        np.hstack([frame_pb, frame_sv]),
        np.hstack([frame_bin1, bordas])
    ])

    # Mostra o frame
    cv2.imshow("Original", frame)
    cv2.imshow("Resultado", pilha)

    # Aguarda o usuário digitar a tecla 'q' para quebrar o processo
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
