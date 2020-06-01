import numpy as np
import cv2

for c in range(4):
    img = cv2.imread(
        'C:/Users/Friday/software-object-detection/assets/image'+str(c+1)+'.jpeg')
    largura = img.shape[1]
    altura = img.shape[0]
    proporcao = float(altura/largura)
    largura_nova = 300  # em pixels
    altura_nova = int(largura_nova*proporcao)
    tamanho_novo = (largura_nova, altura_nova)
    img = cv2.resize(img, tamanho_novo, interpolation=cv2.INTER_AREA)

    # Converte para um img em tons de cinza
    # cv2.cvtColor(<imagem>, <Tipo a converter>)
    img_pb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Suaviza ou borra o img e tira o ruído dele. Aumenta a precisão das bordas Sempre ímpares
    # cv2.GaussianBlur(<imagem>, (<largura>, <altura>), <qtd de desvios no eixo X e Y>)
    img_sv = cv2.GaussianBlur(img_pb, (3, 3), 0)

    # Binariza o img fazendo um cálculo das intensidades dos imgs, onde pixels proximos de 0 se tornam 0 e os pixels proximos ao 255 ficam 255
    # cv2.threshold(<imagem>, <valor de intensidade, nesse caso 160>, <valor máximo>, <Tipo>) retorna o img binarizado e um retVal
    (T, img_bin1) = cv2.threshold(img_sv, 200, 255, cv2.THRESH_BINARY)
    (T, img_bin2) = cv2.threshold(img_sv, 200,
                                  255, cv2.THRESH_BINARY_INV)  # Inverso

    # Detecar bordas no img usando a função Canny
    # cv2.Canny(<imagem>, <limiar 1>, <limiar2>) => Se x > limiar 2 = borda; se x < limiar 1 = não borda; Se  20 < x < 120 = depende
    bordas = cv2.Canny(img_bin1, 50, 150, apertureSize=5)
    cv2.imshow("bordas", bordas)

    
    # Aplica a função goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(bordas, 100, 0.5, 10)
    nome = ''

    # 100 = número de cantos
    # 0.1 = qualidade (0 a 1)
    # 10 = distancia minima entre cantos (que são detectados)
    if corners is not None:
        corners = np.int0(corners)
        if len(corners) == 3:
            nome = 'Triangulo'
        elif len(corners) == 4:
            nome = 'Quadrado'
        elif len(corners) == 5:
            nome = 'Pentagono'
        elif len(corners) == 6:
            nome = 'Hexagono'
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    # Mostra o img
    cv2.imshow(nome, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
