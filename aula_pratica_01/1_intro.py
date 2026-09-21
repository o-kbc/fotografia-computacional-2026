import cv2
import numpy as np

# pip install opencv-python

# ============================================================
# 1. CARREGANDO UMA IMAGEM
# ============================================================

caminho_imagem = "input1.png"

imagem = cv2.imread(caminho_imagem)

# Verifica se a imagem foi carregada corretamente
if imagem is None:
    print("Erro: não foi possível carregar a imagem.")
    exit()

print("\n========== IMAGEM CARREGADA ==========")
print("Tipo do objeto:", type(imagem))
print("Formato (shape):", imagem.shape)
print("Tipo dos valores (dtype):", imagem.dtype)

cv2.imshow("Imagem original", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 2. ENTENDENDO AS DIMENSÕES DA IMAGEM
# ============================================================

altura, largura, canais = imagem.shape

print("\n========== DIMENSOES ==========")
print("Altura:", altura, "pixels")
print("Largura:", largura, "pixels")
print("Numero de canais:", canais)

print(f"Resolucao: {largura} x {altura}")

# ============================================================
# 3. ENTENDENDO AS CORES
# ============================================================

pixel = imagem[0, 0]

print("\n========== PIXEL ==========")
print("Pixel na posicao [0, 0]:", pixel)

B, G, R = pixel

print("Blue :", B)
print("Green:", G)
print("Red  :", R)


# ============================================================
# 4. POSIÇÃO DOS PIXELS
# ============================================================


x = largura // 2
y = altura // 2

pixel_central = imagem[y, x]

print("\n========== POSICAO ==========")
print("Centro da imagem:")
print("x =", x)
print("y =", y)
print("Valor BGR =", pixel_central)

imagem_centro = imagem.copy()

cv2.circle(
    imagem_centro,
    (x, y),
    10,
    (0, 0, 255),
    -1
)

cv2.imshow("Centro da imagem", imagem_centro)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 5. ACESSANDO UMA REGIÃO DA IMAGEM
# ============================================================


x1 = largura // 4
x2 = 3 * largura // 4

y1 = altura // 4
y2 = 3 * altura // 4

recorte = imagem[y1:y2, x1:x2]

print("\n========== RECORTE ==========")
print("Imagem original:", imagem.shape)
print("Recorte:", recorte.shape)

cv2.imshow("Imagem original", imagem)
cv2.imshow("Recorte", recorte)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 6. IMAGEM EM ESCALA DE CINZA
# ============================================================

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

print("\n========== ESCALA DE CINZA ==========")
print("Shape original:", imagem.shape)
print("Shape cinza:", cinza.shape)

print("Pixel cinza [0,0]:", cinza[0, 0])

cv2.imshow("Imagem colorida", imagem)
cv2.imshow("Escala de cinza", cinza)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 7. AUMENTO DE BRILHO
# ============================================================

# Valores maiores deixam a imagem mais clara.

valor_brilho = 60

imagem_clara = cv2.convertScaleAbs(
    imagem,
    alpha=1,
    beta=valor_brilho
)

print("\n========== BRILHO ==========")
print("Brilho adicionado:", valor_brilho)

print("Pixel original:", imagem[0, 0])
print("Pixel com brilho:", imagem_clara[0, 0])

cv2.imshow("Original", imagem)
cv2.imshow("Mais brilho", imagem_clara)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 8. REDIMENSIONAMENTO
# ============================================================

nova_largura = largura // 2
nova_altura = altura // 2

imagem_menor = cv2.resize(
    imagem,
    (nova_largura, nova_altura)
)

print("\n========== REDIMENSIONAMENTO ==========")
print("Original:", imagem.shape)
print("Redimensionada:", imagem_menor.shape)

cv2.imshow("Original", imagem)
cv2.imshow("50 por cento", imagem_menor)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 9. ESPELHAMENTO / FLIP
# ============================================================

horizontal = cv2.flip(imagem, 1)
vertical = cv2.flip(imagem, 0)

print("\n========== FLIP ==========")
print("Flip horizontal e vertical executados.")

cv2.imshow("Original", imagem)
cv2.imshow("Horizontal", horizontal)
cv2.imshow("Vertical", vertical)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 10. ROTACAO
# ============================================================

# Centro de rotação
centro = (largura // 2, altura // 2)

angulo = 45

escala = 1.0

matriz_rotacao = cv2.getRotationMatrix2D(
    centro,
    angulo,
    escala
)

print("\n========== ROTACAO ==========")
print("Centro:", centro)
print("Angulo:", angulo)
print("Escala:", escala)

print("Matriz de rotacao:")
print(matriz_rotacao)

imagem_rotacionada = cv2.warpAffine(
    imagem,
    matriz_rotacao,
    (largura, altura)
)

cv2.imshow("Original", imagem)
cv2.imshow("Rotacionada 45 graus", imagem_rotacionada)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 11. DESENHANDO SOBRE A IMAGEM
# ============================================================

desenho = imagem.copy()

# Linha
cv2.line(
    desenho,
    (0, 0),
    (largura, altura),
    (0, 255, 0),
    3
)

# Retângulo
cv2.rectangle(
    desenho,
    (50, 50),
    (250, 200),
    (255, 0, 0),
    3
)

# Círculo
cv2.circle(
    desenho,
    (largura // 2, altura // 2),
    50,
    (0, 0, 255),
    3
)

print("\n========== DESENHO ==========")

cv2.imshow("Desenhos", desenho)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 12. ADICIONANDO TEXTO
# ============================================================

imagem_texto = imagem.copy()

cv2.putText(
    imagem_texto,
    "Introducao ao OpenCV",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow("Texto", imagem_texto)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 17. SALVANDO UMA IMAGEM
# ============================================================

cv2.imwrite(
    "imagem_rotacionada.jpg",
    imagem_rotacionada
)

print("\n========== SALVAMENTO ==========")
print("Imagem salva como: imagem_rotacionada.jpg")

