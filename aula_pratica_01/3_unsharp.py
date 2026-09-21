import cv2
import numpy as np

# pip install opencv-python

imagem = cv2.imread("input3.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Imagem Original", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ============================================================
# 2. Unsharp MASK
# ============================================================
#
# A ideia da Unsharp Mask e:
#
# 1. Criar uma versao borrada da imagem
# 2. Calcular os detalhes:
#
#       mascara = original - borrada
#
# 3. Somar esses detalhes novamente na imagem:
#
#       sharpened = original + quantidade * mascara
# ============================================================


# ============================================================
# 3. CRIANDO UMA IMAGEM SUAVIZADA
# ============================================================


sigma = 2.0

imagem_blur = cv2.GaussianBlur(
    imagem,
    (0, 0),
    sigmaX=sigma,
    sigmaY=sigma
)

print("\n========== GAUSSIAN BLUR ==========")
print("Sigma:", sigma)
print("Shape:", imagem_blur.shape)
print("Media:", imagem_blur.mean())

cv2.imshow("Original", imagem)
cv2.imshow("Gaussian Blur", imagem_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 4. CALCULANDO A MASCARA DE DETALHES
# ============================================================


imagem_float = imagem.astype(np.float32)
blur_float = imagem_blur.astype(np.float32)

mascara = imagem_float - blur_float

print("\n========== MASCARA UNSHARP ==========")
print("Tipo:", mascara.dtype)
print("Valor minimo:", mascara.min())
print("Valor maximo:", mascara.max())
print("Media:", mascara.mean())

print("\nPequena regiao da mascara:")
print(mascara[100:105, 100:105])


# ============================================================
# 5. VISUALIZANDO A MASCARA
# ============================================================

mascara_visual = cv2.convertScaleAbs(mascara)

cv2.imshow("Original", imagem)
cv2.imshow("Mascara de detalhes", mascara_visual)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 6. APLICANDO A UNSHARP MASK
# ============================================================
#
# sharpened = original + quantidade * mascara
#
# quantidade:
#
# 0.5 -> nitidez suave
# 1.0 -> nitidez moderada
# 2.0 -> nitidez forte
#
# ============================================================

quantidade = 1.0

imagem_sharp = imagem_float + quantidade * mascara

# Limita os valores entre 0 e 255
imagem_sharp = np.clip(imagem_sharp, 0, 255)

# Converte novamente para uint8
imagem_sharp = imagem_sharp.astype(np.uint8)

print("\n========== UNSHARP MASK ==========")
print("Quantidade:", quantidade)
print("Shape:", imagem_sharp.shape)
print("Tipo:", imagem_sharp.dtype)
print("Valor minimo:", imagem_sharp.min())
print("Valor maximo:", imagem_sharp.max())
print("Media:", imagem_sharp.mean())

cv2.imshow("Original", imagem)
cv2.imshow("Unsharp Mask", imagem_sharp)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 7. TESTANDO DIFERENTES INTENSIDADES
# ============================================================

quantidade_05 = 0.5
quantidade_10 = 1.0
quantidade_20 = 2.0


sharp_05 = imagem_float + quantidade_05 * mascara
sharp_05 = np.clip(sharp_05, 0, 255)
sharp_05 = sharp_05.astype(np.uint8)


sharp_10 = imagem_float + quantidade_10 * mascara
sharp_10 = np.clip(sharp_10, 0, 255)
sharp_10 = sharp_10.astype(np.uint8)


sharp_20 = imagem_float + quantidade_20 * mascara
sharp_20 = np.clip(sharp_20, 0, 255)
sharp_20 = sharp_20.astype(np.uint8)


print("\n========== DIFERENTES INTENSIDADES ==========")
print("Quantidade 0.5 -> nitidez suave")
print("Quantidade 1.0 -> nitidez moderada")
print("Quantidade 2.0 -> nitidez forte")

cv2.imshow("Original", imagem)
cv2.imshow("Unsharp 0.5", sharp_05)
cv2.imshow("Unsharp 1.0", sharp_10)
cv2.imshow("Unsharp 2.0", sharp_20)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 8. ALTERANDO O SIGMA DO GAUSSIAN BLUR
# ============================================================


for sigma in [0.5, 1.0, 2.0, 4.0]:

    blur = cv2.GaussianBlur(
        imagem,
        (0, 0),
        sigmaX=sigma
    )

    mascara_sigma = (
        imagem.astype(np.float32)
        - blur.astype(np.float32)
    )

    resultado = (
        imagem.astype(np.float32)
        + mascara_sigma
    )

    resultado = np.clip(
        resultado,
        0,
        255
    ).astype(np.uint8)

    print("\nSigma:", sigma)
    print("Min:", resultado.min())
    print("Max:", resultado.max())
    print("Media:", resultado.mean())

    cv2.imshow(
        f"Unsharp sigma = {sigma}",
        resultado
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# 9. IMPLEMENTACAO MAIS DIRETA COM addWeighted
# ============================================================
#
# A formula:
#
# original + amount * (original - blur)
#
# pode ser reorganizada como:
#
# (1 + amount) * original - amount * blur
#
# O OpenCV permite fazer isso usando addWeighted().
#
# ============================================================

amount = 1.0
sigma = 2.0

blur = cv2.GaussianBlur(
    imagem,
    (0, 0),
    sigmaX=sigma
)

unsharp_opencv = cv2.addWeighted(
    imagem,
    1.0 + amount,
    blur,
    -amount,
    0
)

print("\n========== UNSHARP COM addWeighted ==========")
print("Amount:", amount)
print("Sigma:", sigma)

cv2.imshow("Original", imagem)
cv2.imshow("Blur", blur)
cv2.imshow("Unsharp addWeighted", unsharp_opencv)

cv2.waitKey(0)
cv2.destroyAllWindows()

