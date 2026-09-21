import cv2
import numpy as np

# pip install opencv-python

imagem = cv2.imread("input2.jpg", cv2.IMREAD_GRAYSCALE)

print("\n========== IMAGEM ORIGINAL ==========")
print("Shape:", imagem.shape)
print("Tipo:", imagem.dtype)
print("Valor minimo:", imagem.min())
print("Valor maximo:", imagem.max())
print("Media dos pixels:", imagem.mean())

cv2.imshow("Imagem Original", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 2. SMOOTHING - FILTRO DA MEDIA
# ============================================================
#
#
# 1/9 * [ 1  1  1
#         1  1  1
#         1  1  1 ]
#
# ============================================================

media_3x3 = cv2.blur(imagem, (3, 3))

print("\n========== FILTRO DA MEDIA 3x3 ==========")
print("Shape:", media_3x3.shape)
print("Valor minimo:", media_3x3.min())
print("Valor maximo:", media_3x3.max())
print("Media:", media_3x3.mean())

cv2.imshow("Original", imagem)
cv2.imshow("Filtro da Media 3x3", media_3x3)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 3. FILTRO DA MEDIA COM KERNEL MAIOR
# ============================================================

media_5x5 = cv2.blur(imagem, (5, 5))

print("\n========== FILTRO DA MEDIA 5x5 ==========")
print("Kernel: 5x5")
print("Media:", media_5x5.mean())

cv2.imshow("Original", imagem)
cv2.imshow("Media 3x3", media_3x3)
cv2.imshow("Media 5x5", media_5x5)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 4. COMPARACAO ENTRE DIFERENTES TAMANHOS DE KERNEL
# ============================================================

media_9x9 = cv2.blur(imagem, (9, 9))
media_15x15 = cv2.blur(imagem, (15, 15))

print("\n========== TAMANHO DO KERNEL ==========")
print("Kernel 3x3  -> pouca suavizacao")
print("Kernel 5x5  -> suavizacao moderada")
print("Kernel 9x9  -> maior suavizacao")
print("Kernel 15x15 -> forte suavizacao")

cv2.imshow("Original", imagem)
cv2.imshow("Media 3x3", media_3x3)
cv2.imshow("Media 5x5", media_5x5)
cv2.imshow("Media 9x9", media_9x9)
cv2.imshow("Media 15x15", media_15x15)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 5. SMOOTHING - FILTRO DA MEDIANA
# ============================================================


mediana_3x3 = cv2.medianBlur(imagem, 3)

print("\n========== FILTRO DA MEDIANA 3x3 ==========")
print("Shape:", mediana_3x3.shape)
print("Valor minimo:", mediana_3x3.min())
print("Valor maximo:", mediana_3x3.max())
print("Media:", mediana_3x3.mean())

cv2.imshow("Original", imagem)
cv2.imshow("Mediana 3x3", mediana_3x3)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 6. FILTRO DA MEDIANA 5x5
# ============================================================

mediana_5x5 = cv2.medianBlur(imagem, 5)

print("\n========== FILTRO DA MEDIANA 5x5 ==========")
print("Media:", mediana_5x5.mean())

cv2.imshow("Original", imagem)
cv2.imshow("Mediana 3x3", mediana_3x3)
cv2.imshow("Mediana 5x5", mediana_5x5)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 7. COMPARACAO: MEDIA X MEDIANA
# ============================================================

print("\n========== MEDIA X MEDIANA ==========")
print("Media:")
print("- calcula a media dos pixels da vizinhanca")
print("- tende a borrar bordas")
print("- reduz variacoes locais")

print("\nMediana:")
print("- utiliza o valor mediano da vizinhanca")
print("- preserva melhor algumas bordas")
print("- boa para ruido impulsivo")

cv2.imshow("Original", imagem)
cv2.imshow("Media 5x5", media_5x5)
cv2.imshow("Mediana 5x5", mediana_5x5)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 8. LAPLACIANO
# ============================================================


laplaciano = cv2.Laplacian(
    imagem,
    cv2.CV_64F
)

print("\n========== LAPLACIANO ==========")
print("Shape:", laplaciano.shape)
print("Tipo:", laplaciano.dtype)
print("Valor minimo:", laplaciano.min())
print("Valor maximo:", laplaciano.max())



# ============================================================
# 9. CONVERTENDO O LAPLACIANO PARA VISUALIZACAO
# ============================================================


laplaciano_abs = cv2.convertScaleAbs(laplaciano)

print("\n========== LAPLACIANO PARA VISUALIZACAO ==========")
print("Tipo antes:", laplaciano.dtype)
print("Tipo depois:", laplaciano_abs.dtype)

cv2.imshow("Original", imagem)
cv2.imshow("Laplaciano", laplaciano_abs)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 10. SHARPENING UTILIZANDO O LAPLACIANO
# ============================================================

imagem_float = imagem.astype(np.float64)

sharpening = imagem_float - laplaciano

# Limita os valores para o intervalo de uma imagem de 8 bits
sharpening = np.clip(sharpening, 0, 255)

# Converte novamente para uint8
sharpening = sharpening.astype(np.uint8)

print("\n========== SHARPENING ==========")
print("Shape:", sharpening.shape)
print("Tipo:", sharpening.dtype)
print("Valor minimo:", sharpening.min())
print("Valor maximo:", sharpening.max())

cv2.imshow("Original", imagem)
cv2.imshow("Laplaciano", laplaciano_abs)
cv2.imshow("Sharpening com Laplaciano", sharpening)

cv2.waitKey(0)
cv2.destroyAllWindows()

