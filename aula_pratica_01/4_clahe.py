import cv2

# pip install opencv-python

# ============================================================
# CORRECAO DE CONTRASTE E ILUMINACAO COM CLAHE
# ============================================================


# ============================================================
# 1. CARREGA A IMAGEM
# ============================================================

imagem = cv2.imread("input4.png")


cv2.imshow("Original", imagem)
cv2.waitKey(0)
#cv2.destroyAllWindows()


# ============================================================
# 2. CONVERTE BGR -> LAB
# ============================================================
#
# L = luminosidade
# A = informacao de cor
# B = informacao de cor
#

lab = cv2.cvtColor(
    imagem,
    cv2.COLOR_BGR2LAB
)


L, A, B = cv2.split(lab)


print("\n========== CANAL DE LUMINOSIDADE ==========")
print("Min:", L.min())
print("Max:", L.max())
print("Media:", L.mean())


cv2.imshow("Luminosidade Original", L)
cv2.waitKey(0)
#cv2.destroyAllWindows()


# ============================================================
# 3. CRIANDO O CLAHE
# ============================================================
#
# CLAHE:
#
# Contrast Limited Adaptive Histogram Equalization
#
#
# Diferente da equalizacao global,
# ele trabalha em pequenas regioes da imagem.
#
#
# clipLimit:
# limita quanto o contraste pode ser aumentado.
#
# tileGridSize:
# divide a imagem em pequenas regioes.
#
# ============================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


# ============================================================
# 4. APLICA O CLAHE
# ============================================================

L_corrigido = clahe.apply(L)


print("\n========== CLAHE ==========")
print("Antes:")
print("Min:", L.min())
print("Max:", L.max())
print("Media:", L.mean())

print("\nDepois:")
print("Min:", L_corrigido.min())
print("Max:", L_corrigido.max())
print("Media:", L_corrigido.mean())


cv2.imshow("Luminosidade Original", L)
cv2.imshow("Luminosidade com CLAHE", L_corrigido)

cv2.waitKey(0)
#cv2.destroyAllWindows()


# ============================================================
# 5. RECONSTROI A IMAGEM
# ============================================================

lab_corrigido = cv2.merge(
    (
        L_corrigido,
        A,
        B
    )
)


imagem_corrigida = cv2.cvtColor(
    lab_corrigido,
    cv2.COLOR_LAB2BGR
)


# ============================================================
# 6. COMPARACAO
# ============================================================

cv2.imshow(
    "Original",
    imagem
)

cv2.imshow(
    "Contraste Corrigido - CLAHE",
    imagem_corrigida
)

cv2.waitKey(0)
#cv2.destroyAllWindows()


# ============================================================
# 7. OPCIONAL - PEQUENO SHARPENING
# ============================================================
#
# Depois da correcao de contraste,
# podemos aumentar levemente a nitidez.
#
# ============================================================

imagem_blur = cv2.GaussianBlur(
    imagem_corrigida,
    (0, 0),
    1.0
)


imagem_final = cv2.addWeighted(
    imagem_corrigida,
    1.5,
    imagem_blur,
    -0.5,
    0
)


cv2.imshow(
    "Original",
    imagem
)

cv2.imshow(
    "CLAHE",
    imagem_corrigida
)

cv2.imshow(
    "CLAHE + Sharpening",
    imagem_final
)

cv2.waitKey(0)
cv2.destroyAllWindows()
