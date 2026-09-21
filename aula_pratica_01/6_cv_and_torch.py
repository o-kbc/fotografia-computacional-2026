import cv2
import torch
import numpy as np

# pip install opencv-python torch

# ============================================================
# OPENCV + PYTORCH
# ============================================================

imagem = cv2.imread("input3.png")


cv2.imshow("Imagem Original", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 2. BGR -> RGB
# ============================================================


imagem_rgb = cv2.cvtColor(
    imagem,
    cv2.COLOR_BGR2RGB
)

print("\n========== CONVERSAO BGR -> RGB ==========")
print("Pixel BGR:", imagem[0, 0])
print("Pixel RGB:", imagem_rgb[0, 0])


# ============================================================
# 3. REDIMENSIONAMENTO
# ============================================================

imagem_resize = cv2.resize(
    imagem_rgb,
    (224, 224)
)

print("\n========== RESIZE ==========")
print("Shape:", imagem_resize.shape)


# Para visualizar no OpenCV,
# precisamos voltar de RGB para BGR

visualizacao = cv2.cvtColor(
    imagem_resize,
    cv2.COLOR_RGB2BGR
)

cv2.imshow("Imagem 224x224", visualizacao)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# 4. NORMALIZACAO
# ============================================================


imagem_normalizada = imagem_resize.astype(np.float32) / 255.0

print("\n========== NORMALIZACAO ==========")
print("Min:", imagem_normalizada.min())
print("Max:", imagem_normalizada.max())
print("Tipo:", imagem_normalizada.dtype)


# ============================================================
# 5. NUMPY -> PYTORCH
# ============================================================


tensor = torch.from_numpy(
    imagem_normalizada
)

print("\nTensor antes do permute:")
print(tensor.shape)


tensor = tensor.permute(2, 0, 1)

print("\nTensor depois do permute:")
print(tensor.shape)


# ============================================================
# 6. ADICIONANDO DIMENSAO DE BATCH
# ============================================================


tensor = tensor.unsqueeze(0)

print("\n========== TENSOR FINAL ==========")
print("Shape:", tensor.shape)
print("Tipo:", tensor.dtype)
