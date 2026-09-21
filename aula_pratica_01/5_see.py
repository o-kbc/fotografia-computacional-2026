import cv2


# ============================================================
# 1. CARREGANDO O DETECTOR DE FACES
# ============================================================
det = cv2.CascadeClassifier()
detector_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ============================================================
# 2. ABRINDO A WEBCAM
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro: nao foi possivel abrir a webcam.")
    exit()

print("\nWebcam iniciada.")
print("Pressione Q para sair.")


# ============================================================
# 3. LOOP PRINCIPAL
# ============================================================

while True:

    # Le um frame da webcam
    sucesso, frame = camera.read()

    if not sucesso:
        print("Erro ao capturar imagem.")
        break


    # ========================================================
    # 4. CONVERSAO PARA ESCALA DE CINZA
    # ========================================================
    #
    # O detector Haar trabalha normalmente sobre
    # imagens em escala de cinza.
    #
    # ========================================================

    cinza = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # 5. DETECCAO DAS FACES
    # ========================================================
    #
    # detectMultiScale retorna:
    #
    # x = posicao horizontal
    # y = posicao vertical
    # w = largura da face
    # h = altura da face
    #
    # ========================================================

    faces = detector_faces.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )


    # ========================================================
    # 6. MOSTRA QUANTAS FACES FORAM DETECTADAS
    # ========================================================

    quantidade_faces = len(faces)

    print(
        f"Faces detectadas: {quantidade_faces}",
        end="\r"
    )


    # ========================================================
    # 7. DESENHA UM RETANGULO EM CADA FACE
    # ========================================================

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Mostra a posicao da face no terminal
        print(
            f"\nFace -> x: {x}, y: {y}, largura: {w}, altura: {h}"
        )


    # ========================================================
    # 8. MOSTRA INFORMACOES NA TELA
    # ========================================================

    cv2.putText(
        frame,
        f"Faces: {quantidade_faces}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # ========================================================
    # 9. MOSTRA O RESULTADO
    # ========================================================

    cv2.imshow(
        "Detect",
        frame
    )


    # ========================================================
    # 10. PRESSIONE Q PARA SAIR
    # ========================================================

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):
        break


# ============================================================
# 11. FINALIZACAO
# ============================================================

camera.release()

cv2.destroyAllWindows()
