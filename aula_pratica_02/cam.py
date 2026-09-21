import cv2
import numpy as np


def white_balance_gray_world(imagem):

    # Converte para float para evitar overflow
    img = imagem.astype(np.float32)

    # OpenCV trabalha em BGR
    b, g, r = cv2.split(img)

    media_b = np.mean(b)
    media_g = np.mean(g)
    media_r = np.mean(r)

    # Media geral
    media_cinza = (
        media_b +
        media_g +
        media_r
    ) / 3.0

    # Fatores de correcao
    b *= media_cinza / (media_b + 1e-6)
    g *= media_cinza / (media_g + 1e-6)
    r *= media_cinza / (media_r + 1e-6)

    # Reconstroi imagem
    resultado = cv2.merge((b, g, r))

    # Garante intervalo 0-255
    resultado = np.clip(
        resultado,
        0,
        255
    ).astype(np.uint8)

    return resultado


def filtro_quente(imagem):

    img = imagem.astype(np.float32)

    b, g, r = cv2.split(img)

    # Aumenta vermelho
    r *= 1.15

    # Diminui levemente azul
    b *= 0.90

    resultado = cv2.merge((b, g, r))

    return np.clip(
        resultado,
        0,
        255
    ).astype(np.uint8)


def filtro_frio(imagem):

    img = imagem.astype(np.float32)

    b, g, r = cv2.split(img)

    # Aumenta azul
    b *= 1.15

    # Diminui vermelho
    r *= 0.90

    resultado = cv2.merge((b, g, r))

    return np.clip(
        resultado,
        0,
        255
    ).astype(np.uint8)


def filtro_sepia(imagem):

    kernel_sepia = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ])

    resultado = cv2.transform(
        imagem,
        kernel_sepia
    )

    return np.clip(
        resultado,
        0,
        255
    ).astype(np.uint8)


def filtro_cinza(imagem):

    cinza = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.cvtColor(
        cinza,
        cv2.COLOR_GRAY2BGR
    )


def adicionar_oculos(imagem, face):

    x, y, w, h = face

    # Altura aproximada dos olhos
    y_oculos = y + int(h * 0.40)

    # Tamanho das lentes
    largura_lente = int(w * 0.30)
    altura_lente = int(h * 0.16)

    # Centros aproximados
    centro_esquerdo = (
        x + int(w * 0.30),
        y_oculos
    )

    centro_direito = (
        x + int(w * 0.70),
        y_oculos
    )

    # Lente esquerda
    cv2.ellipse(
        imagem,
        centro_esquerdo,
        (largura_lente // 2, altura_lente),
        0,
        0,
        360,
        (0, 0, 0),
        -1
    )

    # Lente direita
    cv2.ellipse(
        imagem,
        centro_direito,
        (largura_lente // 2, altura_lente),
        0,
        0,
        360,
        (0, 0, 0),
        -1
    )

    # Ponte dos oculos
    cv2.line(
        imagem,
        (
            centro_esquerdo[0] + largura_lente // 2,
            centro_esquerdo[1]
        ),
        (
            centro_direito[0] - largura_lente // 2,
            centro_direito[1]
        ),
        (0, 0, 0),
        5
    )


caminho_detector = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

detector_faces = cv2.CascadeClassifier(
    caminho_detector
)

if detector_faces.empty():
    print("Erro ao carregar detector de faces.")
    exit()


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro ao abrir webcam.")
    exit()


print("0 = apenas balanço de branco")
print("1 = filtro quente")
print("2 = filtro frio")
print("3 = sepia")
print("4 = preto e branco")
print("")
print("A = ativar/desativar acessorio")
print("W = ativar/desativar white balance")
print("Q = sair")
print("")


filtro_atual = 0

usar_acessorio = False

usar_white_balance = False

while True:

    sucesso, frame = camera.read()

    if not sucesso:
        print("Erro ao capturar frame.")
        break

    original = frame.copy()


    cinza = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = detector_faces.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )


    processada = frame.copy()


    if usar_white_balance:

        processada = white_balance_gray_world(
            processada
        )


    if filtro_atual == 1:

        processada = filtro_quente(
            processada
        )

        nome_filtro = "QUENTE"

    elif filtro_atual == 2:

        processada = filtro_frio(
            processada
        )

        nome_filtro = "FRIO"

    elif filtro_atual == 3:

        processada = filtro_sepia(
            processada
        )

        nome_filtro = "SEPIA"

    elif filtro_atual == 4:

        processada = filtro_cinza(
            processada
        )

        nome_filtro = "PRETO E BRANCO"

    else:

        nome_filtro = "NORMAL"


    for (x, y, w, h) in faces:

        cv2.rectangle(
            original,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


    if usar_acessorio:

        for face in faces:

            adicionar_oculos(
                processada,
                face
            )


    cv2.putText(
        original,
        "ORIGINAL",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    texto_processado = "PROCESSADO - " + nome_filtro

    cv2.putText(
        processada,
        texto_processado,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        processada,
        f"Faces: {len(faces)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    comparacao = np.hstack(
        (
            original,
            processada
        )
    )


    cv2.imshow(
        "Fotografia Computacional - Original | Processado",
        comparacao
    )


    tecla = cv2.waitKey(1) & 0xFF


    if tecla == ord("0"):

        filtro_atual = 0

        print("Filtro: Normal")


    elif tecla == ord("1"):

        filtro_atual = 1

        print("Filtro: Quente")


    elif tecla == ord("2"):

        filtro_atual = 2

        print("Filtro: Frio")


    elif tecla == ord("3"):

        filtro_atual = 3

        print("Filtro: Sepia")


    elif tecla == ord("4"):

        filtro_atual = 4

        print("Filtro: Preto e Branco")


    elif tecla == ord("a"):

        usar_acessorio = not usar_acessorio

        print(
            "Acessorio:",
            usar_acessorio
        )


    elif tecla == ord("w"):

        usar_white_balance = not usar_white_balance

        print(
            "White Balance:",
            usar_white_balance
        )


    elif tecla == ord("q"):

        break



camera.release()

cv2.destroyAllWindows()