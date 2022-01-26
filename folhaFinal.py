import cv2

def carregaImagemCerto():
    #carrega imagem do certo com o formato RGB para ser inserido na imagem final
    certoImg = cv2.imread("certo.png")
    if certoImg is not None:
        certoImg = cv2.resize(certoImg, (30, 30))
    else:
        quit("erro ao carregar imagem de certo")
    return certoImg

def folhaFinal(imgFinal, coord, Yi, certoImg):
    #adiciona os certos à imagem final
    (x, y, w, h) = coord
    x = x + w - 30
    imgFinal[y + Yi:y + Yi + 30, x:x + 30] = certoImg

    return imgFinal
