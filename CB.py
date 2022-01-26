from pyzbar.pyzbar import decode
import utils
import cv2

# funcao para cortar a imagem no quadrante superior direito onde está o código
# de barras de forma a melhorar a leitura do cb

def recortacb(img):
    y1 = 0
    y2 = 1000
    x1 = 1000
    img_close = img[y1:y2, x1:utils.IMGX]
    return img_close

def processaCodigoBarras(img, pathimg):
    img_close = recortacb(img)
    pathimg = pathimg[10:]
    file_assinatura= str(pathimg) + ".jpg"
    cv2.imwrite("./erroscb/cb/"+str(file_assinatura), img_close)
    try:
        decodeCB = decode(img_close)
    except:
        print("Erro ao tentar ler o Código de Barras, Folha ilegível")
        return -1

    #retira o último digito do código de barras por não ser necessário.
    if len(decodeCB) > 0:
        codigo = decodeCB[0].data
        codigo = int(codigo)
        codigo = int(codigo / 10)
    else:
        print("Erro ao tentar ler o Código de Barras, Folha ilegível")
        return -1

    return codigo