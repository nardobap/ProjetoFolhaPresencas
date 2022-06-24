import cv2

IMGX = 2000
IMGY = 2500


def sort_contours(cnts, method):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def img_transformation(img):
    #dar tamanho fixo
    img_sized = cv2.resize(img, (215,30))
    #converter para um canal BW
    gray = cv2.cvtColor(img_sized, cv2.COLOR_BGR2GRAY)
    #blur - reduzir ruído e contornos
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    #truncar pixels abaixo de valor - redução de ruido
    ret, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_TRUNC)
    return thresh