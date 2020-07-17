import numpy as np
import cv2
import imutils
import pyzbar.pyzbar as pyzbar
import sys
import criaCSV as csv
import folhaFinal as ff

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


def carregaNumerosSamples():
    #carrega as imagens dos números que vão servir de base para identificar os nr dos alunos
    #e guarda dentro de um array
    NUM_SAMPLES1 = []
    try:
        for i in range(10):
            numeros = cv2.imread("numeros/num" + str(i) + ".JPG")
            numGray = cv2.cvtColor(numeros, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(numGray, 150, 255, cv2.THRESH_BINARY_INV)
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            (x, y, w, h) = cv2.boundingRect(contours[0])
            roi = numGray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (60, 70))
            NUM_SAMPLES1.append(roi)
    except:
        print("Erro ao carregar os numeros de sample \n")
    return (NUM_SAMPLES1)

def carregaImagem():
    #carrega a iamgem da folha de presenças com o formato RGB
    if len(sys.argv) > 1:
        img = cv2.imread(str(sys.argv[1]))
        if img is not None:
            #aplica um redimensionamento à imagem.
            img = cv2.resize(img, (IMGX, IMGY))
        else:
            quit("erro ao carregar folha de presença")
    else:
        quit("erro ao carregar folha de presença")
    return img


def processaCodigoBarras(img):
    try:
        decodeCB = pyzbar.decode(img)
    except:
        quit("Erro ao tentar ler o Código de Barras, Folha ilegível")

    #retira o último digito do código de barras por não ser necessário.
    if len(decodeCB) > 0:
        codigo = decodeCB[0].data
        codigo = int(codigo)
        codigo = int(codigo / 10)
    else:
        quit("Erro ao tentar ler o Código de Barras, Folha ilegível")

    return codigo


def corrigeAlinhamento(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY_INV)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sort_contours(contours, method="top-to-bottom")[0]

    if len(contours) > 0:
        todosContornos = []
        for c in contours:
            perimetro = cv2.arcLength(c, True)
            area = cv2.contourArea(c)
            if (perimetro > 1000 and area > 100000):
                (x, y, w, h) = cv2.boundingRect(c)
                roi = img[y:y + h, x:x + w]
                todosContornos.append(roi)

        cabecalho = todosContornos[0]

        grayCabecalho = cv2.cvtColor(cabecalho, cv2.COLOR_BGR2GRAY)
        grayCabecalho = np.float32(grayCabecalho)
        corners = cv2.goodFeaturesToTrack(grayCabecalho, 500, 0.02, 20)
        corners = np.int0(corners)
        if len(corners) > 0:
            leftCorners = []
            rightCorners = []
            larguraCabecalho = cabecalho.shape[1]

            for corner in corners:
                x, y = corner.ravel()
                if x < 50:
                    leftCorners.append(y)
                if x > (larguraCabecalho - 50):
                    rightCorners.append(y)
            leftCorners.sort()
            rightCorners.sort()
            leftCorner = leftCorners[0]
            rightCorner = rightCorners[0]
            correcao = 0
            if leftCorner > rightCorner:
                correcao = -leftCorner
            if rightCorner > leftCorner:
                correcao = rightCorner

            correcao = correcao * 0.033
            print("correcao ->" + str(correcao))
            img = imutils.rotate(img, correcao)
            img = img[int(IMGY * 0.05):IMGY - int(IMGY * 0.05), int(IMGX * 0.02):IMGX - int(IMGX * 0.02)]

    else:
        quit("Folha ilegível, falha no alinhamento")
    return img


def filtroDeLinhas(img):
    #passa a imagem para uma escala de cinzento e de seguida converte para o inverso.
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    notGray = cv2.bitwise_not(imgray)
    #aplica um threshold à imagem
    adpThresh = cv2.adaptiveThreshold(notGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, -6)

    horizontal = np.copy(adpThresh)
    vertical = np.copy(adpThresh)
    #cria um kernell rectangular e de seguida aplica a op. morfológica OPEN à imagem (OPEN = erosão seuida de uma dilatação) para retirar as linhas horizontais
    kernelHorizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, kernelHorizontal)

    # repete o processo, mas desta vez com um kernell que permite retirar apena as linhas verticais
    kernelVeritcal = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, kernelVeritcal)

    #junta as duas imagens numa só
    VerticalHorizontal = cv2.addWeighted(vertical, 1, horizontal, 1, 0)

    return horizontal, VerticalHorizontal


def encontraTabelasAlunos(img, linhas):
    #Procura todos os contornos externos da imagem com as linhas
    contours = cv2.findContours(linhas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sort_contours(contours, method="top-to-bottom")[0]

    if len(contours) > 0:
        todosContornos = []
        todosContornosLinhas = []
        roiAlunos = []
        roiAlunosLinhas = []
        for c in contours:
            perimetro = cv2.arcLength(c, True)
            area = cv2.contourArea(c)
            if (perimetro > 1000 and area > 50000):
                (x, y, w, h) = cv2.boundingRect(c)
                roi = img[y:y + h, x:x + w]
                todosContornos.append(roi)
                todosContornosLinhas.append((x, y, w, h))

        if len(todosContornos) > 0:
            todosContornosLinhas.pop(0)
            todosContornos.pop(0)
            roiAlunosLinhas = todosContornosLinhas
            roiAlunos = todosContornos

            if len(roiAlunos) > 1:
                (x1,_,_,_) = todosContornosLinhas[0]
                (x2, _, _, _) = todosContornosLinhas[1]
                if (x1 > x2):
                    temp = roiAlunos[0]
                    roiAlunos[0] = roiAlunos[1]
                    roiAlunos[1] = temp
                    temp = roiAlunosLinhas[0]
                    roiAlunosLinhas[0] = roiAlunosLinhas[1]
                    roiAlunosLinhas[1] = temp

        if (roiAlunos[0].shape[1] > IMGX * 0.65):
            largura = roiAlunos[0].shape[1];
            altura = roiAlunos[0].shape[0]
            roi1 = roiAlunos[0][0:altura, 0:int(largura / 2)]
            roi2 = roiAlunos[0][0:altura, int(largura / 2):largura]
            roiAlunos = []
            roiAlunos.append(roi1)
            roiAlunos.append(roi2)

            (x, y, w, h) = roiAlunosLinhas[0]
            roiLinhas1 = (x, y, int(w / 2), h)
            roiLinhas2 = (int(w / 2), y, w, h)
            roiAlunosLinhas = []
            roiAlunosLinhas.append(roiLinhas1)
            roiAlunosLinhas.append(roiLinhas2)
    else:
        quit("Folha ilegível, não foi possivel encontrar nenhuma tabela de alunos")

    return roiAlunos, roiAlunosLinhas


def encontraNumeroAluno(imgAluno):
    imgAlunoGray = cv2.cvtColor(imgAluno, cv2.COLOR_BGR2GRAY)
    notGray = cv2.bitwise_not(imgAlunoGray)

    #Aplica um close ao nr de aluno
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(notGray, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(closed, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    rectKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel2)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    max = cv2.contourArea(contours[0])
    b = contours[0]
    for c in contours:
        area = cv2.contourArea(c)
        if area > max:
            max = area
            b = c

    (x, y, w, h) = cv2.boundingRect(b)
    if (x != 0 and y != 0):
        roi = imgAluno[y: y + h, x: x + w]
    else:
        roi = imgAluno[y: y + h + 2, x: x + w + 3]

    return roi


def identificaNumeros(roi):
    roi = cv2.resize(roi, (60, 70))
    maior = 0
    num = 0
    for i in range(len(NUM_SAMPLES)):
        result = cv2.matchTemplate(roi, NUM_SAMPLES[i], cv2.TM_CCOEFF)
        (_, max_val, _, _) = cv2.minMaxLoc(result)
        if (max_val > maior):
            maior = max_val
            num = i

    return num


def confirmaAssinatura(assinatura):
    assinaturaGrey = cv2.cvtColor(assinatura, cv2.COLOR_BGR2GRAY)
    notGray = cv2.bitwise_not(assinaturaGrey)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))

    closed = cv2.morphologyEx(notGray, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(closed, 30, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (10, 5))
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    maxW = 0
    coord = None
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > maxW:
            coord = (x, y, w, h)
            maxW = w

    return coord


def verificaAssinatura(assinatura):
    assinaturaGrey = cv2.cvtColor(assinatura, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(assinaturaGrey, 200, 255, cv2.THRESH_BINARY_INV)

    assinado = False
    incerto = False
    percentagemAssinado = (cv2.countNonZero(thresh) * 100) / thresh.size
    if percentagemAssinado > 10:
        assinado = True
    if percentagemAssinado <= 10 and percentagemAssinado >= 3:
        (x, y, w, h) = confirmaAssinatura(assinatura)
        if w > assinatura.shape[1] * 0.20 and h > assinatura.shape[0] * 0.4:
            assinado = True
        else:
            incerto = True

    cv2.waitKey(0)
    return assinado, incerto


def extraiLinhasAlunosIndividual(linhasHorizontais, roiAlunosLinhas):
    (x, y, w, h) = roiAlunosLinhas
    roi = linhasHorizontais[y:y + h, x:x + w]
    linhasAlunos = []
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 30, minLineLength=400, maxLineGap=250)

    if len(lines) > 0:
        for line in lines:
            (x1, y1, x2, y2) = line[0]
            linhasAlunos.append((x1, y1, x2, y2))

        linhasAlunos = sorted(linhasAlunos, key=lambda y: y[1])

        linhasAlunosFinal = []
        (_, y, _, _) = linhasAlunos[0]
        for linha in linhasAlunos:
            (x1, y1, x2, y2) = linha
            if y1 - y > 10:
                linhasAlunosFinal.append(linha)
                y = y1
    else:
        quit("Folha ilegível, falha na divisão dos alunos")
    return linhasAlunosFinal


def processaAlunos():
    todosAlunos = []
    alunosPresentes = []
    imgFinal = img.copy()

    linhasHorizontais, linhas = filtroDeLinhas(img)
    roiAlunos, roiAlunosLinhas = encontraTabelasAlunos(img, linhas)

    folhaInvalida = 0
    contador = 1;
    #percorre as tabelas dos alunos encontradas
    for r in range(len(roiAlunos)):
        linhasAlunos = extraiLinhasAlunosIndividual(linhasHorizontais, roiAlunosLinhas[r])
        larg = roiAlunos[r].shape[1]

        #percorre as linhas que definem cada espaço do aluno
        for i in range(len(linhasAlunos) - 1):
            (x1, Yi, x2, y2) = linhasAlunos[i]
            (x1, y1, x2, Yf) = linhasAlunos[i + 1]
            altura = Yf - Yi
            if altura < int(IMGY * 0.005):
                continue
            #extrai um aluno consuante as linhas
            Aluno = roiAlunos[r][Yi:Yf, 5:larg - 10]
            #extrai número de aluno consuante coordenadas fixas.
            nrAluno = Aluno[int(round(altura * 0.1)):int(round(altura * 0.93)),
                      int(round(larg * 0.1)):int(round(larg * 0.29))]
            #aproximação exata ao local do número de aluno
            nrAluno = encontraNumeroAluno(nrAluno)
            nrAlunoGray = cv2.cvtColor(nrAluno, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(nrAlunoGray, 130, 255, cv2.THRESH_BINARY_INV)
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            if len(contours) > 0:
                contours = sort_contours(contours, method="left-to-right")[0]
            larguraNumero = nrAluno.shape[1] / 10
            listaNr = []
            #percorre os números encontrados
            for c in contours:
                (x, y, w, h) = cv2.boundingRect(c)
                roi = nrAlunoGray[y:y + h, x:x + w]
                roi = cv2.addWeighted(roi, 1, roi, 0, -35)
                if cv2.arcLength(c, True) > 25:
                    if (w > larguraNumero + (larguraNumero * 0.5)):
                        primeiroNumero = nrAlunoGray[y:y + h, x:int(x + (w / 2))]
                        primeiroNumero = cv2.addWeighted(primeiroNumero, 1, primeiroNumero, 0, -35)
                        segundoNumero = nrAlunoGray[y:y + h, int(x + (w / 2)):x + w]
                        segundoNumero = cv2.addWeighted(segundoNumero, 1, segundoNumero, 0, -35)
                        listaNr.append(identificaNumeros(primeiroNumero))
                        listaNr.append(identificaNumeros(segundoNumero))
                    else:
                        listaNr.append(identificaNumeros(roi))
            n = ""
            for j in listaNr:
                n = n + str(j)
            if len(n) == 10:
                try:
                    numero_Aluno = int(n)
                except ValueError:
                    print("Não foi possível ler o número do aluno corretamente")
                    numero_Aluno = 0

                assinatura = Aluno[int(round(altura * 0.25)):int(round(altura * 0.94)),
                             int(round(larg * 0.74)):int(round(larg * 0.97))]
                assinado, incerto = verificaAssinatura(assinatura)
                todosAlunos.append(numero_Aluno)
                if assinado:
                    ff.folhaFinal(imgFinal, roiAlunosLinhas[r], Yi, imgCerto)
                    alunosPresentes.append(numero_Aluno)

                print(
                    str(contador) + " - " + str(
                        numero_Aluno) + " = Assinatura incerta, verificar") if incerto else print(
                    str(contador) + " - " + str(numero_Aluno) + " = PRESENTE") if assinado else print(
                    str(contador) + " - " + str(numero_Aluno))
                contador += 1

            else:
                numero_Aluno = int(n)
                print(str(contador) + " - Não foi possível ler o número do aluno corretamente -> " + str(numero_Aluno))
                folhaInvalida += 1
                contador += 1
    if folhaInvalida > len(todosAlunos):
        quit("folha com problemas")

    #imgFinal = cv2.resize(imgFinal, (1000, 1200))
    #cv2.imwrite('C:/Users/renato/Desktop/Relatorio Projeto/imagens/folhafinal.jpg',imgFinal)
    #cv2.imshow("asdas", imgFinal)
    return todosAlunos, alunosPresentes


if __name__ == '__main__':

    img = carregaImagem()
    NUM_SAMPLES = carregaNumerosSamples()
    imgCerto = ff.carregaImagemCerto()
    codigoAula = processaCodigoBarras(img)
    img = corrigeAlinhamento(img)
    todosAlunos, alunosPresentes = processaAlunos()
    csv.criaCSVFile(alunosPresentes, codigoAula)

#cv2.waitKey(0)
cv2.destroyAllWindows()
