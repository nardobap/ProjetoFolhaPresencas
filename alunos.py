import cv2
import utils
import imutils
import numpy as np
import folhaPresenca
import pytesseract
import folhaFinal as ff
from joblib import load, dump
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


IMGX = utils.IMGX
IMGY = utils.IMGY


def encontraNumeroAluno(imgAluno):
    #encontra especificamente onde o número de aluno se encontra.
    src = cv2.cvtColor(imgAluno, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(src, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    dst = cv2.GaussianBlur(thresh,(5,5),cv2.BORDER_DEFAULT)
    #plt.figure()
    notGray = cv2.bitwise_not(dst)
    #plt.imshow(notGray, cmap="gray")
    #plt.show()
    #print("THIS")

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



def confirmaAssinatura(assinatura):
    #Confirma se realmente existe uma assinatura, através do seu tamanho
    assinaturaGrey = cv2.cvtColor(assinatura, cv2.COLOR_BGR2GRAY)
    notGray = cv2.bitwise_not(assinaturaGrey)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(notGray, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(closed, 0, 255,
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
    assinado = False
    #dar tamanho fixo
    img_sized = cv2.resize(assinatura, (215,30))
    #converter para um canal BW
    gray = cv2.cvtColor(img_sized, cv2.COLOR_BGR2GRAY)
    #blur - reduzir ruído e contornos
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    #truncar pixels abaixo de valor - redução de ruido
    ret, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_TRUNC)
    
    #TODO:
    #COLOCAR AQUI CODIGO
    # percorre a imagem da assinatura e soma as linhas e as colunas
    sum_of_rows = np.sum(thresh, axis = 1)
    sum_of_columns = np.sum(thresh, axis = 0)
    
    #concatena as somas e transpoe para vector
    vector_img = np.concatenate((sum_of_rows, sum_of_columns))
    signature = np.transpose(vector_img)
    
    # standardiza os valores das somas
    X_test = StandardScaler().fit_transform(signature)
    print(X_test)
    
    #carrega o modelo
    mlp = load("modelsignature.joblib")
    
    #devolve a label atribuida
    assinado = mlp.predict(X_test)
    
    return assinado



def extraiLinhasAlunosIndividual(linhasHorizontais, roiAlunosLinhas):
    #extrai linhas horizontais das tabelas para ser possível individualizar cada aluno
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


def processaAlunos(img, count_alunos, out_ilegivel, out_incerto, out_presente, out_ausente, out_erro_num, out_problemas, codigoAula):
    todosAlunos = []
    alunosPresentes = []
    imgFinal = img.copy()
    imgCerto = ff.carregaImagemCerto()

    linhasHorizontais, linhas = folhaPresenca.filtroDeLinhas(img)
    roiAlunos, roiAlunosLinhas = folhaPresenca.encontraTabelasAlunos(img, linhas)

    folhaInvalida = 0
    contador = 1
    
    
    # tesseract configuration
    custom_config = r'--oem 3 --psm 6 outputbase digits'

    #percorre as tabelas dos alunos encontradas
    for r in range(len(roiAlunos)):
        linhasAlunos = extraiLinhasAlunosIndividual(linhasHorizontais, roiAlunosLinhas[r])
        larg = roiAlunos[r].shape[1]

    
        for i in range(len(linhasAlunos) - 1):
            (x1, Yi, x2, y2) = linhasAlunos[i]
            (x1, y1, x2, Yf) = linhasAlunos[i + 1]
            altura = Yf - Yi
            if altura < int(IMGY * 0.005):
                continue
            #extrai um aluno consoante as linhas
            Aluno = roiAlunos[r][Yi:Yf, 5:larg - 10]
            #extrai número de aluno consoante coordenadas fixas.
            nrAluno = Aluno[int(round(altura * 0.1)):int(round(altura * 0.93)),
                int(round(larg * 0.1)):int(round(larg * 0.29))]
            
            
            #aproximação exata ao local do número de aluno
            nrAluno = encontraNumeroAluno(nrAluno)
            nrAlunoGray = cv2.cvtColor(nrAluno, cv2.COLOR_BGR2GRAY)
            
            # utiliza o pytesseract para ler o id do aluno
            n_out = pytesseract.image_to_string(nrAlunoGray, config=custom_config)
            
            #remove espaco na string obtida por ocr
            #remove pontos finais na string obtida por ocr
            n_clean = n_out.replace(' ','').replace('.','')
            
            # pega no output do tesseract e retorna uma list com digitos apenas
            # a list e transformada em string e tem de ter tamanho 10
            n = ",".join([str(s) for s in n_clean.split() if s.isdigit()])

            # se tiver tam 10 entao faz cast para inteiro    
            if len(n) == 10:
                try:
                    numero_Aluno = int(n)
                except ValueError:
                    print("*****************************************************************")
                    print("Não foi possível converter o número do aluno corretamente")
                    print("*****************************************************************")
                    out_ilegivel += 1
                    numero_Aluno = 0
    
                assinatura = Aluno[int(round(altura * 0.25)):int(round(altura * 0.94)),
                                     int(round(larg * 0.74)):int(round(larg * 0.97))]
                assinado = verificaAssinatura(assinatura)

                todosAlunos.append(numero_Aluno)
                if assinado:
                    ff.folhaFinal(imgFinal, roiAlunosLinhas[r], Yi, imgCerto)
                    alunosPresentes.append(numero_Aluno)
                elif assinado:
                    print(str(contador) + " - " + str(numero_Aluno) + " = PRESENTE")
                    out_presente += 1
                else:
                    print(str(contador) + " - " + str(numero_Aluno))
                    out_ausente += 1
                contador += 1

            else:
                try:
                    numero_Aluno = int(n)
                    print(str(contador) + " - Não foi possível ler o número do aluno corretamente -> " + str(numero_Aluno))
                except ValueError:
                    print(str(contador) + " - Erro na leitura do número -> NULL ")
                out_erro_num += 1
                contador += 1
        if folhaInvalida > len(todosAlunos):
            out_problemas += 1
            quit("folha com problemas")

    imgFinal = cv2.resize(imgFinal, (1000, 1200))
    #cv2.imshow("imagem Final", imgFinal)
    count_alunos += len(todosAlunos)
    return todosAlunos, alunosPresentes, count_alunos, out_ilegivel, out_incerto, out_presente, out_ausente, out_erro_num, out_problemas

