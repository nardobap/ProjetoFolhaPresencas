import pdfparajpeg
import cv2
import criaCSV as csv
import folhaPresenca
import CB
import alunos
import utils
import sys
import shutil
import os
from assinaturas_validar import load_training_data, classifier_training


# MULTIPLE PHOTOS

if __name__ == '__main__':
    
    #Array to store all filepaths  
    arfiles = []
    
    #aux variables        
    out_codigo_barras = 0
    out_leitura_cabecalho = 0
    out_alinhamento = 0
    count_alunos = 0
    out_ilegivel = 0
    out_incerto = 0
    out_presente = 0
    out_ausente = 0
    out_erro_num = 0
    out_problemas = 0
    
    #log array
    folhas_erros = []
    
    # sem argumento significa que corre do mesmo directorio
    if len(sys.argv) == 1:
        arfiles = pdfparajpeg.find_jpg(".")
    # com argumento significa que corre do directorio fornecido    
    elif len(sys.argv) == 2:
        arfiles = pdfparajpeg.find_jpg(str(sys.argv[1]))
    else:
        sys.exit("Chamada: $python main.py [dirname]")

    #verifica se existem imagens - sai se nao existirem
    if  len(arfiles) == 0:
        sys.exit("Nao foram encontradas imagens")
    else:
        print(f"Foram encontradas {len(arfiles)} imagens")
        
    
    # Classifier
    # loads training data
    X_train, y_train, scaler = load_training_data()
    # creates model
    mlp = classifier_training()
        
    
    i = 1
    for imgs in arfiles:
        print(f"\n{i} - Filedir:" + imgs + f" **** Faltam {len(arfiles) - i} folhas")
        
        img = folhaPresenca.carregaImagem(imgs)
        img, out_leitura_cabecalho, out_alinhamento = folhaPresenca.corrigeAlinhamento(img,out_leitura_cabecalho,out_alinhamento)
        codigoAula = CB.processaCodigoBarras(img, imgs)
        if codigoAula == -1:
            out_codigo_barras = out_codigo_barras + 1
            folhas_erros.append(imgs)
            i=i+1
            
            continue
        else:
            try:
                todosAlunos, alunosPresentes, count_alunos, out_ilegivel, out_incerto, out_presente, out_ausente, out_erro_num, out_problemas = alunos.processaAlunos(img, X_train, y_train, mlp, scaler, count_alunos, out_ilegivel, out_incerto, out_presente, out_ausente, out_erro_num, out_problemas, codigoAula)
                csv.criaCSVFile(alunosPresentes, codigoAula)
                i=i+1
            except:
                out_problemas += 1
                folhas_erros.append(imgs)
                i=i+1
                continue
            
            
print("\n\nERROS:\n")
print(f"\tFalha leitura do Codigo: {out_codigo_barras}\n")
print(f"\tFalha do cabecalho: {out_leitura_cabecalho}\n")
print(f"\tFalha no alinhamento: {out_alinhamento}\n")
print(f"NUMERO ALUNOS:{count_alunos}\n")
print(f"\tAluno presentes: {out_presente}\n")
print(f"\tAluno ausente: {out_ausente}\n")
print(f"\tIlegivel: {out_ilegivel}\n")
print(f"\tAssinatura incerta: {out_incerto}\n")
print(f"\tErro leitura num: {out_erro_num}\n")
print(f"\tFolha problemas {out_problemas}\n")
print("Lista de folhas com erros:\n")
print(folhas_erros)

cv2.waitKey(0)
cv2.destroyAllWindows()


# _FOR FUTURE USAGE____________________________________________________________________________________________
# SINGLE SHEET READING


# if __name__ == '__main__':

#     out_codigo_barras = 0
#     count_alunos = 0
#     out_ilegivel = 0
#     out_incerto = 0
#     out_presente = 0
#     out_ausente = 0
#     out_erro_num = 0
#     out_problemas = 0


#     img = cv2.imread("page309.jpg")
#     img = cv2.resize(img, (utils.IMGX, utils.IMGY))
#     img = folhaPresenca.corrigeAlinhamento(img)
#     codigoAula = CB.processaCodigoBarras(img)
#     if codigoAula == -1:
#         print("CODIGO BARRAS NAO LIDO")
#     else:
#         out_codigo_barras = out_codigo_barras +1
#     todosAlunos, alunosPresentes, count_alunos, out_ilegivel, out_incerto, out_presente, out_ausente, out_erro_num, out_problemas = alunos.processaAlunos(img, count_alunos, out_ilegivel, out_incerto, out_presente, out_ausente, out_erro_num, out_problemas)
#     csv.criaCSVFile(alunosPresentes, codigoAula)


# print(f"ERROS:\ncodigo barras: {out_codigo_barras}\n")
# print(f"NUMERO ALUNOS:{count_alunos}\n")
# print(f"ERROS:\nAluno presente: {out_presente}\n")
# print(f"ERROS:\nAluno ausente: {out_ausente}\n")
# print(f"ERROS:\nIlegivel: {out_ilegivel}\n")
# print(f"ERROS:\nAssinatura incerta: {out_incerto}\n")
# print(f"ERROS:\nErro leitura num: {out_erro_num}\n")
# cv2.waitKey(0)
# cv2.destroyAllWindows()
