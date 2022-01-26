import pdfparajpeg
import cv2
import criaCSV as csv
import folhaPresenca
import CB
import alunos
import utils
import shutil


# MULTIPLE PHOTOS


if __name__ == '__main__':
    arfiles = []
    arfiles = pdfparajpeg.find_jpg("./resolucaovaria/400")

    print(f"O ARRAY TEM {len(arfiles)} IMAGENS!!!")
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
    

    i = 1
    for imgs in arfiles:
        print(f"ITERA {i} **** FALTAM {len(arfiles) - i} IMAGENS!!!")
        print(imgs)
        img = folhaPresenca.carregaImagem(imgs)
        img, out_leitura_cabecalho, out_alinhamento = folhaPresenca.corrigeAlinhamento(img,out_leitura_cabecalho,out_alinhamento)
        codigoAula = CB.processaCodigoBarras(img, imgs)
        if codigoAula == -1:
            out_codigo_barras = out_codigo_barras +1
            i=i+1
            shutil.move(imgs,"./erroscb")
            continue
        else:
            #NUM_SAMPLES = newalunos.carregaNumerosSamples()
            try:
                todosAlunos, alunosPresentes, count_alunos, out_ilegivel, out_incerto, out_presente, out_ausente, out_erro_num, out_problemas = alunos.processaAlunos(img, count_alunos, out_ilegivel, out_incerto, out_presente, out_ausente, out_erro_num, out_problemas, codigoAula)
                csv.criaCSVFile(alunosPresentes, codigoAula)
                i=i+1
            except:
                out_problemas += 1
                i=i+1
                continue
            
            
print("ERROS:\n\n")
print(f"Falha leitura do Codigo: {out_codigo_barras}\n")
print(f"ERROS:\nFalha do cabecalho: {out_leitura_cabecalho}\n")
print(f"ERROS:\nFalha no alinhamento: {out_alinhamento}\n")
print(f"NUMERO ALUNOS:{count_alunos}\n")
print(f"ERROS:\nAluno presentes: {out_presente}\n")
print(f"ERROS:\nAluno ausente: {out_ausente}\n")
print(f"ERROS:\nIlegivel: {out_ilegivel}\n")
print(f"ERROS:\nAssinatura incerta: {out_incerto}\n")
print(f"ERROS:\nErro leitura num: {out_erro_num}\n")
print(f"ERROS:\nFolha problemas {out_problemas}\n")

cv2.waitKey(0)
cv2.destroyAllWindows()

# _____________________________________________________________________________________________
# SINGLE PHOTO


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
