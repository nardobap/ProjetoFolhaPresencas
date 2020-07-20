import cv2
import criaCSV as csv
import folhaPresenca
import CB
import alunos


if __name__ == '__main__':
    img = folhaPresenca.carregaImagem()
    img = folhaPresenca.corrigeAlinhamento(img)
    codigoAula = CB.processaCodigoBarras(img)
    NUM_SAMPLES = alunos.carregaNumerosSamples()
    todosAlunos, alunosPresentes = alunos.processaAlunos(img)
    csv.criaCSVFile(alunosPresentes, codigoAula)

cv2.waitKey(0)
cv2.destroyAllWindows()
