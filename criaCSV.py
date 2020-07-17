import csv

def criaCSVFile(alunos, codigoAula):
    with open('Aula_' + str(codigoAula) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for aluno in alunos:
            writer.writerow([codigoAula, aluno, len(alunos)])


