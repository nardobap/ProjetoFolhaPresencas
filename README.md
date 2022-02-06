# ProjetoFolhaPresencas

### Projecto Licenciatura DEIS-ISEC
### Bernardo Baptista - 2018013802
***
## Table of Contents
1. [Prerequisites](#project-description)
2. [Project description](#users-manual)
3. [User's manual](#users-manual)
4. [Fourth Example](#fourth)



#### Previous work: [GitHub](https://github.com/renatogomes17/ProjetoFolhaPresencas)

## Prerequisites

| Library       | Version used     | 
| ------------- |:-------------:| 
| Imutils |	0.5.4 |
| Matplotlib |	3.4.2|
| Numpy	|1.20.3|
| OpenCV	|4.5.3|
| Pandas	|1.3.2|
| Pdf2image	|1.16.0|
| Pytesseract|	0.3.8|
| Python	|3.8.8|
| Pyzbar |0.1.8
| Scikit-learn|	0.24.2|
| Seaborn|	0.11.2|
| Tessaract|4.1.1| 

It's useful to have all the frameworks used available at your local machine if you want to test it:

>$ pip install -U scikit-learn numpy opencv-python pandas pyzbar matplotlib seaborn pytesseract imutils pdf2image

It is also required to have Tessaract installed: [Tesseract](https://tesseract-ocr.github.io/tessdoc/Home.html)


## Project description
Input: One or more images of attendance sheets;
Ouput: A .csv file with the students present in a given lesson.

The aim of this project is to develop a functional prototype for reading, through computer vision, of attendance sheets in classes. The IPC uses in all its schools the management system NONIO, developed by the company XWS - eXpress Web Solutions. The goal is to automate the process of recognizing signatures on attendance sheets and validate attendance. The application's primary goal is to identify the students that were present in each class and that have signed the attendance sheet.

This project will be combined with another to establish the connection of the application to the NONIO system. This means that the output of this program will be redirected to an API that will automatically register the students present in each class. That was or will be developed, in parallel, to route the data regarding the presence of the students in class to the NONIO system, so these features are not relevant in this project.

## User's manual
yada

## fourth
yada
