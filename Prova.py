import math
import numpy as np

fout = open('Output.csv', 'w')

Header = "File,startCity,cost\n"
riga0 = "prova,prova,lungaprova\n"
riga1 = "Nome File," + str(234) + "," + str(124.345) + "\n"
fout.write(Header)
fout.write(riga0)
fout.write(riga1)
fout.close()

def prova():
    print("Ciao")
    return

Ciao = prova()

print(Ciao)

listaTest = [2,4,6,76,564]

for i in listaTest:
    print(i)



lista = [3,6,9]

mean_calc = (3+6+9)/3
mean_func = np.mean(lista)

print(mean_calc)
print(mean_func)