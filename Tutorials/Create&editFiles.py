import math

fout = open('Output.csv', 'w')

Header = "File,startCity,cost\n"
riga0 = "prova,prova,lungaprova\n"
riga1 = "Nome File," + str(234) + "," + str(124.345) + "\n"
fout.write(Header)
fout.write(riga0)
fout.write(riga1)
fout.close()

