import  arimaplus_dna as dna
import os.path

#wczytaj plik
plik = os.path.expanduser("~/Documents/dane.csv")
with open(plik) as f:
    dane = f.readlines()
dane = [x.strip() for x in dane]
dane = [z.replace(',', '.') for z in dane]
dane = list(map(float, dane))

#czy jest kod genetyczny?
if os.path.isfile(os.path.expanduser("~/Documents/the_dna.txt")):
    with open(os.path.expanduser("~/Documents/the_dna.txt")) as f:
        to_dna = f.readline()
    jednostka = dna.jednostka(to_dna)
    jednostka.przewidywanie(dane)
    #jednostka.rysowanie(8, jednostka.dane_out)
else:
    pop = dna.populacja()
    pop.ewaluacja(dane)
    while (not pop.anageneza(dane)):
        pop.ewaluacja(dane)
    #for el in pop.pokolenie:
    #    el.rysowanie(8, el.dane_out)


