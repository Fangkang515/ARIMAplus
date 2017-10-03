import  arimaplus_dna as dna
import os.path

# Read input data
theFile = os.path.expanduser("~/Documents/dane.csv")
with open(theFile) as f:
    data = f.readlines()
data = [x.strip() for x in data]
data = [z.replace(',', '.') for z in data]
data = list(map(float, data))
print("Original data of length " + str(len(data)) + ":")
print(data)

# check for file with DNA code - if not find, create population and evaluate until winner organism emerges
if os.path.isfile(os.path.expanduser("~/Documents/the_dna.txt")):
    with open(os.path.expanduser("~/Documents/the_dna.txt")) as f:
        this_dna = f.readline()
    entity = dna.entity(this_dna)
    entity.forecasting(data)
    #entity.rysowanie(8, jednostka.dane_out)
else:
    pop = dna.population()
    pop.evaluation(data)
    while (not pop.anagenesis(data)):
        pop.evaluation(data)
    #for el in pop.pokolenie:
    #    el.rysowanie(8, el.dane_out)


