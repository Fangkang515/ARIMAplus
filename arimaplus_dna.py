"""
This module describes creating and evoliving population
of solutions to prediction problem.
"""

import arimaplus_neurony as ann
import arimaplus_przewidywanie as przewidywanie
import arimaplus_math as ap_math
import math
import sys
import random
import os.path
import struct


class populacja(object):
    def __init__(self):
        self.liczebnosc = 15
        self.dna_pokolenie = []
        self.pokolenie = []
        self.wyniki = []
        self.nr_pokolenia = 1
        self.maks_pokolenie = 50
        self.mod_pureARIMA = "00010011010010110010110010110010100001001010100000000000000000000000000000000000000000"
        self.mod_pureANN = "00100010000011000000100101001011001101111100010110001000110000000000000000000000000000"
        self.mod_ANNreg = "10000000000000000000000000000000010000010011110110001011000000000000000000000001011000"
        self.mod_ARIMANN = "11010101000100010010000100010001010110000001100000001011000000000011110000000001011000"
        self.dna_pokolenie.append(self.mod_pureARIMA)
        self.dna_pokolenie.append(self.mod_pureANN)
        self.dna_pokolenie.append(self.mod_ANNreg)
        self.dna_pokolenie.append(self.mod_ARIMANN)
        while len(self.dna_pokolenie) != self.liczebnosc:
            self.dna_pokolenie.append(self.generuj_kod())

    def ewaluacja(self, dane_in=[]):
        # otrzymując kody dna tworzy jednostki i doprowadza do uzyskania rmse
        self.wyniki = []
        for n in self.dna_pokolenie:
            self.pokolenie.append(jednostka(n))
        for j in self.pokolenie:
            self.wyniki.append(j.przewidywanie(dane_in))

    def anageneza(self, dane_in=[]):
        # sprawdza liczbę pokoleń (war. końcowy 1) lub rmse najlepszych - sortowanie - (war. końcowy 2)
        # przeprowadza mutacje i krzyżowanie oraz zapomnienie nieskutczenych
        self.srednie_rmse = sum([n.rmse for n in self.pokolenie]) / len(self.pokolenie)
        self.odchylenie_rmse = ap_math.odchylenie([n.rmse for n in self.pokolenie])
        for i, n in enumerate(self.pokolenie):
            if n.rmse > (self.srednie_rmse + self.odchylenie_rmse):
                del self.pokolenie[i]
        self.pokolenie.sort(lambda x: x.rmse, reverse=False)
        if self.pokolenie[0].rmse < ap_math.odchylenie(dane_in) or self.nr_pokolenia >= self.maks_pokolenie:
            return True
        else:
            if len(self.pokolenie) >= 2:
                while len(self.pokolenie) < math.ceil(self.liczebnosc / 2):
                    self.pokolenie.append(self.mutacja(self.dziedziczenie(self.pokolenie[0], self.pokolenie[1])))
                while len(self.pokolenie) <= self.liczebnosc:
                    self.pokolenie.append(self.generuj_kod())
            else:
                while len(self.pokolenie) <= self.liczebnosc:
                    self.pokolenie.append(self.generuj_kod())
            self.nr_pokolenia += 1
            return False

    def generuj_kod(self):
        kod = ""
        for i in range(0, 86):
            p = random.randint(0, 100)
            kod += "1" if p % 2 == 0 else "0"
        return kod

    def mutacja(self, kod):
        for i, c in enumerate(kod):
            kod[i] = kod[i] if random.randint(0, len(kod)) != 0 else ("1" if kod[i] == "0" else "0")
        return kod

    def dziedziczenie(self, kodA, kodB):
        self.kodC = ""
        self.ile_krzyzowan = random.randint(1, math.floor(self.liczebnosc / 2))
        self.pkt_krzyzowania = []
        for i in range(0, self.ile_krzyzowan):
            self.pkt_krzyzowania.append(math.floor(random.gauss(len(kodA) / 2, len(kodB) / 6)))
        self.pkt_krzyzowania = list(set(self.pkt_krzyzowania))
        if len(self.pkt_krzyzowania) == 1:
            kodC = kodA[0:self.pkt_krzyzowania[0]] + kodB[self.pkt_krzyzowania[0]:len(kodB)] if random.randint(0,
                                                                                                               2) == 0 else kodB[
                                                                                                                            0:
                                                                                                                            self.pkt_krzyzowania[
                                                                                                                                0]] + kodA[
                                                                                                                                      self.pkt_krzyzowania[
                                                                                                                                          0]:len(
                                                                                                                                          kodA)]
        else:
            kodC = kodA[0:self.pkt_krzyzowania[0]] if random.randint(0, 2) == 0 else kodB[0:self.pkt_krzyzowania[0]]
            for i, n in enumerate(self.pkt_krzyzowania, start=1):
                kodC += kodA[self.pkt_krzyzowania[i - 1]:n] if random.randint(0, 2) == 0 else kodB[self.pkt_krzyzowania[
                    i - 1]:n]
            kodC += kodA[self.pkt_krzyzowania[len(self.pkt_krzyzowania) - 1]:] if random.randint(0, 2) == 0 else kodB[
                                                                                                                 self.pkt_krzyzowania[
                                                                                                                     len(
                                                                                                                         self.pkt_krzyzowania) - 1]:]
        return kodC

    def zapisz_dna(self):
        with open(os.path.expanduser("~/the_dna.txt")) as f:
            f.write(self.pokolenie[0])


class jednostka(object):
    def __init__(self, dna):
        self.rmse = 0
        self.dane_in = []
        self.dane_out = []
        self.dane_posrednie = []
        #
        self.typ_przewidywania = 0
        self.typ_AR = 0
        self.typ_I = 0
        self.typ_MA = 0
        self.typ_coefA = 1
        self.typ_coefB = 1
        self.typ_coefC = 1
        self.typ_errorA = 1
        self.typ_errorB = 1
        self.typ_errorC = 1
        self.typ_okno = 0
        self.ile_warstw = 0
        self.typ_neuronu = 0
        self.wsp_nauki = 0.1
        self.topologia = {}

        for i in range(0, 3):
            self.typ_przewidywania += int(dna[i]) * math.pow(2, i)
        for i in range(3, 5):
            self.typ_AR += int(dna[i]) * math.pow(2, i - 3)
        for i in range(5, 7):
            self.typ_I += int(dna[i]) * math.pow(2, i - 5)
        for i in range(7, 9):
            self.typ_MA += int(dna[i]) * math.pow(2, i - 7)
        for i in range(9, 13):
            self.typ_coefA += int(dna[i]) * math.pow(2, i - 9)
        for i in range(13, 17):
            self.typ_coefB += int(dna[i]) * math.pow(2, i - 13)
        for i in range(17, 21):
            self.typ_coefC += int(dna[i]) * math.pow(2, i - 17)
        for i in range(21, 25):
            self.typ_errorA += int(dna[i]) * math.pow(2, i - 21)
        for i in range(25, 29):
            self.typ_errorB += int(dna[i]) * math.pow(2, i - 25)
        for i in range(29, 33):
            self.typ_errorC += int(dna[i]) * math.pow(2, i - 29)
        for i in range(33, 37):
            self.typ_okno += int(dna[i]) * math.pow(2, i - 33)
        self.typ_okno = self.typ_okno + 1
        for i in range(37, 39):
            self.typ_neuronu += int(dna[i]) * math.pow(2, i - 37)
        for i in range(39, 45):
            self.wsp_nauki += int(dna[i]) * math.pow(2, i - 39)
        self.wsp_nauki = 1 / (self.wsp_nauki + 1)

        for i in range(0, 6):
            if int(dna[45 + i * 7]) == 1:
                self.topologia[self.ile_warstw] = sum(
                    list(map(lambda x: int(x[1]) * math.pow(2, int(x[0])), enumerate(dna[45 + i * 7:45 + i * 7 + 6]))))
                self.topologia[self.ile_warstw] = self.topologia[self.ile_warstw] + 1
                self.ile_warstw += 1

    def przewidywanie(self, dane=[]):
        """Kompletne przewidywanie dla zestawu danych, od utworzenia
        odpowiednich-ewentualnych sieci aż do wyznaczenia RMSE"""
        while (len(dane) % self.typ_okno != 0):
            del dane[0]

        if self.typ_I == 0:
            self.dane_in = ap_math.normalizuj(dane)
        else:
            self.dane_in = przewidywanie.roznicoj_dane(self.typ_I, dane)

        # czysta arima
        if self.typ_przewidywania == 0:
            self.dane_posrednie = przewidywanie.ARIMA(self.typ_AR, self.typ_MA, self.typ_coefA, self.typ_coefB,
                                                      self.typ_coefC, self.typ_errorA, self.typ_errorB, self.typ_errorC,
                                                      dane)
            self.dane_out = [el[0] for el in self.dane_posrednie]
            self.rmse = ap_math.rmse(self.dane_in, self.dane_out)
            return self.dane_out

        # czysta ann
        elif self.typ_przewidywania == 1:
            if self.typ_neuronu == 0:
                self.siec = ann.siec_zwykly(self.typ_okno, self.topologia, self.typ_przewidywania, self.typ_AR+self.typ_MA)
            elif self.typ_neuronu == 1:
                self.siec = ann.siec_GRU(self.typ_okno, self.topologia, self.typ_przewidywania, self.typ_AR+self.typ_MA)
            elif self.typ_neuronu == 2:
                self.siec = ann.siec_LSTM(self.typ_okno, self.topologia, self.typ_przewidywania, self.typ_AR+self.typ_MA)
            elif self.typ_neuronu == 3:
                self.siec = ann.siec_zwykly(self.typ_okno, self.topologia, self.typ_przewidywania, self.typ_AR+self.typ_MA)  # powtorzenie umyślne
            for i in range(0, len(self.dane_in) / self.typ_okno):
                self.dane_posrednie.extend(
                    self.siec.forward_pass(self.dane_in[i * self.typ_okno:i * self.typ_okno + self.typ_okno]))
                if i < len(self.dane_in) / self.typ_okno - 1:
                    self.siec.backward_pass(
                        self.dane_in[(i + 1) * self.typ_okno:(i + 1) * self.typ_okno + self.typ_okno], self.wsp_nauki,
                        self.dane_in[i * self.typ_okno:i * self.typ_okno + self.typ_okno])
            self.rmse = ap_math.rmse(self.dane_in, self.dane_posrednie)
            return self.dane_posrednie

        # regresja liniowa BEZ ann
        elif self.typ_przewidywania == 2:
            self.rmse = sys.maxsize
            return 0

        # regresja wielomianowa BEZ ann
        elif self.typ_przewidywania == 3:
            self.rmse = sys.maxsize
            return 0

        # regresja liniowa z ann
        elif self.typ_przewidywania == 4:
            wskazniki = przewidywanie.licz_reg_liniowa(self.typ_okno, self.dane_in)
            if self.typ_neuronu == 0:
                self.siec = ann.siec_zwykly(self.typ_okno, self.topologia, self.typ_przewidywania, self.typ_AR+self.typ_MA)
            elif self.typ_neuronu == 1:
                self.siec = ann.siec_GRU(self.typ_okno, self.topologia, self.typ_przewidywania, self.typ_AR+self.typ_MA)
            elif self.typ_neuronu == 2:
                self.siec = ann.siec_LSTM(self.typ_okno, self.topologia, self.typ_przewidywania, self.typ_AR+self.typ_MA)
            elif self.typ_neuronu == 3:
                self.siec = ann.siec_zwykly(self.typ_okno, self.topologia, self.typ_przewidywania, self.typ_AR+self.typ_MA)  # powtorzenie umyślne
            for i in range(0, int(len(self.dane_in) / self.typ_okno)):
                self.dane_posrednie.append(
                    self.siec.forward_pass(self.dane_in[int(i * self.typ_okno):int(i * self.typ_okno + self.typ_okno)]))
                if i < len(self.dane_in) / self.typ_okno - 1:
                    self.siec.backward_pass(wskazniki[i], self.wsp_nauki,
                                            self.dane_in[int(i * self.typ_okno):int(i * self.typ_okno + self.typ_okno)])
            mem = 1
            for i in self.dane_in:
                mem = mem + 1 if mem < self.typ_okno else 1
                self.dane_out.append(mem * self.dane_posrednie[i / self.typ_okno])
            self.rmse = ap_math.rmse(self.dane_in, self.dane_out)
            return self.dane_out

        # regresja wielomianowa z ann
        elif self.typ_przewidywania == 5:
            wskazniki = przewidywanie.licz_reg_wielomianiowa(self.typ_okno, 2, self.dane_in)
            if self.typ_neuronu == 0:
                self.siec = ann.siec_zwykly(self.typ_okno, self.topologia, self.typ_przewidywania, self.typ_AR+self.typ_MA)
            elif self.typ_neuronu == 1:
                self.siec = ann.siec_GRU(self.typ_okno, self.topologia, self.typ_przewidywania, self.typ_AR+self.typ_MA)
            elif self.typ_neuronu == 2:
                self.siec = ann.siec_LSTM(self.typ_okno, self.topologia, self.typ_przewidywania, self.typ_AR+self.typ_MA)
            elif self.typ_neuronu == 3:
                self.siec = ann.siec_zwykly(self.typ_okno, self.topologia, self.typ_przewidywania, self.typ_AR+self.typ_MA)  # powtorzenie umyślne
            for i in range(0, len(self.dane_in) / self.typ_okno):
                self.dane_posrednie.append(
                    self.siec.forward_pass(self.dane_in[i * self.typ_okno:i * self.typ_okno + self.typ_okno]))
            if i < len(self.dane_in) / self.typ_okno - 1:
                self.siec.backward_pass(wskazniki[i], self.wsp_nauki,
                                        self.dane_in[i * self.typ_okno:i * self.typ_okno + self.typ_okno])
            mem = 1
            for i in self.dane_in:
                mem = mem + 1 if mem < self.typ_okno else 1
                self.dane_out.append(
                    mem ** 2 * self.dane_posrednie[i / self.typ_okno] + mem * self.dane_posrednie[i / self.typ_okno])
            self.rmse = ap_math.rmse(self.dane_in, self.dane_out)
            return self.dane_out

        # arima z ann (ann przeiwduje błąd)
        elif self.typ_przewidywania == 6 and self.typ_AR != 0:
            self.arima_out = przewidywanie.ARIMA(self.typ_MA, self.typ_AR, self.typ_coefA, self.typ_coefB,
                                                 self.typ_coefC, self.typ_errorA, self.typ_errorB, self.typ_errorC,
                                                 self.dane_in)
            for i in range(0, len(self.dane_in) / self.typ_okno):
                self.dane_posrednie.extend(
                    self.siec.forward_pass(self.dane_in[i * self.typ_okno:i * self.typ_okno + self.typ_okno]))
                if i < len(self.dane_in) / self.typ_okno - 1:
                    self.siec.backward_pass(self.arima_out[i * self.typ_okno:i * self.typ_okno + self.typ_okno][1],
                                            self.wsp_nauki,
                                            self.dane_in[i * self.typ_okno:i * self.typ_okno + self.typ_okno])
                self.dane_out.append(self.arima_out[i][0] - self.dane_posrednie[i])
            self.rmse = ap_math.rmse(self.dane_in, self.dane_out)
            return self.dane_out

        # arima z ann (ann przeiwduje błąd) : powtorzenie umyślne
        elif self.typ_przewidywania == 7 and self.typ_AR != 0:
            self.arima_out = przewidywanie.ARIMA(self.typ_MA, self.typ_AR, self.typ_coefA, self.typ_coefB,
                                                 self.typ_coefC, self.typ_errorA, self.typ_errorB, self.typ_errorC,
                                                 self.dane_in)
            for i in range(0, len(self.dane_in) / self.typ_okno):
                self.dane_posrednie.extend(
                    self.siec.forward_pass(self.dane_in[i * self.typ_okno:i * self.typ_okno + self.typ_okno]))
                if i < len(self.dane_in) / self.typ_okno - 1:
                    self.siec.backward_pass(self.arima_out[i * self.typ_okno:i * self.typ_okno + self.typ_okno][1],
                                            self.wsp_nauki,
                                            self.dane_in[i * self.typ_okno:i * self.typ_okno + self.typ_okno])
                self.dane_out.append(self.arima_out[i][0] - self.dane_posrednie[i])
            self.rmse = ap_math.rmse(self.dane_in, self.dane_out)
            return self.dane_out

    def jakie_roznicowanie(self):
        return self.typ_I

    def rysowanie(self, bity, *szeregi):
        self.rowno4 = lambda n: int(math.ceil(n / 4)) * 4
        self.rowno8 = lambda n: int(math.ceil(n / 8)) * 8
        self.format_short = lambda n: struct.pack("<h", n)
        self.format_int = lambda n: struct.pack("<i", n)
        self.wynik = []
        for i, el in enumerate(szeregi):
            wysokosc = len(el)
            szerokosc = int(self.rowno8(bity) / 8)
            przesuniecie = [0] * (self.rowno4(szerokosc) - szerokosc)
            rozmiar = self.format_int(self.rowno4(bity) * wysokosc + 0x20)
            self.wynik.append((b"BM" + s + b"\x00\x00\x00\x00\x20\x00\x00\x00\x0C\x00\x00\x00" +
                self.format_short(szerokosc) + self.format_short(wysokosc) + b"\x01\x00\x01\x00\xff\xff\xff\x00\x00\x00" +
                b"".join([bytes(rzad + przesuniecie) for rzad in reversed(el)])))
