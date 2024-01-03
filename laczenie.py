import pandas as pd
# łączenie zbioru danych
zmienne_wielkosc = pd.read_csv('28/analiza_wielkosci_warminskie.csv', sep=';')
zmienne_zabudowa = pd.read_csv('28/analiza_zabudowy_warminskie.csv', sep=';')
zmienne_funkcjonalnosc = pd.read_csv('28/analiza_funkcje_warminskie.csv', sep=';')

dane_merged = pd.merge(zmienne_wielkosc, zmienne_zabudowa, how='left', on='SIMC_id')
dane_merged = dane_merged.merge(zmienne_funkcjonalnosc, how='left', on='SIMC_id')
print(dane_merged.columns)
dane_merged.drop_duplicates(subset=['SIMC_id'], inplace=True)
dane_merged.to_csv('warminskie_dane_do_analizy.csv', sep=';', header=True, index=False)

