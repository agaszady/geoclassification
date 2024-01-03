# wstępna analiza ile jest jakich rodzajów miejscowości w prng, podział na województwa, jest też w pliku notatnikowym warminskie_dane
# wynik w dane_pomocnicze/woj_liczbowo.csv i dane_pomocnicze/woj_procentowo.csv
import geopandas as gpd

# miejscowości z prng — warstwa punktowa
miejscowosci_prng = gpd.read_file('PRNG_MIEJSCOWOSCI_SHP/PRNG_MIEJSCOWOSCI_SHP.shp')

# liczność każdego z rodzajów z podziałem na samodzielny/niesamodzielny
zliczenie_rodzajow = (miejscowosci_prng.groupby('rodzaj')['idMscNd'].
                      agg([("Liczba niesamodzielnych", lambda x: x.count()),
                           ("Liczba samodzielnych", lambda x: x.isna().sum())]))
# sortowanie malejąco
zliczenie_rodzajow = (
    zliczenie_rodzajow.reset_index().sort_values(by=['Liczba samodzielnych',
                                                     'Liczba niesamodzielnych'],
                                                 ascending=False))

print(zliczenie_rodzajow)

# ograniczenie miejscowości do samodzielnych — bez miejscowości nadrzędnej
miejscowosci_prng_2 = miejscowosci_prng[miejscowosci_prng['idMscNd'].isna()]
# wyłączenie miast i pochodnych z analizy
miejscowosci_prng_2 = (
    miejscowosci_prng_2)[~miejscowosci_prng_2['rodzaj'].isin({'miasto', 'osiedle', 'część miasta'})]

tabela_licznosci = miejscowosci_prng_2.groupby(['wojewodz', 'rodzaj']).size().unstack(fill_value=0)
tabela_procentow = tabela_licznosci.div(tabela_licznosci.sum(axis=1), axis=0) * 100
tabela_procentow = tabela_procentow.round(2)

print(tabela_licznosci)
print(tabela_procentow)
