# funkcjonalność
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, MultiPoint
from scipy.spatial.distance import cdist
import numpy as np

# pobranie granic miejscowości już po usunięciu tych bez własnej adresacji
granice_miejscowosci_z_adresacja = gpd.read_file('28/warminskie_granice_miejscowosci_z_adresacja.shp')
# pobranie budynków
budynki = gpd.read_file('28/warminskie_budynki.shp')
# ograniczenie warstwy granic tylko do dwóch kolumn
granice_temp = granice_miejscowosci_z_adresacja[['SIMC_id', 'geometry']]
# pogrupowanie budynków wg tego w jakiej miejscowości (granicy miejscowości się znajdują)
budynki_grouped = gpd.sjoin(budynki, granice_temp, how='inner', op="within")
budynki_grouped.drop_duplicates(inplace=True)

# informacyjne wyświetlenie, jakie są typy budynków
print(budynki_grouped['FUNOGBUD'].unique().tolist())
# cechy związane z licznością konkretnych typów budynków
rodzaje = (budynki_grouped.groupby(['SIMC_id', 'FUNOGBUD']).
           size().reset_index(name='Liczba'))
# zliczenie ile jest jakiego rodzaju budynku dla poszczególnych numerów miejscowości
rodzaje_budynkow = (rodzaje.pivot(index='SIMC_id', columns='FUNOGBUD', values='Liczba').
                    reset_index())
rodzaje_budynkow = rodzaje_budynkow.fillna(0).astype('Int64')
# print(rodzaje_budynkow)

# warstwy poligonowe terenów zalesionych i terenów uprawnych
lasy = gpd.read_file('28/warminskie_lasy.shp')
pola = gpd.read_file('28/warminskie_uprawa.shp')
# intersekcja warstwy z granicami miejscowości-
# podzielenie warstwy wg granic miejscowości i dopisanie simc_id
lasy_grouped = gpd.overlay(granice_temp, lasy, how='intersection')
pola_grouped = gpd.overlay(granice_temp, pola, how='intersection')
# stworzenie multipoligonów odpowiadającym terenom zalesionym i terenom uprawnym
# konkretnych miejscowości
lasy_merged = lasy_grouped[['SIMC_id', 'geometry']].dissolve(by='SIMC_id').reset_index()
pola_merged = pola_grouped[['SIMC_id', 'geometry']].dissolve(by='SIMC_id').reset_index()

# łączna powierzchnia
powierzchnia_terenow_zalesionych = lasy_grouped.groupby('SIMC_id')['area'].sum().reset_index()
powierzchnia_terenow_zalesionych.rename(columns={'area': 'powierzchnia_terenow_zalesionych'},
                                        inplace=True)
powierzchnia_terenow_uprawnych = pola_grouped.groupby('SIMC_id')['area'].sum().reset_index()
powierzchnia_terenow_uprawnych.rename(columns={'area': 'powierzchnia_terenow_uprawnych'},
                                      inplace=True)
# łączenie tabel i obliczenie stosunków powierzchni
tereny_uprawne_zalesione = pd.merge(powierzchnia_terenow_zalesionych,
                        granice_miejscowosci_z_adresacja[['SIMC_id', 'area']],
                        how='left', on='SIMC_id')
tereny_uprawne_zalesione = pd.merge(powierzchnia_terenow_uprawnych, tereny_uprawne_zalesione,
                        how='left', on='SIMC_id')
tereny_uprawne_zalesione['stosunek_powierzchni_upraw'] = (
    tereny_uprawne_zalesione)['powierzchnia_terenow_uprawnych']/tereny_uprawne_zalesione['area']
tereny_uprawne_zalesione['stosunek_powierzchni_lasow'] = (
    tereny_uprawne_zalesione)['powierzchnia_terenow_zalesionych']/tereny_uprawne_zalesione['area']
tereny_uprawne_zalesione.drop('area', axis=1, inplace=True)
dane_funkcjonalnosc = pd.merge(tereny_uprawne_zalesione, rodzaje_budynkow, how='left', on='SIMC_id')
print(tereny_uprawne_zalesione.sort_values(by='SIMC_id').head())
# zapis do csv
dane_funkcjonalnosc.to_csv("28/analiza_funkcje_warminskie.csv",
                          sep=';', header=True, index=False)


