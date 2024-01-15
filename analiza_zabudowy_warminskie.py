# ekstraktowanie cech związanych z zabudową (forma miejscowości)
# wynik w 28/analiza_zabudowy_warminskie.csv
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

# ekstraktowanie cech z zabudowy
# liczba budynków
liczba_budynkow = budynki_grouped['SIMC_id'].value_counts().reset_index()
print(liczba_budynkow.sort_values(by='SIMC_id').head())
# łączna powierzchnia zabudowy
powierzchnia_budynkow = budynki_grouped.groupby('SIMC_id')['area'].sum().reset_index()
print(powierzchnia_budynkow.sort_values(by='SIMC_id').head())
# gęstość zabudowy
gestosc_zabudowy_pomoc = pd.merge(powierzchnia_budynkow,
                            granice_miejscowosci_z_adresacja[['SIMC_id','area']],
                            how='left', on='SIMC_id')
gestosc_zabudowy_pomoc['gestosc_zabudowy'] = gestosc_zabudowy_pomoc[
                                                 'area_x']/gestosc_zabudowy_pomoc['area_y']*100
gestosc_zabudowy = gestosc_zabudowy_pomoc[['SIMC_id', 'gestosc_zabudowy']]
print(gestosc_zabudowy.sort_values(by='SIMC_id').head())
# średnia liczba kondygnacji
srednia_liczba_kondygnacji = budynki_grouped.groupby('SIMC_id')['LKOND'].mean().reset_index()
print(srednia_liczba_kondygnacji.sort_values(by='SIMC_id').head())

# ekstraktowanie cech związanych ze stopniem rozproszenia zabudowy

# zamiana warstwy budynków z poligonowej na punktową — z pomocą centroidu poligonu
centroidy_budynkow = gpd.GeoDataFrame(data=budynki_grouped['SIMC_id'],
                                      geometry=budynki_grouped.centroid,
                                      crs=budynki_grouped.crs)
# zamiana na multipoint — centroidy zgrupowane względem SIMC_id
centroidy_merged = (centroidy_budynkow.groupby('SIMC_id')['geometry'].
                    apply(lambda x: MultiPoint(list(x))).reset_index())

# funkcja do wyznaczania macierzy odległości
def compute_distances_matrix(point_list):
    # zamiana współrzędnych na postać, którą akceptuje numpy
    coordinates = [(point.x, point.y) for point in point_list]
    # macierz odległości
    return cdist(coordinates, coordinates)

# funkcja do obliczania nna_index
def compute_nna_index(multipoint, area):
    # zamiana multipoint na listę punktów
    point_list = list(multipoint.geoms)
    distances_matrix = compute_distances_matrix(point_list)
    # zmiana wartości na diagonali z 0 (odległość punktu od jego samego)
    # na największą wartość z danej kolumny
    np.fill_diagonal(distances_matrix, np.max(distances_matrix, axis=0))
    # obliczenie sumy odległości do najbliższego sąsiada dla każdego punktu
    shortest_distance_sum = np.sum(np.min(distances_matrix, axis=1))
    # liczba punktów poddawanych analizie
    point_numer = len(point_list)
    # wzór na składnik d0
    d0 = shortest_distance_sum/point_numer
    # wzór na składnik de
    de = 0.5/np.sqrt(point_numer/area)
    # zwrócenie nna index
    return d0/de

# nna index — indeks analizy najbliższego sąsiada
nna_indexes = centroidy_merged.copy()
# powierzchnia pobrana z granic miejscowosci — w m2
nna_indexes = nna_indexes.merge(granice_miejscowosci_z_adresacja[['SIMC_id', 'area']],
                                how='left', on='SIMC_id')
nna_indexes.dropna(inplace=True)
nna_indexes.drop_duplicates(subset='SIMC_id', inplace=True)
nna_indexes['nna_index'] = [compute_nna_index(multipoint, area) for multipoint, area in
                            zip(nna_indexes['geometry'], nna_indexes['area'])]
nna_indexes = nna_indexes[['SIMC_id', 'nna_index']]
print(nna_indexes.sort_values(by='SIMC_id').head())

# obliczenie średniej odległości między punktami dla każdego multipointa (zbioru punktów)
def compute_mean_distance(multipoint):
    point_list = list(multipoint.geoms)
    distances_matrix = compute_distances_matrix(point_list)
    return np.mean(distances_matrix)
mean_distance = centroidy_merged.copy()
mean_distance['srednia_odleglosci'] = \
    [compute_mean_distance(multipoint) for multipoint in mean_distance['geometry']]
# obliczenie odchylenia standardowego odległości między punktami dla kazdego multipointa
def compute_std(multipoint):
    point_list = list(multipoint.geoms)
    distances_matrix = compute_distances_matrix(point_list)
    return np.std(distances_matrix)
std_distance = mean_distance.copy()
std_distance['odchylenie_standardowe'] = \
    [compute_std(multipoint) for multipoint in std_distance['geometry']]
# obliczenie współczynnika zmienności średniej odległości dla kazdego multipointa
std_distance['wspolczynnik_zmiennosci'] = (
        (std_distance['odchylenie_standardowe']/std_distance['srednia_odleglosci'])*100)

min_max_distance = centroidy_merged.copy()
# obliczenie min odległości między punktami dla każdego multipointa
def compute_min(multipoint):
    point_list = list(multipoint.geoms)
    distances_matrix = compute_distances_matrix(point_list)
    np.fill_diagonal(distances_matrix, np.max(distances_matrix, axis=0))
    return np.min(distances_matrix)
# obliczenie max odległości między punktami dla każdego multipointa
def compute_max(multipoint):
    point_list = list(multipoint.geoms)
    distances_matrix = compute_distances_matrix(point_list)
    return np.max(distances_matrix)
# obliczenie ptp odległości między punktami dla każdego multipointa
def compute_ptp(multipoint):
    point_list = list(multipoint.geoms)
    distances_matrix = compute_distances_matrix(point_list)
    return np.ptp(distances_matrix)
min_max_distance['min_odleglosc'] = \
    [compute_min(multipoint) for multipoint in min_max_distance['geometry']]
min_max_distance['max_odleglosc'] = \
    [compute_max(multipoint) for multipoint in min_max_distance['geometry']]
min_max_distance['min_max_rozstep'] = \
    [compute_ptp(multipoint) for multipoint in min_max_distance['geometry']]

std_distance_df = std_distance.drop('geometry', axis=1)
min_max_distance_df = min_max_distance.drop('geometry', axis=1)

# print(std_distance_df.sort_values(by='SIMC_id').head())
# print(min_max_distance_df.sort_values(by='SIMC_id').head())

# połączenie dotychczas wydobytych cech
zabudowa_merged = centroidy_merged.copy()
# merge wszystkich tabel po kolei po simc_id
tables_to_merge = [liczba_budynkow, powierzchnia_budynkow, gestosc_zabudowy,
                   srednia_liczba_kondygnacji,
                   nna_indexes, std_distance_df, min_max_distance_df]
for table in tables_to_merge:
    zabudowa_merged = zabudowa_merged.merge(table, how='left', on='SIMC_id')
    zabudowa_merged.drop_duplicates(inplace=True)
# zmiana nazw niektórych kolumn na bardziej czytelne i usunięcie geometrii centroidów
zabudowa_merged.rename(columns={'count': 'liczba_budynkow',
                                'area': 'powierzchnia_budynkow',
                                'LKOND': 'srednia_liczba_kondygnacji'}, inplace=True)
zabudowa_merged_df = zabudowa_merged.drop('geometry', axis=1)
print(zabudowa_merged_df.columns.to_list())

# zbadanie, czy występują wartości null albo nan
na = zabudowa_merged_df[zabudowa_merged_df.isnull().any(axis=1)]
# zamienienie wartości NaN na 0
zabudowa_merged_df.fillna(0, inplace=True)

# zapis do csv
# zabudowa_merged_df.to_csv("28/analiza_zabudowy_warminskie.csv",
#                          sep=';', header=True, index=False)

