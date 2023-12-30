# sprawdzenie ile jest miejscowosci niesamodzielnych z adresacją w warminsko-mazurskim
# wynik w dane_pomocnicze/warminskie_nsam_adres.csv
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint

# miejscowości z prng — warstwa punktowa
miejscowosci_prng = gpd.read_file('PRNG_MIEJSCOWOSCI_SHP/PRNG_MIEJSCOWOSCI_SHP.shp')

# ograniczenie do województwa warmińsko-mazurskiego
warminskie_prng = miejscowosci_prng[miejscowosci_prng['wojewodz'] == 'warmińsko-mazurskie']

# zmiana typu kolumn na numeryczne - int64
warminskie_prng['idZewnetrz'] = pd.to_numeric(warminskie_prng['idZewnetrz'], errors='coerce').astype('Int64')
warminskie_prng['idMscNd'] = pd.to_numeric(warminskie_prng['idMscNd'], errors='coerce').astype('Int64')

# ograniczenie zbioru tylko do kilku najważniejszych kolumn
warminskie_prng = warminskie_prng[['idZewnetrz', 'rodzaj', 'nazwaGlown', 'idMscNd', 'geometry']]
# zmiana nazwy kolumny z idZewnetrz na SIMC_id (dla lepszej czytelności i umożliwienia późniejszego łączenia tabel)
warminskie_prng.rename(columns={'idZewnetrz': 'SIMC_id'}, inplace=True)
# usunięcie rekordów, w których brakuje identyfikatora SIMC
warminskie_prng.dropna(subset=['SIMC_id'], inplace=True)

# wyłączenie miast z analizy
warminskie_prng = warminskie_prng[~warminskie_prng['rodzaj'].isin(['miasto', 'część miasta', 'osiedle'])]
# usunięcie takich samych rekordów
warminskie_prng.drop_duplicates(inplace=True)

# dodanie kolumny 'czy_samodzielna'
warminskie_prng['czy_samodzielna'] = [True if pd.isna(id_msc_nd) else False for id_msc_nd in warminskie_prng['idMscNd']]

# pobranie punktów adresowych — warstwa punktowa
punkty_adresowe = gpd.read_file('28/PRG_PunktyAdresowe_28.shp')
# zmiana typu kolumny na numeryczny - int64
punkty_adresowe['SIMC_id'] = pd.to_numeric(punkty_adresowe['SIMC_id'], errors='coerce').astype('Int64')

# usunięcie rekordów, w których brakuje identyfikatora SIMC
punkty_adresowe_simc = punkty_adresowe.dropna(subset=['SIMC_id'])
# ograniczenie zbioru tylko do najważniejszych kolumn i identyfikatora SIMC i geometrii
punkty_adresowe_simc = punkty_adresowe_simc[['SIMC_id', 'geometry']]

# zgrupowanie rekordów wg identyfikatora SIMC — zmiana z geometrii pojedynczych punktów (Point) na skupiska (MultiPoint)
punkty_adresowe_simc = (punkty_adresowe_simc.groupby('SIMC_id')['geometry'].apply
                        (lambda x: MultiPoint(list(x))).reset_index())

# łączenie tabel
miejscowosci_z_adresacja = pd.merge(punkty_adresowe_simc,
                                    warminskie_prng[['SIMC_id', 'czy_samodzielna', 'nazwaGlown', 'rodzaj']], how='left',
                                    on='SIMC_id')
# usunięcie rekordów z brakującym identyfikatorem SIMC
miejscowosci_z_adresacja.dropna(subset=['SIMC_id'], inplace=True)
# usunięcie duplikatów
miejscowosci_z_adresacja.drop_duplicates(inplace=True)

# podział na zbiór miejscowości samodzielnych i niesamodzielnych
zbior_samodzielny = miejscowosci_z_adresacja[miejscowosci_z_adresacja['czy_samodzielna'] == True]
zbior_niesamodzielny = miejscowosci_z_adresacja[miejscowosci_z_adresacja['czy_samodzielna'] == False]

print(zbior_samodzielny)
print(zbior_niesamodzielny)
