# wielkość
import pandas as pd
import geopandas as gpd

# przygotowanie warstwy prng
# miejscowości z prng — warstwa punktowa
miejscowosci_prng = gpd.read_file('PRNG_MIEJSCOWOSCI_SHP/PRNG_MIEJSCOWOSCI_SHP.shp')
# ograniczenie do województwa warmińsko-mazurskiego
warminskie_prng = miejscowosci_prng[miejscowosci_prng['wojewodz'] == 'warmińsko-mazurskie']
# ograniczenie do najistotniejszych kolumn
warminskie_prng = warminskie_prng[['idZewnetrz', 'rodzaj', 'nazwaGlown', 'idMscNd']]
# zmiana nazw dla czytelności
warminskie_prng.rename(columns={'idZewnetrz': 'SIMC_id', 'nazwaGlown': 'nazwa',
                                'idMscNd': 'id_msc_nd'}, inplace=True)
# zmiana typów z object na int64
warminskie_prng['SIMC_id'] = pd.to_numeric(warminskie_prng['SIMC_id'],
                                           errors='coerce').astype('Int64')
warminskie_prng['id_msc_nd'] = pd.to_numeric(warminskie_prng['id_msc_nd'],
                                             errors='coerce').astype('Int64')
# wyłączenie miast i pochodnych z analizy
warminskie_prng = warminskie_prng[~warminskie_prng['rodzaj'].isin(
    {'miasto', 'osiedle', 'część miasta'})]
warminskie_prng.dropna(subset=['SIMC_id'], inplace=True)
# pozbycie się ewentualnych duplikatów
warminskie_prng.drop_duplicates(inplace=True)
# dodanie kolumny 'czy_samodzielna' i usunięcie kolumny 'id_msc_nd'
warminskie_prng['czy_samodzielna'] = \
    [True if pd.isna(id_msc_nd) else False for id_msc_nd in warminskie_prng['id_msc_nd']]
warminskie_prng.drop('id_msc_nd', axis=1, inplace=True)
print(warminskie_prng.sort_values(by='SIMC_id').head())

# pobranie punktów adresowych
punkty_adresowe = gpd.read_file("28/PRG_PunktyAdresowe_28.shp")
# zmiana typu z object na int64
punkty_adresowe['SIMC_id'] = pd.to_numeric(punkty_adresowe['SIMC_id'],
                                           errors='coerce').astype('Int64')
# usunięcie adresów bez przypisania do simc (id miejscowości)
punkty_adresowe = punkty_adresowe.dropna(subset=['SIMC_id'])
# stworzenie setów zawierających numery simc z warstwy adresowej i miejscowosci prng
set_punkty_adresowe = set(punkty_adresowe['SIMC_id'])
set_prng = set(warminskie_prng['SIMC_id'])
# wyznaczenie części wspólnej, zostawienie tylko miejscowości z adresacją
simc_z_adresacja = set_prng & set_punkty_adresowe
# ograniczenie warstwy miejscowości tylko do tych z adresacją
warminskie_prng = warminskie_prng[warminskie_prng['SIMC_id'].isin(simc_z_adresacja)]
print(warminskie_prng.sort_values(by='SIMC_id').head())

# ekstraktowanie danych z warstwy adresów — liczba adresów
liczba_adresow = punkty_adresowe['SIMC_id'].value_counts().reset_index()
liczba_adresow.rename(columns={'count': 'liczba_adresow'}, inplace=True)
# print(liczba_adresow.sort_values(by='SIMC_id').head())

# wczytanie granic miejscowości — warstwa poligonowa
granice_miejscowosci = gpd.read_file('28/warminskie_granice_miejscowosci.shp')
# zmiana typu i zmiana nazwy kolumny klucza dla czytelności
granice_miejscowosci['IDTERYTMSC'] = pd.to_numeric(granice_miejscowosci['IDTERYTMSC'],
                                                   errors='coerce').astype('Int64')
granice_miejscowosci.rename(columns={'IDTERYTMSC': 'SIMC_id'}, inplace=True)
# set z identyfikatorami simc (miejscowościami) występującymi w warstwie granic miejscowości
set_granice_miejscowosci = set(granice_miejscowosci['SIMC_id'])
# stworzenie nowej warstwy pomocniczej — granice miejscowości z adresacją
set_granice_z_adresacja = simc_z_adresacja & set_granice_miejscowosci
granice_miejscowosci_pomoc = granice_miejscowosci[granice_miejscowosci['SIMC_id'].isin(
    set_granice_z_adresacja)]
granice_miejscowosci_pomoc.drop_duplicates(inplace=True)
granice_miejscowosci_pomoc.to_file("28/warminskie_granice_miejscowosci_z_adresacja.shp")

# ekstraktowanie danych z warstwy granic — liczba mieszkańców i powierzchnia
granice_miejscowosci_mieszkancy = granice_miejscowosci_pomoc[['SIMC_id', 'LMIESZKANC']]
granice_miejscowosci_powierzchnia = granice_miejscowosci_pomoc[['SIMC_id', 'area']]

granice_miejscowosci_mieszkancy.dropna(inplace=True)
granice_miejscowosci_mieszkancy.rename(columns={'LMIESZKANC': 'liczba_mieszkancow'}, inplace=True)
granice_miejscowosci_mieszkancy['liczba_mieszkancow'] = (
    granice_miejscowosci_mieszkancy['liczba_mieszkancow'].astype('Int64'))
granice_miejscowosci_mieszkancy.drop_duplicates(inplace=True)
print(granice_miejscowosci_mieszkancy.sort_values(by='SIMC_id').head())

granice_miejscowosci_powierzchnia.drop_duplicates(inplace=True)
granice_miejscowosci_powierzchnia.rename(columns={'area': 'powierzchnia'}, inplace=True)
print(granice_miejscowosci_powierzchnia.sort_values(by='SIMC_id').head())

# łączenie tabel
# dodanie liczby adresów
warminskie_prng = warminskie_prng.merge(liczba_adresow, how='left', on='SIMC_id')
# dodanie liczby mieszkańców
warminskie_prng = warminskie_prng.merge(granice_miejscowosci_mieszkancy, how='left', on='SIMC_id')
warminskie_prng['liczba_mieszkancow'] = warminskie_prng['liczba_mieszkancow'].fillna(0)
# dodanie powierzchni
warminskie_prng = warminskie_prng.merge(granice_miejscowosci_powierzchnia, how='left', on='SIMC_id')
# zagęszczenie ludności
warminskie_prng['zageszczenie_ludnosci'] = warminskie_prng['liczba_mieszkancow']/warminskie_prng[
    'powierzchnia']*100
warminskie_prng.drop_duplicates(subset='SIMC_id', inplace=True)
print(warminskie_prng.sort_values(by='SIMC_id').head())
warminskie_prng.to_csv("28/analiza_wielkosci_warminskie.csv", sep=';', header=True, index=False)


# granice_miejscowosci_pomoc.to_file("28/warminskie_granice_miejscowosci_z_adresacja.shp")

