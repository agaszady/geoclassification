# zastosowanie najlepszego modelu na zbiorze miejscowości niesamodzielnych z własną adresacją
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# pobranie zbioru danych
dane_warminskie = pd.read_csv('warminskie_dane_do_analizy.csv', sep=';')
mapowanie = \
    {0: 'wieś',
     1: 'osada',
     2: 'osada leśna',
     3: 'kolonia'}
# ograniczenie zbioru danych do miejscowości niesamodzielnych
zbior_niesamodzielny = (
    dane_warminskie)[dane_warminskie['czy_samodzielna'] == False]
zbior_niesamodzielny = zbior_niesamodzielny.fillna(0)
# usunięcie kolumn pomocniczych — w tym aktualnego rodzaju
zbior_niesamodzielny_pomoc = zbior_niesamodzielny[['SIMC_id', 'nazwa', 'rodzaj']]
zbior_niesamodzielny_pomoc = zbior_niesamodzielny_pomoc.rename(columns={'rodzaj':
                                                                            'aktualny_rodzaj'})
zbior_niesamodzielny = zbior_niesamodzielny.drop(['SIMC_id', 'nazwa', 'czy_samodzielna', 'rodzaj'],
                                              axis=1)
zbior_niesamodzielny_pomoc = zbior_niesamodzielny_pomoc.sort_values(by='SIMC_id', ascending=True)

# transformacja danych
X = zbior_niesamodzielny
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
zbior_niesamodzielny = pd.DataFrame(X_scaled, columns=X.columns)
np.set_printoptions(suppress=True)

# pobranie modelu
model_gb = joblib.load('modele/gb_over_samp.joblib')
# zastosowanie modelu
y_proba = np.round(model_gb.predict_proba(zbior_niesamodzielny), 3)
y_pred = np.argmax(y_proba, axis=1)

# wypisanie prawdopodobieństw przynależności do każdej z klas (rodzajów)
for i in range(y_proba.shape[1]):
    zbior_niesamodzielny_pomoc[f'Prawdopodobieństwo {mapowanie[i]}'] = y_proba[:, i]
zbior_niesamodzielny_pomoc[f'Proponowany rodzaj (indeks)'] = y_pred
zbior_niesamodzielny_pomoc[f'Proponowany rodzaj'] = (
    zbior_niesamodzielny_pomoc[(f'Proponowany rodzaj (indeks)')].map(mapowanie))
print(zbior_niesamodzielny_pomoc['Proponowany rodzaj'].value_counts())

# zapis do pliku
# zbior_niesamodzielny_pomoc.to_csv('prawdziwe_propozycje.csv', header=True, index=False, sep=';',
#                                   encoding='utf-8-sig')