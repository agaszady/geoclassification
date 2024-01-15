# analiza zbioru danych, wizualizacja, trenowanie modeli
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, \
                             ConfusionMatrixDisplay, classification_report, \
                             balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold, cross_val_score
from scipy.stats import randint, norm, gamma, skew, kurtosis, mode
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from imblearn.metrics import specificity_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, CondensedNearestNeighbour, NearMiss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
import joblib
from wizualizacja_cech import wizualizacja

# pobranie zbioru danych
dane_warminskie = pd.read_csv('warminskie_dane_do_analizy.csv', sep=';')

''' ---------------------------- ANALIZA ZBIORU DANYCH ---------------------------- '''

# badanie występowania braków danych
na = dane_warminskie[dane_warminskie.isnull().any(axis=1)]
# print(na)
# zbadanie, dla jakich rekordów braki danych są najbardziej rozległe
wielkosc_i_forma = dane_warminskie.iloc[:, :19]
na2 = wielkosc_i_forma[wielkosc_i_forma.isnull().any(axis=1)]
# print(na2)
set_na2 = set(na2['SIMC_id'])
# usunięcie najbardziej wybrakownych rekordów
dane_warminskie_do_analizy = dane_warminskie[~dane_warminskie['SIMC_id'].isin(set_na2)]
# uzupełnienie pozostałych braków zerami
dane_warminskie_do_analizy = dane_warminskie_do_analizy.fillna(0)
# print(dane_warminskie_do_analizy)
'''
# próby ręcznego usunięcia cech
dane_warminskie_do_analizy = (
    dane_warminskie_do_analizy.drop(['srednia_odleglosci', 'min_odleglosc',
                            'max_odleglosc', 'powierzchnia_terenow_uprawnych',
                            'powierzchnia_terenow_zalesionych', 'liczba_adresow',
                            'liczba_budynkow'], axis=1))

dane_warminskie_do_analizy = dane_warminskie_do_analizy.drop(['budynkiODwochMieszkaniach',
                                                    'budynkiOTrzechIWiecejMieszkaniach',
                                                    'budynkiPrzemyslowe',
                                                    'budynkiSzkolIInstytucjiBadawczych',
                                                    'budynekZabytkowy', 'budynkiBiurowe',
                                                    'budynkiGarazy', 'budynkiHandlowoUslugowe',
                                                    'budynkiHoteli', 'budynkiKultuReligijnego',
                                                    'budynkiKulturyFizycznej',
                                                    'budynkiLacznosciDworcowITerminali',
                                                    'budynkiMuzeowIBibliotek',
                                                    'budynkiSzpitaliIZakladowOpiekiMedycznej',
                                                    'budynkiZakwaterowaniaTurystycznegoPozostale',
                                                    'budynkiZbiorowegoZamieszkania',
                                                    'ogolnodostepneObiektyKulturalne',
                                                    'pozostaleBudynkiNiemieszkalne',
                                                    'zbiornikSilosIBudynkiMagazynowe'], axis=1)
'''

# ograniczenie zbioru danych do miejscowości samodzielnych
zbior_samodzielny = (
    dane_warminskie_do_analizy)[dane_warminskie_do_analizy['czy_samodzielna'] == True]
zbior_niesamodzielny = (
    dane_warminskie_do_analizy)[dane_warminskie_do_analizy['czy_samodzielna'] == False]
zbior_samodzielny = (
    zbior_samodzielny)[zbior_samodzielny['rodzaj'].isin({'wieś', 'osada', 'osada leśna', 'kolonia'})]
# usunięcie pomocniczych kolumn
zbior_samodzielny = zbior_samodzielny.drop(['SIMC_id', 'nazwa', 'czy_samodzielna'], axis=1)
mapowanie = \
    {'wieś': 0,
     'osada': 1,
     'osada leśna': 2,
     'kolonia': 3}
zbior_samodzielny['rodzaj'] = zbior_samodzielny['rodzaj'].map(mapowanie)

''' ---------------------------- WIZUALIZACJA ---------------------------- '''
wizualizacja(zbior_samodzielny)

''' ---------------------------- MODELE ---------------------------- '''

# parametry
standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()
smote = SMOTE(random_state=42, k_neighbors=3)
adasyn = ADASYN(random_state=42)

# słowniki — najlepsze wersje klasyfikatorów
KNN = {'nazwa': 'KNN',
       'model': KNeighborsClassifier(n_neighbors=3),
       'normalizacja': standard_scaler,
       'over_samp': smote}

SVM = {'nazwa': 'SVM',
       'model': svm.SVC(kernel='rbf', decision_function_shape='ovr', random_state=42, probability=True),
       'normalizacja': standard_scaler,
       'over_samp': adasyn}

RF = {'nazwa': 'RF',
      'model': RandomForestClassifier(random_state=42, n_estimators=100),
      'normalizacja': min_max_scaler,
      'over_samp': adasyn}

GB = {'nazwa': 'GB',
      'model': GradientBoostingClassifier(random_state=42, n_estimators=200),
      'normalizacja': min_max_scaler,
      'over_samp': smote}

def train_model(zbior_samodzielny, model_dict, rfe=False, over_samp=False,
          znaczenie_cech=False, nazwa_pliku_modelu=False,
          plik_wynikowy=False):
    nazwa = model_dict['nazwa']
    model = model_dict['model']
    scaler = model_dict['normalizacja']
    os_method = model_dict['over_samp']
    # normalizacja
    X = zbior_samodzielny.drop('rodzaj', axis=1)
    y = zbior_samodzielny['rodzaj'].values
    X_scaled = scaler.fit_transform(X)
    # X_scaled = X

    zbior_samodzielny = pd.DataFrame(X_scaled, columns=X.columns)
    zbior_samodzielny['rodzaj'] = y

    # train test split
    train_df, test_df = train_test_split(zbior_samodzielny,
                                         test_size=0.3, stratify=zbior_samodzielny['rodzaj'],
                                         random_state=42)

    # print(train_df['rodzaj'].value_counts())
    # print(test_df['rodzaj'].value_counts())

    x_train = train_df.drop('rodzaj', axis=1)
    y_train = train_df['rodzaj']
    x_test = test_df.drop('rodzaj', axis=1)
    y_test = test_df['rodzaj']

    X_resampled, y_resampled = x_train, y_train
    if over_samp:
        X_resampled, y_resampled = os_method.fit_resample(x_train, y_train)

    model.fit(X_resampled, y_resampled)
    model.fit(X_resampled, y_resampled)
    model_wytrenowany = model

    if rfe:
        rfe = RFE(model, n_features_to_select=20)
        rfe.fit(X_resampled, y_resampled)
        model_wytrenowany = rfe
    y_pred = model_wytrenowany.predict(x_test)
    # y_proba = model_wytrenowany.predict_proba(x_test)

    # zapis modelu do pliku
    if nazwa_pliku_modelu:
        joblib.dump(model_wytrenowany, nazwa_pliku_modelu)
    reversefactor = dict(zip(range(5),mapowanie.keys()))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)
    pd.crosstab(y_test, y_pred, rownames=['Prawdziwy rodzaj'], colnames=['Przewidywany rodzaj'])

    precision_per_class = precision_score(y_test, y_pred, labels=list(mapowanie.keys()), average=None)
    recall_per_class = recall_score(y_test, y_pred, labels=list(mapowanie.keys()), average=None)
    specificity_per_class = specificity_score(y_test, y_pred, labels=list(mapowanie.keys()), average=None)
    f1_per_class = f1_score(y_test, y_pred, labels=list(mapowanie.keys()), average=None)
    stats = pd.DataFrame(columns=['Klasa', 'Precyzja', 'Czulosc', 'Specyficznosc', 'F1 Score'])
    stats = stats._append({'Klasa': f'{nazwa} {scaler} {os_method}'},
                 ignore_index=True)
    for cls, precision, recall, specificity, f1_score2 in (
            zip(mapowanie.keys(), precision_per_class, recall_per_class, specificity_per_class, f1_per_class)):
        stats = stats._append({'Klasa': cls,
                                'Precyzja': round(precision, 3),
                                'Czulosc': round(recall, 3),
                                'Specyficznosc': round(specificity, 3),
                                'F1 Score': round(f1_score2, 3)},
                                ignore_index=True)
    stats = stats._append({'Klasa': 'Dokladnosc',
                           'Precyzja': round(accuracy_score(y_test, y_pred), 3)},
                          ignore_index=True)
    stats = stats._append({'Klasa': 'Balanced Accuracy',
                           'Precyzja': round(balanced_accuracy_score(y_test, y_pred), 3)},
                          ignore_index=True)
    # plik csv z wynikami
    if plik_wynikowy:
        stats.to_csv(plik_wynikowy, mode='a', header=True, index=False, sep=';')

    if znaczenie_cech:
        znaczenie_cech = pd.DataFrame(columns=['cecha', 'waznosc'])
        for feature, importance in zip(x_train.columns, model_wytrenowany.feature_importances_):
            znaczenie_cech = znaczenie_cech._append({'cecha': feature,
                                                     'waznosc': round(importance, 3)},
                                                    ignore_index=True)
        znaczenie_cech.to_csv(f'wyniki/znaczenie_cech_{nazwa}.csv', mode='a', header=True,
                              index=False,
                              sep=';')

# wywołanie funkcji trenującej podany klasyfikator
# zbior_samodzielny - zbiór danych po przygotowaniu
# wybrany model — jeden ze słowników: KNN, SVM, RF, GB
# rfe — True/False, czy stosować rekursywną eliminację cech, tylko dla RF i GB, domyślnie False
# over_samp - True/False, czy stosować nadpróbkowanie, domyślnie False
# znaczenie_cech - True/False, czy generować plik ze znaczeniem cech, tylko dla RF i GB,
# domyślnie False
# nazwa_pliku_modelu - nazwa pliku *.joblib do zapisania wytrenowanego modelu, domyślnie False
# plik_wynikowy - nazwa pliku *.csv do zapisu wyników modelu, domyślnie False

# NALEŻY WYBRAĆ JEDEN Z MODELI
wybrany_model = GB

train_model(zbior_samodzielny, wybrany_model, over_samp=True, znaczenie_cech=True,
            plik_wynikowy='wyniki/wyniki_gb.csv', nazwa_pliku_modelu='modele/gb_over_samp.joblib')