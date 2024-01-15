import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


# pobranie zbioru danych
dane_warminskie = pd.read_csv('../warminskie_dane_do_analizy.csv', sep=';')


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
# normalizacja
# knn i svm - normalizacja
# gradient i rf - min-max
X = zbior_samodzielny.drop('rodzaj', axis=1)
y = zbior_samodzielny['rodzaj'].values
scaler1 = StandardScaler()
scaler2 = MinMaxScaler()
X_scaled1 = scaler1.fit_transform(X)
X_scaled2 = scaler2.fit_transform(X)
# X_scaled = X

zbior_samodzielny1 = pd.DataFrame(X_scaled1, columns=X.columns)
zbior_samodzielny1['rodzaj'] = y
zbior_samodzielny2 = pd.DataFrame(X_scaled2, columns=X.columns)
zbior_samodzielny2['rodzaj'] = y

# train test split
_, test_df1 = train_test_split(zbior_samodzielny1,
                                     test_size=0.3, stratify=zbior_samodzielny1['rodzaj'],
                                     random_state=42)

x_test1 = test_df1.drop('rodzaj', axis=1)
y_test1 = test_df1['rodzaj']

_, test_df2 = train_test_split(zbior_samodzielny2,
                                     test_size=0.3, stratify=zbior_samodzielny2['rodzaj'],
                                     random_state=42)

x_test2 = test_df2.drop('rodzaj', axis=1)
y_test2 = test_df2['rodzaj']

model_knn_oversamp = joblib.load('../modele/knn_over_samp.joblib')
y_test1_bin = label_binarize(y_test1, classes=list(set(y_test1)))
# Użyj OneVsRestClassifier, aby dostosować do wielu klas
classifier = OneVsRestClassifier(model_knn_oversamp)

# Pobierz prawdopodobieństwa predykcji dla każdej klasy
probabilities_model_knn_oversamp = classifier.predict_proba(x_test1)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(set(y_test1))):
    fpr[i], tpr[i], _ = roc_curve(y_test1_bin[:, i], probabilities_model_knn_oversamp[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(len(set(y_test1))):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# Dodaj legende i etykiety osi
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.legend(loc='lower right')
plt.xlabel('False Positive')
plt.ylabel('True Positive')

# Wyświetl wykres
plt.show()