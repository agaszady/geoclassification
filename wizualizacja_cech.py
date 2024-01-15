# wizualizacja rozkłądu i korelacji cech w zbiorze
import pandas as pd
import numpy as np
from scipy.stats import randint, norm, gamma, skew, kurtosis, mode
import matplotlib.pyplot as plt
import seaborn as sns

def wizualizacja(zbior_samodzielny):
    # print(zbior_samodzielny)
    x_samodzielny = zbior_samodzielny.drop('rodzaj', axis=1)
    print(x_samodzielny.columns.tolist())
    # Histogramy dla wszystkich 40 cech
    # plt.rcParams.update({'font.size': 26})
    fig, axes = plt.subplots(nrows=5, ncols=8, figsize=(40, 30))

    for ax, column in zip(axes.flatten(), x_samodzielny.columns.tolist()):
        n, bins, patches = ax.hist(x_samodzielny[column], bins=50,
                                   density=True, alpha=0.7)
        ax.set_title(column)
        # dodatkowa krzywa rozkładu normalnego
        mu, std = norm.fit(x_samodzielny[column])
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2, label='Krzywa rozkładu normalnego')
        # oznaczenia dla średniej, mediany
        ax.axvline(mu, color='r', linestyle='dashed', linewidth=2, label='Średnia')
        ax.axvline(x_samodzielny[column].median(), color='g',
                   linestyle='dashed', linewidth=2, label='Mediana')
        # dominanta
        dominant_value = mode(x_samodzielny[column]).mode
        ax.axvline(dominant_value, color='b', linestyle='dashed', linewidth=2, label='Dominanta')
        lines_labels = ax.get_legend_handles_labels()

    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.tight_layout()
    # print(lines_labels)
    fig.legend(lines_labels[0], lines_labels[1], loc='lower right')
    # plt.savefig('wizualizacja/histogramy.png', bbox_inches='tight')

    # Oblicz macierz korelacji
    correlation_matrix_wew = x_samodzielny.corr(method='spearman')
    correlation_matrix = zbior_samodzielny.corr()['rodzaj'].drop('rodzaj')
    print(correlation_matrix)

    # heatmap - mapa ciepła
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix_wew, cmap='coolwarm', fmt=".2f")
    plt.tight_layout()
    # plt.savefig('wizualizacja/heatmap.png', bbox_inches='tight')
    # korelacja ze zmienną objaśnianą
    plt.figure(figsize=(12, 10))
    correlation_matrix.plot(kind='bar')
    # plt.savefig('wizualizacja/zmienne.png', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # dodatkowe miary
    zmienne_objasniajace = x_samodzielny
    summary_stats = pd.DataFrame(columns=['Średnia', 'Mediana', 'Dominanta', 'Wariancja',
                                          'Odchylenie standardowe',
                                          'Skośność', 'Kurtoza'])
    for column in zmienne_objasniajace.columns:
        mean_value = zmienne_objasniajace[column].mean()
        median_value = zmienne_objasniajace[column].median()
        mode_value = zmienne_objasniajace[column].mode().iloc[0]  # Może być więcej niż jedna moda, bierzemy pierwszą
        variance_value = zmienne_objasniajace[column].var()
        std_dev_value = zmienne_objasniajace[column].std()
        skewness_value = skew(zmienne_objasniajace[column])
        kurtosis_value = kurtosis(zmienne_objasniajace[column])

        # dodanie wyników do DataFrame
        summary_stats = summary_stats._append({'Średnia': mean_value,
                                              'Mediana': median_value,
                                              'Dominanta': mode_value,
                                              'Wariancja': variance_value,
                                              'Odchylenie standardowe': std_dev_value,
                                              'Skośność': skewness_value,
                                              'Kurtoza': kurtosis_value},
                                             ignore_index=True)
    # dodanie nazw kolumn/cech
    summary_stats['Zmienna'] = zmienne_objasniajace.columns

    summary_stats.set_index('Zmienna', inplace=True)
    # zapis do pliku
    # summary_stats.to_csv('stats.csv', sep=';', header=True)
    print(summary_stats)