from generate import generate_data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Generam setul de antrenament cu 600 de pacienti
train_df = generate_data(600)
# Generam setul de test cu 200 de pacienti
test_df = generate_data(200)
# Salvam ambele seturi in fisiere CSV
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

# Cream un folder numit "grafice" in care vom stoca graficele pe care urmeaza
# sa le generam
os.makedirs("grafice", exist_ok=True)

# Functie in care analizam valorile lipsa
def missing_analysis(df, name=""):
    print(f"\nAnaliza valorilor lipsa în {name}:")
    # Verificam cate valori lipsa sunt in fiecare coloana
    missing_count = df.isnull().sum()
    # Aflam procentul de valori lipsa pe fiecare coloana
    missing_pct = (df.isnull().mean() * 100).round(2)
    # Cream un data frame cu numarul de valori lipsa si procentul pentru
    # fiecare coloana in care detectam lipsuri
    missing_df = pd.DataFrame({
        "Numar valori lipsa": missing_count,
        "Procent %": missing_pct
    })
    # Daca nu exita valori lipsa pentru nicio coloana, afisam un mesaj
    # corespunzator
    if missing_df.empty:
        print("Nu exista valori lipsa.")
    else:
        # Daca avem valori lipsa, afisam pentru fiecare coloana numarul de
        # valori lipsa si procentul
        print(missing_df)
    return missing_df

# Aplicam functia care analizeaza valorile lipsa pentru ambele seturi de date
missing_train = missing_analysis(train_df, "train")
missing_test = missing_analysis(test_df, "test")

# Functie care trateaza valorile lipsa dintr-un data frame pentru fiecare
# coloana
def complete_missing_values(df, nume_set=""):
    for col in df.columns:
        # Verificam daca exista valori lipsa in coloana respectiva
        if df[col].isnull().sum() > 0:
            # Daca exista:
            # Daca coloana contine valori binare (0 si 1), completam cu moda
            if df[col].iloc[0] in [0, 1]:
                # Calculam moda coloanei
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                print(f"{nume_set}: Valorile lipsa lipsa din coloana '{col}' au fost completate cu moda: {mode_value}")
            # Daca coloana contine valori numerice, completam cu media
            elif df[col].dtype in ['float64', 'int64']:
                # Calculam media coloanei
                mean_value = df[col].mean()
                # Completam valorile lipsa cu media
                df[col] = df[col].fillna(mean_value)
                # Afisam un mesaj cu media folosita pentru completare
                print(f"{nume_set}: Valorile lipsă din coloana '{col}' au fost completate cu media: {mean_value:.2f}")
            else:
                # Daca coloana contine stringuri sau categorii, completam cu moda
                # Calculam moda coloanei
                mode_value = df[col].mode()[0]
                # Completam valorile lipsa cu moda
                df[col] = df[col].fillna(mode_value)
                # Afisam un mesaj cu moda folosita pentru completare
                print(f"{nume_set}: Valorile lipsă din coloana '{col}' au fost completate cu moda: {mode_value}")
    return df

print("\n")
train_df = complete_missing_values(train_df, "train")
print("\n")
test_df = complete_missing_values(test_df, "test")

# Statistica descriptiva pentru seturile de date
print("\nStatistici descriptive pentru train:")
print(train_df.describe(include='all'))
print("\nStatistici descriptive pentru test:")
print(test_df.describe(include='all'))

# Realizarea histogramelor variabilelor numerice. Coloanele pe care le vom
# analiza sunt age, weight, fever
cols = ['age', 'weight', 'fever']
# Pentru setul de antrenament
for col in cols:
    # Cream histograma pentru fiecare variabila numerica
    sns.histplot(train_df[col], kde=True)
    # Setam un titlu pentru fiecare grafic
    plt.title(f"Distribuaia variabilei {col}")
    # Salvam graficul in folderul grafice
    plt.savefig(f"grafice/train_{col}_hist.png")
    plt.close()
# Pentru setul de test
for col in cols:
    sns.histplot(test_df[col], kde=True)
    plt.title(f"Distributia variabilei {col}")
    plt.savefig(f"grafice/test_{col}_hist.png")
    plt.close()

# Countplot variabile categorice
cols = ['sex', 'smokes', 'family_history', 'eats_fast_food', 'lifestyle']
# Pentru setul de antrenament
for col in cols:
    # Pentru fiecare variabila categorica, cream un countplot: pe axa x vom
    # avea valorile varibilei, iar pe axa y numarul de aparitii
    sns.countplot(data=train_df, x=col)
    plt.title(f"Distributia variabilei {col}")
    plt.savefig(f"grafice/train_{col}_count.png")
    plt.close()
# Pentru setul de test
for col in cols:
    sns.countplot(data=test_df, x=col)
    plt.title(f"Distributia variabilei {col}")
    plt.savefig(f"grafice/test_{col}_count.png")
    plt.close()

# Functie care detecteaza outlieri folosind boxplot-uri pentru coloanele age,
# weight și fever
def detect_outliers(df, nume_set="train"):
    cols = ['age', 'weight', 'fever']
    for col in cols:
        sns.boxplot(data=df, x=col)
        plt.title(f"Boxplot pentru {col}")
        plt.savefig(f"grafice/{nume_set}_{col}_box.png")
        plt.close()

detect_outliers(train_df, "train")
detect_outliers(test_df, "test")

# Matrice corelatii pentru train, variabile numerice
# Selectam doar coloanele numerice
cols = ['age', 'weight', 'fever', 'smokes', 'family_history', 'eats_fast_food', 'sick']
# Calculam corelatia doar pentru acele coloane
corelation = train_df[cols].corr()
sns.heatmap(corelation, annot=True, cmap='coolwarm')
plt.title("Matricea de corelatii")
plt.savefig("grafice/train_corelatii.png")
plt.close()

# Matrice corelatii pentru test, variabile numerice
cols = ['age', 'weight', 'fever', 'smokes', 'family_history', 'eats_fast_food', 'sick']
corelation = test_df[cols].corr()
sns.heatmap(corelation, annot=True, cmap='coolwarm')
plt.title("Matricea de corelatii")
plt.savefig("grafice/test_corelatii.png")
plt.close()

# Violinplot pentru variabilele numerice in functie de stare (sick)
# Pentru train
for col in ['age', 'weight', 'fever']:
    sns.violinplot(data=train_df, x='sick', y=col)
    plt.title(f"Distributia {col} in functie de sick")
    plt.savefig(f"grafice/train_violin_{col}_sick.png")
    plt.close()
# Pentru test
for col in ['age', 'weight', 'fever']:
    sns.violinplot(data=test_df, x='sick', y=col)
    plt.title(f"Distributia {col} in functie de sick")
    plt.savefig(f"grafice/test_violin_{col}_sick.png")
    plt.close()

# Procesarea datelor
# Combinam seturile de date pentru preprocesare
df = pd.concat([train_df, test_df])
# Folosim LabelEncoder pentru coloanele categorice si coloanele cu siruri de
# caractere
label_enc = LabelEncoder()

df['sex'] = label_enc.fit_transform(df['sex'])
df['lifestyle'] = label_enc.fit_transform(df['lifestyle'])

# Scalam coloanele numerice 
scaler = StandardScaler()
df[['age', 'weight']] = scaler.fit_transform(df[['age', 'weight']])

# Refacem seturile de antrenament si testare
train_df = df.iloc[:len(train_df)]
test_df = df.iloc[len(train_df):]
# Extragem matricea ce contine caracteristicile fiecarui pacient pentru
# antrenament
X_train = train_df.drop(columns=['sick'])
# Extragem rezultatele pentru fiecare pacient pentru antrenament
y_train = train_df['sick']
# Facem acelasi lucru pentru setul de testare
X_test = test_df.drop(columns=['sick'])
y_test = test_df['sick']

# Cream modelul de regresie logistica
model = LogisticRegression(max_iter=1000)
# Antrenam modelul pe setul de antrenament
model.fit(X_train, y_train)
# Folosim modelul pentru a face predictii pe setul de testare
y_pred = model.predict(X_test)

# Interpretarea rezultatelor
print("\nInterpretarea rezultatelor:")
print(classification_report(y_test, y_pred))

# Calculam matricea de confuzie
confusion_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Sanatos", "Bolnav"]).plot(cmap='Blues')
plt.title("Matrice de confuzie")
plt.savefig("grafice/matrice_confuzie.png")
plt.close()

# Generam un grafic care descrie erorile de clasificare
errors = y_test - y_pred
# Graficul va fi tip histograma, cu 3 bin-uri: -1, 0, 1 (-1 pentru bolnav care
# a fost prezis sanatos, 0 pentru prezis corect, 1 pentru sanatos care a fost
# prezis bolnav)
plt.hist(errors, bins=3, edgecolor='black')
plt.title("Distributia erorilor")
plt.xlabel("Eroare")
plt.ylabel("Frecventa")
plt.xticks([-1, 0, 1])
plt.savefig("grafice/erori_logistic_regression.png")
plt.close()

# Salvam predictiile intr-un fisier CSV
pd.DataFrame(y_pred, columns=["predictie"]).to_csv("predictii.csv", index=False)
