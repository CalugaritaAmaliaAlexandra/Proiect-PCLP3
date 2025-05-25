import pandas as pd
import numpy as np

# Functie care genereaza datele unui data frame cu n exemple de pacienti
# caracterizati prin: varsta, sex, greutate, stil de viata, istoric familial,
# daca fumeaza sau nu, daca prezinta simptome de febra sau nu, daca consuma
# fast-food sau nu si eticheta (sick).
def generate_data(n):
    # Setam un seed (se vor genera aceleasi valori aleatorii la fiecare rulare)
    np.random.seed(2025)  # Setam seed pentru reproducibilitate
    # Generam coloana age cu valori intregi intre 18 si 90 pentru n pacienti.
    age = np.random.randint(18, 90, size=n)
    # Generam aleator coloana sex cu valori aleatorii (50% barbati, 50% femei)
    # pentru n pacienti.
    sex = np.random.choice(['male', 'female'], size=n, p=[0.5, 0.5])
    # Generam coloana weight cu valori aleatorii (distributie normala cu media
    # de 70 kg si deviatie standard de 10 kg si cu valori intre 45 si 100)
    # pentru n pacienti. Rotunjim la 2 zecimale.
    weight = np.round(np.clip(np.random.normal(70, 10, size=n), 45, 100), 2)
    # Generam variabilele binare (smokes, fever, family_history,
    # eats_fast_food) cu valori aleatorii (0 sau 1) pentru n pacienti.
    # 40% sanse sa fie fumator
    smokes = np.random.binomial(1, 0.4, size=n)
    # Generam coloana fever cu valori aleatorii intre 36.0 si 40.0 grade
    fever = np.round(np.random.uniform(36.0, 40.0, size=n), 1)
    # 30 % sanse sa aiba istoric familial
    family_history = np.random.binomial(1, 0.3, size=n)
    # 40 % sanse sa consume fast-food
    eats_fast_food = np.random.binomial(1, 0.4, size=n)
    # Generam datele pentru coloana lifestyle pentru n persoane: 25% activ,
    # 40% moderat, 35% sedentar.
    lifestyle = np.random.choice(['active', 'moderate', 'sedentary'], size=n, p=[0.25, 0.4, 0.35])
    # Adaugam valori aberante pentru coloanele weight (valori mari) si fever
    # (valori mici)
    # Pentru age adaugam 10 astfel de valori
    outlier_indices_weight = np.random.choice(n, size=10, replace=False)
    weight[outlier_indices_weight] = np.round(np.random.uniform(110, 150, size=10), 2)
    # Pentru fever adaugam 10 astfel de valori
    outlier_indices_fever = np.random.choice(n, size=5, replace=False)
    fever[outlier_indices_fever] = np.round(np.random.uniform(32, 35, size=5), 1)
    outlier_indices_fever = np.random.choice(n, size=5, replace=False)
    fever[outlier_indices_fever] = np.round(np.random.uniform(41, 43, size=5), 1)
    # Calculam riscul de febra: 1 daca temperatura > 38 sau < 35, altfel 0
    fever_risk = ((fever > 38) | (fever < 36)).astype(int)
    # Mapam stilul de viata in scoruri de rist : active = 0, moderate = 0.5,
    # sedentary = 1
    lifestyle_risk_map = {'active': 0, 'moderate': 0.5, 'sedentary': 1}
    lifestyle_risk = np.vectorize(lifestyle_risk_map.get)(lifestyle)
    # Calculam greutatea considerata factor de risc (1 dacă > 85 kg, altfel 0)
    overweight = (weight > 85).astype(int)
    # Setam regula de etichetare: un pacient este considerat "bolnav" daca suma
    # factorii de risc (smokes, fever, family_history, eats_fast_food,
    # overweight, lifestyle) luate in anumite proportii > = 1.5
    sick = ((smokes + 0.4 * fever_risk + 0.5 * family_history + 0.8 * eats_fast_food + 0.5 * overweight + 0.8 * lifestyle_risk) >= 2).astype(int)
    # Adaugam erori aleatorii in eticheta pentru a obtine posibiliatatea de
    # clasificare gresita (3% din datele etichetate sunt gresite).
    wrong_idx = np.random.choice(n, size=int(0.03 * n), replace=False)
    # 0 devine 1, 1 devine 0
    sick[wrong_idx] = 1 - sick[wrong_idx] 

    # Construim DataFrame-ul cu toate variabilele
    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'weight': weight,
        'smokes': smokes,
        'fever': fever,
        'family_history': family_history,
        'eats_fast_food': eats_fast_food,
        'lifestyle': lifestyle,
        # Variabila care decide daca pacientul este bolnav sau nu (1 = bolnav,
        # 0 = sanatos)
        'sick': sick
    })
    # Adaugam valori lipsa in coloanele fever, lifestyle si family_history (5%)
    for col in ['fever', 'lifestyle', 'family_history']:
        missing_indices = np.random.choice(n, size=int(0.05 * n), replace=False)
        df.loc[missing_indices, col] = np.nan
    return df
