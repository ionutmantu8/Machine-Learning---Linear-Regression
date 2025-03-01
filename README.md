# Regresie Liniară cu Regularizare

Acest proiect implementează un model de regresie liniară cu regularizare (Ridge Regression) utilizând NumPy și Pandas. Modelul antrenează parametrii folosind Gradient Descent și aplică normalizare Z-score pentru a îmbunătăți performanța antrenamentului.

## Structura Proiectului

Proiectul este împărțit în 4 fișiere:

1. **cost_function.py** - Conține funcțiile pentru calcularea costului cu regularizare.
2. **gradient.py** - Conține funcțiile pentru calcularea gradientului și algoritmul de Gradient Descent.
3. **normalization.py** - Include funcția de normalizare Z-score.
4. **main.py** - Scriptul principal care antrenează modelul și face predicții.

## Instalare și Configurare

### Cerințe
- Python 3.x
- NumPy
- Pandas

### Instalare
```bash
pip install numpy pandas
```

## Utilizare

1. **Preprocesarea Datelor:**
   - Se normalizează caracteristicile folosind Z-score.
   - Se inversează semnul pentru etajele și vârsta casei pentru a reflecta corect influența asupra prețului.

2. **Antrenarea Modelului:**
   - Se inițializează parametrii w și b.
   - Se rulează Gradient Descent pentru a optimiza parametrii.

3. **Predicții:**
   - Se normalizează noile date de intrare.
   - Se aplică modelul antrenat pentru a face predicții.
   - Se denormalizează rezultatul pentru a obține valoarea reală a prețului.

### Exemplu de rulare
```bash
python main.py
```

## Rezultate
Modelul afișează costul la fiecare iterație și face predicții pentru datele de test.


---
