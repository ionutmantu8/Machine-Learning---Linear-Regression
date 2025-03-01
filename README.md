# Regresie Liniara cu Regularizare

Acest proiect implementeaza un model de regresie liniara cu regularizare (Ridge Regression) utilizand NumPy si Pandas. Modelul antreneaza parametrii folosind Gradient Descent si aplica normalizare Z-score pentru a imbunatati performanta antrenamentului.

## Structura Proiectului

Proiectul este impartit in 4 fisiere:

1. **cost_function.py** - Contine functiile pentru calcularea costului cu regularizare.
2. **gradient.py** - Contine functiile pentru calcularea gradientului si algoritmul de Gradient Descent.
3. **normalization.py** - Include functia de normalizare Z-score.
4. **main.py** - Scriptul principal care antreneaza modelul si face predictii.

## Instalare si Configurare

### Cerinte
- Python 3.x
- NumPy
- Pandas

### Instalare
```bash
pip install numpy pandas
```

## Utilizare

1. **Preprocesarea Datelor:**
   - Se normalizeaza caracteristicile folosind Z-score.
   - Se inverseaza semnul pentru etajele si varsta casei pentru a reflecta corect influenta asupra pretului.

2. **Antrenarea Modelului:**
   - Se initializeaza parametrii w si b.
   - Se ruleaza Gradient Descent pentru a optimiza parametrii.

3. **Predictii:**
   - Se normalizeaza noile date de intrare.
   - Se aplica modelul antrenat pentru a face predictii.
   - Se denormalizeaza rezultatul pentru a obtine valoarea reala a pretului.

### Exemplu de rulare
```bash
python main.py
```

## Rezultate
Modelul afiseaza costul la fiecare iteratie si face predictii pentru datele de test.


---
