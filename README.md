# Linear Regression with Regularization  

This project implements a linear regression model with regularization (Ridge Regression) using NumPy and Pandas. The model trains its parameters using Gradient Descent and applies Z-score normalization to improve training performance. The algorithm aims to predict house prices, with input features structured as `[a, b, c, d]`, where:  
- `a` = house area  
- `b` = number of bedrooms  
- `c` = number of floors  
- `d` = years since construction  

## Project Structure  

The project is divided into four files:  

1. **cost_function.py** - Contains functions for computing the cost with regularization.  
2. **gradient.py** - Includes functions for computing the gradient and implementing Gradient Descent.  
3. **normalization.py** - Implements Z-score normalization.  
4. **main.py** - The main script for training the model and making predictions.  

## Installation & Setup  

### Requirements  
- Python 3.x  
- NumPy  
- Pandas  

### Installation  
```bash
pip install numpy pandas
