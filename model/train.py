from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import numpy as np
import pathlib

DATA_PATH = pathlib.Path('data/PremierLeague100.csv')
print('Cargando dataset:', DATA_PATH)

df = pd.read_csv(DATA_PATH)

# Features históricos por equipo
df['avg_points_last3'] = df.groupby('team')['points'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
df['std_points_last3'] = df.groupby('team')['points'].rolling(3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
df['avg_gf_last3'] = df.groupby('team')['gf'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
df['avg_ga_last3'] = df.groupby('team')['ga'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
df['avg_gd_last3'] = df['avg_gf_last3'] - df['avg_ga_last3']
df['avg_position_last3'] = df.groupby('team')['position'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)

# Codificar equipo
teams = sorted(df['team'].unique())
team_to_enc = {team: idx for idx, team in enumerate(teams)}
df['team_encoded'] = df['team'].map(team_to_enc)

# Columnas de salida
target_cols = ['won', 'drawn']

# Columnas de features
feature_cols = [
    'season_end_year',
    'team_encoded',
    'played',
    'avg_points_last3',
    'std_points_last3',
    'avg_gf_last3',
    'avg_ga_last3',
    'avg_gd_last3',
    'avg_position_last3'
]

X = df[feature_cols]
y = df[target_cols]

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Entrenando modelo...')
base_model = GradientBoostingRegressor(n_estimators=500, max_depth=8, random_state=42)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

print('Guardando modelo y artefactos...')
dump(model, pathlib.Path('model/football-v4.joblib'))
dump({
    'feature_cols': feature_cols,
    'target_cols': target_cols,
    'team_classes': teams
}, pathlib.Path('model/artifacts-v4.joblib'))

print('Entrenamiento completado. Modelo listo en model/football-v4.joblib')
