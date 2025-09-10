from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = pathlib.Path('model/football-v4.joblib')
ARTIFACTS_PATH = pathlib.Path('model/artifacts-v4.joblib')
DATA_PATH = pathlib.Path('data/PremierLeague100.csv')

app = FastAPI(title='Premier League - Position & Record Prediction (Monte Carlo)')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Cargar modelo y artefactos
model = load(MODEL_PATH)
artifacts = load(ARTIFACTS_PATH)
feature_cols = artifacts['feature_cols']
target_cols = artifacts['target_cols']
team_classes = artifacts['team_classes']
team_to_enc = {team: idx for idx, team in enumerate(team_classes)}

def _team_to_enc(team: str):
    return team_to_enc.get(team, abs(hash(team)) % 1000)

# Modelos de entrada y salida
class InputData(BaseModel):
    team: str = Field(..., example="Manchester Utd")
    season_end_year: int = Field(..., example=2026)
    simulations: int = Field(1000, example=1000, description="Número de simulaciones Monte Carlo (default 1000)")

class OutputData(BaseModel):
    position: int
    won: int
    drawn: int
    lost: int
    points: int
    champion_probability: float = Field(..., example=12.34)

@app.post('/score', response_model=OutputData)
def score(data: InputData):
    # Cargar datos históricos (tu dataset)
    df = pd.read_csv(DATA_PATH)

    # --- Construir features del equipo objetivo ---
    team_enc = _team_to_enc(data.team)
    team_df = df[df['team'] == data.team]

    if not team_df.empty:
        avg_points_last3 = team_df['points'].tail(3).mean()
        std_points_last3 = team_df['points'].tail(3).std()
        if np.isnan(std_points_last3):
            std_points_last3 = 0.0
        avg_gf_last3 = team_df['gf'].tail(3).mean()
        avg_ga_last3 = team_df['ga'].tail(3).mean()
        avg_gd_last3 = (avg_gf_last3 or 0.0) - (avg_ga_last3 or 0.0)
        avg_position_last3 = team_df['position'].tail(3).mean()
        played = int(round(team_df['played'].mean()))
    else:
        avg_points_last3 = std_points_last3 = avg_gf_last3 = avg_ga_last3 = avg_gd_last3 = avg_position_last3 = 0.0
        played = 38

    # Fila de features para predecir (determinista)
    row = [
        data.season_end_year,
        team_enc,
        played,
        avg_points_last3,
        std_points_last3,
        avg_gf_last3,
        avg_ga_last3,
        avg_gd_last3,
        avg_position_last3
    ]
    model_input = np.array(row).reshape(1, -1)

    # Predicción puntual (determinista) para el equipo solicitado
    pred = model.predict(model_input)[0]
    won_pred, drawn_pred = pred[0], pred[1]
    won, drawn = map(lambda x: max(0, int(round(x))), (won_pred, drawn_pred))
    lost = played - won - drawn
    points = won * 3 + drawn

    # --- Preparar predicciones y parámetros por cada equipo para Monte Carlo ---
    all_teams = list(df['team'].unique())
    n_teams = len(all_teams)
    n_sim = max(1, int(data.simulations))

    # Arrays para almacenar la media esperada de puntos y la desviación por equipo
    points_means = np.zeros(n_teams, dtype=np.float64)
    points_stds = np.zeros(n_teams, dtype=np.float64)

    # Si el dataset no tiene algún equipo (improbable), se usan valores por defecto basados en la liga
    global_std = float(df['points'].std()) if 'points' in df.columns and not df['points'].isnull().all() else 5.0
    global_std = max(1.0, global_std)  # mínimo ruido razonable

    for i, t in enumerate(all_teams):
        t_enc = _team_to_enc(t)
        t_df = df[df['team'] == t]
        if not t_df.empty:
            t_avg_points = t_df['points'].tail(3).mean()
            t_std_points = t_df['points'].tail(3).std()
            if np.isnan(t_std_points):
                t_std_points = 0.0
            # Predecir con el modelo (media de wins/draws -> puntos esperados)
            row_t = np.array([
                data.season_end_year,
                t_enc,
                int(round(t_df['played'].mean())) if not t_df['played'].isnull().all() else 38,
                t_avg_points,
                t_std_points,
                t_df['gf'].tail(3).mean() if not t_df['gf'].isnull().all() else 0.0,
                t_df['ga'].tail(3).mean() if not t_df['ga'].isnull().all() else 0.0,
                (t_df['gf'].tail(3).mean() or 0.0) - (t_df['ga'].tail(3).mean() or 0.0),
                t_df['position'].tail(3).mean() if not t_df['position'].isnull().all() else 0.0
            ]).reshape(1, -1)

            try:
                pred_t = model.predict(row_t)[0]
                won_mean, drawn_mean = float(pred_t[0]), float(pred_t[1])
            except Exception:
                # Si falla la predicción por cualquier razón, fallback a puntos históricos
                won_mean, drawn_mean = 10.0, 8.0

            points_mean = won_mean * 3.0 + drawn_mean
            points_means[i] = max(0.0, points_mean)
            # usar desviación histórica si existe, si no usar global_std
            points_stds[i] = max(1.0, t_std_points if t_std_points > 0 else global_std)
        else:
            # Equipo sin historial en dataset: asignar una media conservadora y desviación global
            points_means[i] = 40.0
            points_stds[i] = global_std

    # --- Monte Carlo vectorizado: generar matrix (n_teams, n_sim) de puntos simulados ---
    rng = np.random.default_rng(seed=42)  # semilla para reproducibilidad
    samples = rng.normal(loc=np.repeat(points_means[:, np.newaxis], n_sim, axis=1),
                         scale=np.repeat(points_stds[:, np.newaxis], n_sim, axis=1))
    # Asegurar valores válidos y enteros
    samples = np.clip(np.round(samples), a_min=0, a_max=None).astype(int)

    # Para cada simulación, obtener el equipo con mayor puntos (si hay empates, se elige el primero ordenado por índice)
    winners_idx = np.argmax(samples, axis=0)  # shape (n_sim,)
    # Contar campeonatos por equipo
    counts = np.bincount(winners_idx, minlength=n_teams)

    # Probabilidad de campeón para el equipo solicitado
    try:
        team_index = all_teams.index(data.team)
        champ_prob = (counts[team_index] / n_sim) * 100.0
    except ValueError:
        # equipo no está en all_teams (muy raro), prob = 0
        champ_prob = 0.0
        team_index = None

    # Calcular posición esperada del equipo en las simulaciones
    # argsort de cada columna (orden descendente) para obtener posiciones
    order = np.argsort(-samples, axis=0)  # shape (n_teams, n_sim)
    if team_index is not None:
        is_team = (order == team_index)  # boolean matrix
        # primera aparición de True en cada columna es la posición (0-indexed)
        positions_sim = np.argmax(is_team, axis=0) + 1  # posiciones 1..n_teams, shape (n_sim,)
        expected_position = int(round(np.mean(positions_sim)))
    else:
        expected_position = -1

    # Mantener la predicción puntual original para won/drawn/lost/points (para coherencia)
    return {
        "position": expected_position,
        "won": won,
        "drawn": drawn,
        "lost": lost,
        "points": points,
        "champion_probability": float(champ_prob)
    }
