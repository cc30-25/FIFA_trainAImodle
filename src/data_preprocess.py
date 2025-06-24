import pandas as pd
import numpy as np
import json
import random
from src import config
from src.data_loader import load_data
from sklearn.preprocessing import StandardScaler

def drop_highly_correlated_features(df, threshold=0.95):
    """
    Drop highly correlated features from the dataframe.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print("Dropping highly correlated features:", to_drop)
    return df.drop(to_drop)

def process_data(data):
    """
    Processs the data to extract static features and labels.
    """
    pl = data["play_log"].copy()
    pl = pl.dropna(subset=["outcome"])

    # Add random team names by getting Team col in per_game_stats
    if "team_a" not in pl.columns:
        pgs = data["per_game_stats"].copy()
        team_a = pgs["Team"].sample().values[0]
        team_b = pgs["Team"].sample().values[0]
        pl["team_a"] = team_a
        pl["team_b"] = team_b

    advanced = data["advanced_data"].copy()
    advanced.columns = advanced.columns.str.strip() # Remove leading/trailing whitespaces
    # print("Advanced data columns:", advanced.columns.tolist())

    try:
        pl = pl.merge(advanced[['Team', 'ORtg', 'DRtg', 'Pace']],
                    left_on='team_a', right_on='Team', how='left')
        pl.rename(columns={'ORtg': 'team_a_ORtg', 'DRtg': 'team_a_DRtg', 'Pace': 'team_a_Pace'}, inplace=True)

        pl = pl.merge(advanced[['Team', 'ORtg', 'DRtg', 'Pace']],
                        left_on='team_b', right_on='Team', how='left')
        pl.rename(columns={'ORtg': 'team_b_ORtg', 'DRtg': 'team_b_DRtg', 'Pace': 'team_b_Pace'}, inplace=True)
    except KeyError as e:
        print("Error in merging advanced data:", e)
        raise

    # Calculate the time remaining in the game
    GAME_TIME = 48 * 60
    if "time_remaining" not in pl.columns:
        pl["time_remaining"] = GAME_TIME - pl["timestamp"]

    if "score_differential" not in pl.columns:
        pl["score_differential"] = abs(pl['score_team_a'] - pl['score_team_b'])

    # calculate ball position
    pl["ball_x"] = pl["basketball"].apply(lambda x: eval(x)["x"] if isinstance(x, str) else x["x"])
    pl["ball_y"] = pl["basketball"].apply(lambda x: eval(x)["y"] if isinstance(x, str) else x["y"])

    # Spatial features
    court_center_x = 94 / 2
    court_center_y = 50 / 2
    pl["ball_dis_to_center"] = np.sqrt((pl["ball_x"] - court_center_x) ** 2 + (pl["ball_y"] - court_center_y) ** 2)

    # Game state
    states = ["time_remaining", "score_differential", "ball_x", "ball_y", "ball_dis_to_center", 
              "team_a_ORtg", "team_a_DRtg", "team_a_Pace", "team_b_ORtg", "team_b_DRtg", "team_b_Pace"]
    df_features = pd.DataFrame(pl[states].values, columns=states)

    # Drop highly correlated features
    df_features_reduced = drop_highly_correlated_features(df_features, threshold=0.95)
    x_static = df_features_reduced.values.astype(np.float32)

    # Label the outcomes
    result_map = {
        "team_a three pointer": 0,
        "team_b three pointer": 1,
        "team_a dunk": 2,
        "team_b dunk": 3,
        "team_a layup": 4,
        "team_b layup": 5,
        "team_a steal": 6,
        "team_b steal": 7,
        "miss": 8,
    }
    pl["label"] = pl["outcome"].map(
        lambda x: result_map.get(x, -1)
    )
    # Filter out rows with missing labels
    pl = pl[pl["label"] != -1]
    y = pl["label"].values.astype(np.int32)

    # Scale the x_static features
    scaler = StandardScaler()
    x_static = scaler.fit_transform(x_static)

    return x_static, y, result_map

def preprocess_sequential_data(data):
    x_static, y, result_map = process_data(data)
    x_seq = np.repeat(x_static[:, np.newaxis, :], config.HISTORICAL_EVENT, axis=1)
    return x_static, x_seq, y, result_map

if __name__ == "__main__":
    data = load_data()
    x_static, x_seq, y, result_map = preprocess_sequential_data(data)
    print("Static features shape:", x_static.shape)
    print("Sequential features shape:", x_seq.shape)
    print("Labels shape:", y.shape)

    
    