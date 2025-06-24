import os
import torch


# Constants
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
DROPOUT = 0.5
HISTORICAL_EVENT = 10      # Number of past events to consider   
VALIDATION_SIZE = 0.15
SEED = 42


# Base directory to data folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "log")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

PLAY_LOG = os.path.join(DATA_DIR, "synthetic_play_log.csv")
TRACKING = os.path.join(DATA_DIR, "synthetic_tracking.csv")
ADVANCED_DATA = os.path.join(DATA_DIR, "advanced_data.csv")
PER_100_POSS = os.path.join(DATA_DIR, "per_100_poss.csv")
PER_GAME_STATS = os.path.join(DATA_DIR, "per_game_stats.csv")
PLAYER_STATS = os.path.join(DATA_DIR, "player_stats_data.csv")
PLAYER_META = os.path.join(DATA_DIR, "player_meta_data.csv")
SHOOTING_STATS = os.path.join(DATA_DIR, "shooting_stats.csv")
PLAYER_PROCESSED_DATA = os.path.join(DATA_DIR, "nba_data_processed.csv")
EVENTS_FILE = os.path.join(DATA_DIR, "nba_data.json")

MODEL_NAME = os.path.join(MODEL_DIR, "predictor.h5")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_PATH = os.path.join(MODEL_DIR, "nf_model")
# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
