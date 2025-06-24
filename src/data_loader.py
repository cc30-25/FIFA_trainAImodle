import pandas as pd
from src import config

def load_data():
    play_log = pd.read_csv(config.PLAY_LOG)
    tracking = pd.read_csv(config.TRACKING)
    advanced_data = pd.read_csv(config.ADVANCED_DATA)
    per_100_poss = pd.read_csv(config.PER_100_POSS)
    per_game_stats = pd.read_csv(config.PER_GAME_STATS)
    player_stats = pd.read_csv(config.PLAYER_STATS)
    player_meta = pd.read_csv(config.PLAYER_META)
    shooting_stats = pd.read_csv(config.SHOOTING_STATS)
    player_processed_data = pd.read_csv(config.PLAYER_PROCESSED_DATA)

    return {
        "play_log": play_log,
        "tracking": tracking,
        "advanced_data": advanced_data,
        "per_100_poss": per_100_poss,
        "per_game_stats": per_game_stats,
        "player_stats": player_stats,
        "player_meta": player_meta,
        "shooting_stats": shooting_stats,
        "player_processed_data": player_processed_data
    }

if __name__ == "__main__":
    data = load_data()
    for key, value in data.items():
        print(f"{key}: {value.shape}")
        