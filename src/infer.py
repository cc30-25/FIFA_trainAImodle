import json
import torch
import numpy as np
import random
from src import config
from src.model import EventPredictorWithSeq

def process_event(event):
    """
    Process the event data to get the static and sequential features.
    """
    time_remaining = random.uniform(0, 60 * 48)  # 48 minutes
    score_differential = random.randint(-20, 20)
    ball_x = event["basketball"]["x"]
    ball_y = event["basketball"]["y"]
    court_center_x = 94 / 2
    court_center_y = 50 / 2
    ball_dis_to_center = np.sqrt((ball_x - court_center_x) ** 2 + (ball_y - court_center_y) ** 2)

    team_a_ORtg = random.randint(100, 120)
    team_a_DRtg = random.randint(100, 120)
    team_a_Pace = random.randint(90, 110)
    team_b_ORtg = random.randint(100, 120)
    team_b_DRtg = random.randint(100, 120)
    team_b_Pace = random.randint(90, 110)

    static_features = np.array([
        time_remaining, score_differential, ball_x, ball_y, ball_dis_to_center,
        team_a_ORtg, team_a_DRtg, team_a_Pace, team_b_ORtg, team_b_DRtg, team_b_Pace
    ], dtype=np.float32)

    return static_features

def prev_event_window(events, current_event_idx, window_size):
    """
    Get the previous events in a window of size `window_size`.
    """
    window_features = []
    start = max(0, current_event_idx - window_size + 1)
    for j in range(start, current_event_idx + 1):
        features = process_event(events[j])
        window_features.append(features)

    while len(window_features) < window_size:
        window_features.insert(0, window_features[0])
    
    return np.array(window_features)

def main():
    events = json.load(open(config.EVENTS_FILE, "r"))
    # Load the model
    temp_static_dim = 11
    temp_seq_dim = 11
    temp_num_classes = 9
    model = EventPredictorWithSeq(temp_static_dim, temp_seq_dim, temp_num_classes)
    model.load_state_dict(torch.load(config.MODEL_NAME, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    result_map = {
        0: "team_a three pointer",
        1: "team_b three pointer",
        2: "team_a dunk",
        3: "team_b dunk",
        4: "team_a layup",
        5: "team_b layup",
        6: "team_a steal",
        7: "team_b steal",
        8: "miss"
    }

    window_size = config.HISTORICAL_EVENT
    predictions = []

    for i, event in enumerate(events):
        static_features = process_event(event)
        x_seq_array = prev_event_window(events, i, window_size)
        x_static = static_features.reshape(1, -1)
        x_seq = x_seq_array.reshape(1, window_size, -1)

        x_static_tensor = torch.tensor(x_static, dtype=torch.float32, device=config.DEVICE)
        x_seq_tensor = torch.tensor(x_seq, dtype=torch.float32, device=config.DEVICE)

        with torch.no_grad():
            logits = model(x_static_tensor, x_seq_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        predictions.append({
            "event_index": i,
            "predicted_class": pred,
            "predicted_event": result_map[pred],
            "probabilities": probs.cpu().numpy().tolist()[0]
        })

    for p in predictions:
        print(f"Event Index: {p['event_index']}: {p['predicted_event']} (class {p['predicted_class']}) with probabilities: {p['probabilities']}")
        # Save to a output file
        with open("predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)

if __name__ == "__main__":
    main()