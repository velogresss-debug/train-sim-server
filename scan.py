# check.py
import pandas as pd

columns = [
    "timestamp", "core_temp", "temp_delta", "water_level",
    "steam_pressure", "throttle", "vent_open", "rods_pos",
    "current_speed", "distance", "reward", "action_taken"
]

df = pd.read_csv("train_telemetry_data.csv", names=columns, header=0)
df = df[df["action_taken"] != "cooldown"]

print(f"Рядків після чистки: {len(df)}")
print(f"\nРозподіл дій:")
print(df["action_taken"].value_counts())
print(f"\nДіапазони значень:")
print(df[["core_temp", "steam_pressure", "throttle", "rods_pos", "speed" if "speed" in df.columns else "current_speed"]].describe())