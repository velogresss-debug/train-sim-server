import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import pickle

columns = [
    "timestamp", "core_temp", "temp_delta", "water_level",
    "steam_pressure", "throttle", "vent_open", "rods_pos",
    "current_speed", "distance", "reward", "action_taken"
]
df = pd.read_csv("train_telemetry_data.csv", names=columns, header=0)

df = df[df["action_taken"] != "cooldown"]
print(f"Після чистки: {len(df)} рядків")
print(f"Reward діапазон: min={df['reward'].min():.1f}, max={df['reward'].max():.1f}, mean={df['reward'].mean():.1f}")

target = 200

df_none     = resample(df[df["action_taken"] == "none"],         n_samples=target, random_state=42)
df_stop     = resample(df[df["action_taken"] == "full_stop"],    n_samples=target, random_state=42)
df_vent     = resample(df[df["action_taken"] == "vent_steam"],   n_samples=target, random_state=42)
df_cool     = resample(df[df["action_taken"] == "cool_reactor"], n_samples=target, random_state=42)
df_reduce   = resample(df[df["action_taken"] == "reduce_speed"], n_samples=target, random_state=42)

df_balanced = pd.concat([df_none, df_stop, df_vent, df_cool, df_reduce])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nЗбалансований датасет: {len(df_balanced)} рядків")
print(df_balanced["action_taken"].value_counts())

rewards = df_balanced["reward"]
r_min, r_max = rewards.min(), rewards.max()

if r_max > r_min:
    rewards_norm = (rewards - r_min) / (r_max - r_min)
else:
    rewards_norm = pd.Series([1.0] * len(rewards))

sample_weights = rewards_norm.clip(lower=0.1)

print(f"\nВаги прикладів: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}, mean={sample_weights.mean():.3f}")

features = [
    "core_temp", "temp_delta", "water_level", "steam_pressure",
    "throttle", "vent_open", "rods_pos", "current_speed", "distance"
]

X = df_balanced[features]
y = df_balanced["action_taken"]
w = sample_weights

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train, sample_weight=w_train)

print("\nТочність на тестових даних:")
print(classification_report(y_test, model.predict(X_test)))

cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Крос-валідація (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

print("\nВажливість ознак:")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:20s} {imp:.3f}")

with open("train_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nМодель збережено в train_model.pkl")