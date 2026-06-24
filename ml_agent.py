import json
import csv
import os
from datetime import datetime
import pickle
import pandas as pd

class RuleBasedAgent:    
    def __init__(self, log_file="train_telemetry_data.csv"):
        self._last_action = "none"
        self._action_cooldown = 0
        self.log_file = log_file
        self.prev_distance = None
        print(f"CSV логується в: {os.path.abspath(self.log_file)}") 
        self._init_csv()
    
    def _init_csv(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "core_temp", "temp_delta", "water_level",
                    "steam_pressure", "throttle", "vent_open", "rods_pos",
                    "current_speed", "distance", "reward", "action_taken"
                ])
    
    def get_action(self, state: dict) -> dict:
        temp     = state.get("core_temp", 20)
        temp_delta = state.get("temp_delta", 0)
        pressure = state.get("steam_pressure", 0)
        speed    = state.get("current_speed", 0)
        distance = state.get("distance", 0)
        water    = state.get("water_level", 100)

        if temp > 500 or temp_delta > 8:
            self._action_cooldown = 0
            self._log_state(state, "cool_reactor")
            return {"command": "cool_reactor"}

        if pressure > 150:
            self._action_cooldown = 0
            self._log_state(state, "vent_steam")
            return {"command": "vent_steam"}

        if distance < 40 and distance > 0 and speed > 8:
            self._action_cooldown = 0
            self._log_state(state, "full_stop")
            return {"command": "full_stop"}

        if self._action_cooldown > 0:
            self._action_cooldown -= 1
            self._log_state(state, "cooldown")
            return {}

        action = "none"

        if temp > 500 or temp_delta > 3:
            action = "cool_reactor"
        elif pressure > 150:
            action = "vent_steam"
        elif distance < 400 and distance > 0 and speed > 20:
            action = "reduce_speed"
        elif distance < 25 and distance > 0 and speed > 5:
            action = "full_stop"
        elif water < 20:
            action = "none"

        self._log_state(state, action)

        if action == "none":
            return {}

        self._action_cooldown = 10
        return {"command": action}
    
    def _log_state(self, state: dict, action: str):
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                state.get("core_temp", 0),
                state.get("temp_delta", 0),
                state.get("water_level", 0),
                state.get("steam_pressure", 0),
                state.get("throttle", 0),
                state.get("vent_open", 0),
                state.get("rods_pos", 0),
                state.get("current_speed", 0),
                state.get("distance", 0),
                state.get("reward", 0),
                action
            ])
            
class MLAgent:
    def __init__(self, model_path="train_model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self._action_cooldown = 0
        print("ML модель завантажена!")
    
    def get_action(self, state: dict) -> dict:
        temp     = state.get("core_temp", 20)
        pressure = state.get("steam_pressure", 0)
        speed    = state.get("current_speed", 0)
        distance = state.get("distance", 0)

        if distance < 15 and distance > 0 and speed > 3:
            self._action_cooldown = 0
            return {"command": "full_stop"}

        if self._action_cooldown > 0:
            self._action_cooldown -= 1
            return {}

        features = pd.DataFrame([{
            "core_temp":      state.get("core_temp", 0),
            "temp_delta":     state.get("temp_delta", 0),
            "water_level":    state.get("water_level", 0),
            "steam_pressure": state.get("steam_pressure", 0),
            "throttle":       state.get("throttle", 0),
            "vent_open":      state.get("vent_open", 0),
            "rods_pos":       state.get("rods_pos", 0),
            "current_speed":  state.get("current_speed", 0),
            "distance":       state.get("distance", 0),
        }])
        action = self.model.predict(features)[0]

        print(f"Модель вирішила: {action} | temp={state.get('core_temp',0):.0f} pressure={state.get('steam_pressure',0):.0f} speed={state.get('current_speed',0):.1f} distance={state.get('distance',0):.0f}")

        if action == "none":
            return {}

        self._action_cooldown = 10
        return {"command": action}