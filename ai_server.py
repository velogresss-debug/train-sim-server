import asyncio
import json
import websockets
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import os
import numpy as np
from rapidfuzz import process, fuzz
import edge_tts
import pygame
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model")
speech_queue = queue.Queue()
audio_queue = queue.Queue()

if not os.path.exists(model_path):
    print(f"ПОМИЛКА: Папку моделі не знайдено в {model_path}")
    exit(1)

model = Model(model_path)
rec = KaldiRecognizer(model, 16000)

from ml_agent import MLAgent
ml_agent = MLAgent()

global_state = {
    "last_temp_for_delta": 20.0
}

commands_map = {
    "full_stop": ["зупини", "стоп", "гальмуй"],
    "cool_reactor": ["охолоди", "стержні", "охолодження"],
    "vent_steam": ["скинь", "пара", "тиск"],
    "report": ["статус", "звіт", "доповідь"],
    "distance": ["відстань", "дистанція", "дальність"] 
}

roots = {
    "зупин": "full_stop", "гальм": "full_stop", 
    "стерж": "cool_reactor", "охолод": "cool_reactor",
    "пар": "vent_steam", "тиск": "vent_steam",
    "відстан": "distance", "дистанц": "distance"
}

def get_refined_command(text):
    text = text.lower().strip()
    if not text: return "none"
    for root, cmd in roots.items():
        if root in text: return cmd
    
    all_synonyms = [word for synonyms in commands_map.values() for word in synonyms]
    best_match = process.extractOne(text, all_synonyms, scorer=fuzz.WRatio)
    
    if best_match and best_match[1] > 75:
        matched_word = best_match[0]
        for cmd, synonyms in commands_map.items():
            if matched_word in synonyms: return cmd
    return "none"

def get_ai_advice(state):
    water = state.get("water_level", 100)
    if water < 30: return "ВОДА ЗАКІНЧУЄТЬСЯ!"
    return ""

def speech_worker():
    pygame.mixer.init()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        text = speech_queue.get()
        if text is None: break
        
        voice = "uk-UA-PolinaNeural"
        temp_file = os.path.join(BASE_DIR, "temp_voice.mp3")
        
        for attempt in range(3):
            try:
                communicate = edge_tts.Communicate(text, voice)
                loop.run_until_complete(communicate.save(temp_file))
                
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.music.unload()
                break
            except Exception as e:
                print(f"Спроба {attempt + 1}/3 невдала: {e}")
                if attempt < 2:
                    import time
                    time.sleep(1)
                else:
                    print(f"Озвучка пропущена: {text}")
        
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    if text: speech_queue.put(text)

def audio_callback(indata, frames, time, status):
    audio_queue.put(bytes(indata))

def process_audio_sync():
    if not audio_queue.empty():
        data = audio_queue.get()
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            return res.get('text', '')
    return None

async def handle_train_logic(websocket):
    print("локомотив підключено")
    train_state = {}
    last_advice = ""

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        while True:
            try:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.005)
                    data = json.loads(message)
                    
                    if data.get("type") == "speak_request":
                        speak(data.get("text"))
                    else:
                        train_state = data
                        
                        current_core_temp = train_state.get("core_temp", 20.0)
                        train_state["temp_delta"] = current_core_temp - global_state["last_temp_for_delta"]
                        global_state["last_temp_for_delta"] = current_core_temp
                        
                        ml_action = ml_agent.get_action(train_state)
                        if ml_action: 
                            await websocket.send(json.dumps(ml_action))
                        
                        advice = get_ai_advice(train_state)
                        if advice and advice != last_advice:
                            await websocket.send(json.dumps({"command": "ai_warning", "text": advice}))
                            speak(advice)
                            last_advice = advice
                except asyncio.TimeoutError:
                    pass

                raw_text = await asyncio.to_thread(process_audio_sync)
                
                if raw_text:
                    print(f"Почуто: {raw_text}")
                    command_id = get_refined_command(raw_text)
                    
                    if command_id != "none":
                        reply = ""
                        if command_id == "report":
                            reply = f"Температура: {int(train_state.get('core_temp',0))} градусів. Тиск: {int(train_state.get('steam_pressure',0))}"
                        elif command_id == "distance":
                            dist = int(train_state.get('distance', 0))
                            reply = f"Дистанція до цілі: {dist} метрів."
                        
                        await websocket.send(json.dumps({"command": command_id, "reply": reply}))
                        speak(reply)
                        print(f"Виконано: {command_id}")

            except Exception as e:
                print(f"Зв'язок перервано або помилка: {e}")
                break

            await asyncio.sleep(0.005)

async def main():
    async with websockets.serve(handle_train_logic, "127.0.0.1", 5001):
        print("Сервер працює на порту 5001...")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nСервер зупинено.")