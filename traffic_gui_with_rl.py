import tkinter as tk
import numpy as np
import torch
from rl_agent.environment import FourWayIntersectionEnv
from rl_agent.adam_agent import AdamDQNAgent
from rl_agent.agent import DQNAgent as PSOAgent
from typing import Any

# --- Load trained agent if available ---
AGENT_MODEL_PATH = 'adam_traffic_model.pth'

# --- RL Environment and Agent Setup ---
env = FourWayIntersectionEnv(max_steps=9999)
adam_agent = AdamDQNAgent(state_size=4, action_size=2, learning_rate=0.0005)
pso_agent = PSOAgent(state_size=4, action_size=2)
try:
    adam_agent.load_model(AGENT_MODEL_PATH)
except Exception as e:
    print(f"⚠️ Could not load trained Adam model: {e}. Using untrained Adam agent.")

state = env.reset()
reward = 0
selected_agent: list[Any] = [adam_agent]  # Use a list for mutability in nested functions

# --- Agent selection logic ---
def set_agent(agent_name):
    if agent_name == "Adam":
        selected_agent[0] = adam_agent
    else:
        selected_agent[0] = pso_agent
    reset_env()

# --- GUI Setup ---
root = tk.Tk()
root.title("Traffic RL Simulation - Four Way Intersection")
root.geometry("900x700")

canvas = tk.Canvas(root, bg="white", width=700, height=700)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

stats_frame = tk.Frame(root)
stats_frame.pack(side=tk.RIGHT, fill=tk.Y)

font_large = ("Arial", 16, "bold")
font_med = ("Arial", 12)
font_small = ("Arial", 10)

step_count = 0

# --- Drawing Function ---
def draw_env():
    canvas.delete("all")
    center_x = 350
    center_y = 350
    road_len = 6
    segment_len = 40
    road_width = 40
    light_radius = 16
    # Draw horizontal road (EW)
    for i in range(road_len):
        x_e = center_x + (i + 1) * segment_len
        x_w = center_x - (i + 1) * segment_len
        canvas.create_rectangle(x_e, center_y - road_width // 2, x_e + segment_len, center_y + road_width // 2, fill="#e0e0e0")
        canvas.create_rectangle(x_w - segment_len, center_y - road_width // 2, x_w, center_y + road_width // 2, fill="#e0e0e0")
    # Draw vertical road (NS)
    for i in range(road_len):
        y_n = center_y - (i + 1) * segment_len
        y_s = center_y + (i + 1) * segment_len
        canvas.create_rectangle(center_x - road_width // 2, y_n, center_x + road_width // 2, y_n - segment_len, fill="#e0e0e0")
        canvas.create_rectangle(center_x - road_width // 2, y_s, center_x + road_width // 2, y_s + segment_len, fill="#e0e0e0")
    # Draw traffic lights
    ns_color, ew_color = ("green", "red") if env.phase == 0 else ("red", "green")
    # NS lights
    canvas.create_oval(center_x - light_radius, center_y - 3 * light_radius,
                       center_x + light_radius, center_y - light_radius, fill=ns_color)
    canvas.create_oval(center_x - light_radius, center_y + light_radius,
                       center_x + light_radius, center_y + 3 * light_radius, fill=ns_color)
    # EW lights
    canvas.create_oval(center_x - 3 * light_radius, center_y - light_radius,
                       center_x - light_radius, center_y + light_radius, fill=ew_color)
    canvas.create_oval(center_x + light_radius, center_y - light_radius,
                       center_x + 3 * light_radius, center_y + light_radius, fill=ew_color)
    # Draw cars as dots proportional to queue length
    q = env.queues.astype(int)
    for i in range(q[0]):  # North
        x = center_x - 10
        y = center_y - 60 - i * 8
        canvas.create_oval(x, y, x + 8, y + 8, fill="#2222ff")
    for i in range(q[1]):  # South
        x = center_x + 10
        y = center_y + 60 + i * 8
        canvas.create_oval(x, y, x + 8, y + 8, fill="#ff2222")
    for i in range(q[2]):  # East
        x = center_x + 60 + i * 8
        y = center_y - 10
        canvas.create_oval(x, y, x + 8, y + 8, fill="#22aa22")
    for i in range(q[3]):  # West
        x = center_x - 60 - i * 8
        y = center_y + 10
        canvas.create_oval(x, y, x + 8, y + 8, fill="#aaaa22")
    # Labels
    canvas.create_text(center_x, center_y - 180, text="NORTH", font=font_small)
    canvas.create_text(center_x, center_y + 180, text="SOUTH", font=font_small)
    canvas.create_text(center_x - 180, center_y, text="WEST", font=font_small)
    canvas.create_text(center_x + 180, center_y, text="EAST", font=font_small)
    # Phase label
    phase_label = "NS green" if env.phase == 0 else "EW green"
    canvas.create_text(center_x, center_y + 100, text=f"Phase: {phase_label}", font=font_large, fill="darkgreen")
    # Step label
    canvas.create_text(center_x, center_y - 100, text=f"Step: {step_count}", font=font_med, fill="black")
    # Reward label
    canvas.create_text(center_x, center_y + 130, text=f"Reward: {reward:.1f}", font=font_med, fill="blue")
    # Queue stats
    canvas.create_text(center_x + 200, center_y - 120, text=f"N queue: {q[0]}", font=font_med, fill="#2222ff")
    canvas.create_text(center_x + 200, center_y - 90, text=f"S queue: {q[1]}", font=font_med, fill="#ff2222")
    canvas.create_text(center_x + 200, center_y - 60, text=f"E queue: {q[2]}", font=font_med, fill="#22aa22")
    canvas.create_text(center_x + 200, center_y - 30, text=f"W queue: {q[3]}", font=font_med, fill="#aaaa22")

def step_env():
    global state, reward, step_count
    action = selected_agent[0].act(state)
    state, reward, done = env.step(action)
    step_count += 1
    draw_env()
    if not done:
        root.after(500, step_env)

def reset_env():
    global state, reward, step_count
    state = env.reset()
    reward = 0
    step_count = 0
    draw_env()

def start_simulation():
    step_env()

def stop_simulation():
    pass

# --- Control Buttons ---
btn_frame = tk.Frame(stats_frame)
btn_frame.pack(pady=20)

start_btn = tk.Button(btn_frame, text="Start", command=start_simulation, width=10, font=font_med, bg="#4caf50", fg="white")
start_btn.grid(row=0, column=0, padx=5)
reset_btn = tk.Button(btn_frame, text="Reset", command=reset_env, width=10, font=font_med, bg="#2196f3", fg="white")
reset_btn.grid(row=0, column=1, padx=5)
exit_btn = tk.Button(btn_frame, text="Exit", command=root.destroy, width=10, font=font_med, bg="#f44336", fg="white")
exit_btn.grid(row=0, column=2, padx=5)

# --- Agent Selection ---
agent_var = tk.StringVar(value="Adam")
agent_frame = tk.LabelFrame(stats_frame, text="Select Agent", font=font_med)
agent_frame.pack(pady=10)

adam_radio = tk.Radiobutton(agent_frame, text="Adam", variable=agent_var, value="Adam", font=font_med, command=lambda: set_agent("Adam"))
adam_radio.pack(anchor="w")
pso_radio = tk.Radiobutton(agent_frame, text="PSO", variable=agent_var, value="PSO", font=font_med, command=lambda: set_agent("PSO"))
pso_radio.pack(anchor="w")

draw_env()

root.mainloop()
