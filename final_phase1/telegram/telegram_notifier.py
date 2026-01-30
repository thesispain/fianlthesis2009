import os
import sys
import threading
import time
import json

# Try importing requests
try:
    import requests
except ImportError:
    requests = None

# --- CONFIGURATION ---
CONFIG_FILE = "telegram_config.json"
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
USER_IDS = [123456789]

def load_config():
    global BOT_TOKEN, USER_IDS
    # Look for config in the same directory as this script
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                BOT_TOKEN = cfg.get("bot_token", BOT_TOKEN)
                USER_IDS = cfg.get("user_ids", USER_IDS)
                print(f"[Telegram] Loaded config. Token ends with: ...{BOT_TOKEN[-5:] if len(BOT_TOKEN)>5 else '???'}")
        except Exception as e:
            print(f"[Telegram] Error loading config: {e}")

# Load Config on Import
load_config()

# --- BROADCAST FUNCTION ---
def telegram_broadcast(message_text):
    if not requests:
        return
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        return

    def task():
        for user_id in USER_IDS:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            try:
                # Telegram has a 4096 char limit
                text = message_text[:4000]
                requests.post(url, json={"chat_id": user_id, "text": text}, timeout=5, verify=False)
            except:
                pass

    threading.Thread(target=task).start()

# --- REDIRECTION ENGINE ---
def catch_all_outputs():
    load_config()
    
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("[Telegram] Token not configured. Use telegram_config.json to set it.")
        return

    # Flush buffers
    sys.stdout.flush()
    sys.stderr.flush()

    # Save original file descriptors so we can write to the *real* terminal
    # os.dup(1) creates a COPY of fd 1 (stdout) that points to the monitor
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)

    # Create pipes
    pipe_out_read, pipe_out_write = os.pipe()
    pipe_err_read, pipe_err_write = os.pipe()

    # Replace system stdout/stderr with our pipe inputs
    # Any subsequent print() or C-level printf will go into the pipe
    os.dup2(pipe_out_write, 1)
    os.dup2(pipe_err_write, 2)

    # Rate Limiter state
    last_broadcast_time = 0
    BROADCAST_INTERVAL = 45  # 45 Seconds (User Request)

    def reader_loop(pipe_rd, label, original_fd):
        nonlocal last_broadcast_time
        with os.fdopen(pipe_rd, 'r') as f:
            for line in iter(f.readline, ''):
                # 1. Write to the REAL terminal
                try:
                    os.write(original_fd, line.encode('utf-8', errors='replace'))
                except:
                    pass 

                # 2. Logic to filter what goes to Telegram
                clean_line = line.strip()
                if clean_line:
                    # Keywords
                    is_routine = "Loss" in clean_line or "it/s" in clean_line
                    is_critical = any(k in clean_line for k in ["Error", "Exception", "Finished", "Starting", "Optimization Complete"])
                    
                    if is_critical:
                        telegram_broadcast(f"[{label}] {clean_line}")
                    elif is_routine:
                        # Only send routine updates every 10 mins
                        if time.time() - last_broadcast_time > BROADCAST_INTERVAL:
                            telegram_broadcast(f"[{label}] ⏱️ Update: {clean_line}")
                            last_broadcast_time = time.time()

    # Start threads to read from the pipes
    t_out = threading.Thread(target=reader_loop, args=(pipe_out_read, "LOG", original_stdout_fd))
    t_out.daemon = True
    t_out.start()

    t_err = threading.Thread(target=reader_loop, args=(pipe_err_read, "ERR", original_stderr_fd))
    t_err.daemon = True
    t_err.start()

    print("[Telegram] Output Redirection Active. Monitoring for keywords: Epoch, Loss, AUC...")
    telegram_broadcast("[Telegram] 🚀 Training Script Started with Bot Monitoring!")

if __name__ == "__main__":
    # Test
    catch_all_outputs()
    print("This is a test log.")
    print("Epoch 1: Loss=0.5")
    time.sleep(1)
