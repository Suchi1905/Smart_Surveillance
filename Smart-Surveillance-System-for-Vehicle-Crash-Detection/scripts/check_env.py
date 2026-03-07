import os
from pathlib import Path

def check_env():
    env_path = Path(".env")
    print(f"Checking .env at {env_path.absolute()}")
    
    if not env_path.exists():
        print("❌ .env file NOT FOUND")
        return
    
    print("✅ .env file exists")
    
    with open(env_path) as f:
        content = f.read()
        
    bot_token = "MISSING"
    chat_id = "MISSING"
    
    for line in content.splitlines():
        if line.startswith("BOT_TOKEN="):
            bot_token = line.split("=", 1)[1].strip()
        if line.startswith("CHAT_ID="):
            chat_id = line.split("=", 1)[1].strip()
            
    print(f"BOT_TOKEN: {'*' * 5 + bot_token[-5:] if len(bot_token) > 5 else bot_token}")
    print(f"CHAT_ID: {chat_id}")
    
    if bot_token in ["YOUR_BOT_TOKEN", "MISSING"] or chat_id in ["YOUR_CHAT_ID", "MISSING"]:
        print("❌ Telegram credentials are not set correctly.")
    else:
        print("✅ Telegram credentials appear to be set.")

if __name__ == "__main__":
    check_env()
