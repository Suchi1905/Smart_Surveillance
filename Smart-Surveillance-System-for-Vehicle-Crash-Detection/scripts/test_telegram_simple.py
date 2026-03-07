import requests
import os
from pathlib import Path

def test_telegram_api():
    # Load env manually
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("BOT_TOKEN="):
                    os.environ["BOT_TOKEN"] = line.split("=", 1)[1].strip()
                if line.startswith("CHAT_ID="):
                    os.environ["CHAT_ID"] = line.split("=", 1)[1].strip()

    bot_token = os.environ.get("BOT_TOKEN")
    chat_id = os.environ.get("CHAT_ID")
    
    print(f"Testing with Token: ...{bot_token[-5:] if bot_token else 'None'}")
    print(f"Testing with Chat ID: {chat_id}")
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": "🔍 Direct API Test from Script"
    }
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ SUCCESS")
        else:
            print("❌ FAILED")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")

if __name__ == "__main__":
    test_telegram_api()
