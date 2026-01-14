import requests
import time
import sys

def get_chat_id(token):
    """
    Fetch the latest chat ID from the bot's updates.
    """
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    
    print(f"Checking for messages on bot...")
    print(f"Please send a message 'Hello' to your bot on Telegram NOW.")
    
    # Poll for 60 seconds
    for i in range(12):
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get("ok"):
                results = data.get("result", [])
                if results:
                    # Get the most recent message
                    last_update = results[-1]
                    if "message" in last_update:
                        chat = last_update["message"]["chat"]
                        chat_id = chat["id"]
                        user = chat.get("username", "Unknown")
                        print(f"\n✅ SUCCESS! Found Chat ID for user @{user}")
                        print(f"Chat ID: {chat_id}")
                        return chat_id
            else:
                print(f"Error from Telegram: {data}")
                
        except Exception as e:
            print(f"Connection error: {e}")
            
        print(".", end="", flush=True)
        time.sleep(5)
        
    print("\n❌ No messages found. Did you send a message to the bot?")
    return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = input("Enter your Bot Token: ").strip()
        
    get_chat_id(token)
