import requests

# This is a common User-Agent for a web browser (like Chrome on a Mac)
browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'

# This is the custom User-Agent from the notebook, identifying a bot
bot_user_agent = 'Mozilla/5.0 (compatible; YourBot/0.1; +http://yourwebsite.com/bot.html)'

# --- Request 1: Pretending to be a regular browser ---
print("--- Making request with a BROWSER User-Agent ---")
headers_browser = {'User-Agent': browser_user_agent}
response_browser = requests.get('https://httpbin.org/headers', headers=headers_browser)

# httpbin.org/headers returns the headers it received in JSON format
# We are printing the 'User-Agent' that the server saw.
print(f"Server saw this User-Agent: {response_browser.json()['headers']['User-Agent']}\n")


# --- Request 2: Identifying as a bot ---
print("--- Making request with a BOT User-Agent ---")
headers_bot = {'User-Agent': bot_user_agent}
response_bot = requests.get('https://httpbin.org/headers', headers=headers_bot)

print(f"Server saw this User-Agent: {response_bot.json()['headers']['User-Agent']}") 