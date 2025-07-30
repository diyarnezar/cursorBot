import requests
import logging
from typing import Optional

class TelegramNotifier:
    """
    Handles sending notifications to a Telegram chat.
    """
    def __init__(self, token: str, chat_id: str):
        """
        Initializes the notifier with the bot token and chat ID.
        :param token: Your Telegram Bot's API token.
        :param chat_id: The chat ID to send messages to.
        """
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        if not token or not chat_id or token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
            logging.warning("Telegram token or chat_id not configured. Notifications will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logging.info("Telegram Notifier initialized.")

    def send_message(self, message: str) -> bool:
        """
        Sends a message to the configured Telegram chat.
        Includes robust error handling.
        :param message: The text message to send.
        :return: True if the message was sent successfully, False otherwise.
        """
        if not self.enabled:
            return False

        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        try:
            response = requests.post(self.base_url, data=payload, timeout=10)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            logging.info(f"Sent Telegram message: {message}")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send Telegram message due to a network error: {e}")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred while sending Telegram message: {e}")
            return False
