from telegram.error import TelegramError

class InvalidPixelSize(TelegramError):
    def __init__(self, message: str = None) -> None:
        super().__init__(message or "Invalid pixel size")
        
class InvalidColorLvl(TelegramError):
    def __init__(self, message: str = None) -> None:
        super().__init__(message or "Invalid color level")