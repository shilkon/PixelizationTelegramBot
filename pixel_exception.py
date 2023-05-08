from telegram.error import TelegramError


class InvalidPixelSize(TelegramError):
    def __init__(self) -> None:
        super().__init__("Invalid pixel size")


class InvalidColorLvl(TelegramError):
    def __init__(self) -> None:
        super().__init__("Invalid color level")
