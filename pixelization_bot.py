import logging
from os import getenv
from warnings import filterwarnings

from dotenv import load_dotenv
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
from telegram.warnings import PTBUserWarning

import pixel_face_tg as pixel_face
import pixel_image_tg as pixel_image
import pixel_video_tg as pixel_video

filterwarnings(
    action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Я могу создавать пиксель-арт из изображения и пикселизировать видео!\n"
        "Используйте команды и следуйте инструкциям.\n\n"
        "Доступные команды:\n"
        "/image - стилизация изображения под пиксель-арт\n"
        "/video - пикселизация видео"
    )


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Извините, я не понял вашу команду.")


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(context.error)
    if isinstance(context.error, TelegramError):
        match context.error.message:
            case "File is too big":
                await update.message.reply_text(
                    "Файл слишком большой!\n" "Максимальный размер файла равен 20 МБ"
                )


if __name__ == "__main__":
    load_dotenv()
    TOKEN = getenv("TOKEN")
    application = ApplicationBuilder().token(TOKEN).build()

    pixel_image_conversation_handler = ConversationHandler(
        entry_points=[CommandHandler("image", pixel_image.frame)],
        states={
            pixel_image.COLOR_LEVEL: [CallbackQueryHandler(pixel_image.color_level)],
            pixel_image.PIXEL_SIZE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, pixel_image.pixel_size)
            ],
            pixel_image.PROCESS_IMAGE: [
                MessageHandler(
                    filters.PHOTO | filters.Document.IMAGE, pixel_image.process
                )
            ],
        },
        fallbacks=[
            CommandHandler("cancel", pixel_image.cancel),
            MessageHandler(filters.COMMAND, pixel_image.cancel_required),
        ],
    )

    pixel_video_conversation_handler = ConversationHandler(
        entry_points=[CommandHandler("video", pixel_video.video)],
        states={
            pixel_video.PIXEL_SIZE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, pixel_video.pixel_size)
            ],
            pixel_video.PROCESS_VIDEO: [
                MessageHandler(filters.VIDEO, pixel_video.process_video)
            ],
        },
        fallbacks=[
            CommandHandler("cancel", pixel_video.cancel),
            MessageHandler(filters.COMMAND, pixel_video.cancel_required),
        ],
    )

    anonymization_conversation_handler = ConversationHandler(
        entry_points=[CommandHandler("face", pixel_face.face)],
        states={
            pixel_face.ANONYMIZATION: [
                MessageHandler(
                    filters.PHOTO | filters.Document.IMAGE, pixel_face.anonymize_image
                ),
                MessageHandler(filters.VIDEO, pixel_face.anonymize_video),
            ]
        },
        fallbacks=[
            CommandHandler("cancel", pixel_face.cancel),
            MessageHandler(filters.COMMAND, pixel_face.cancel_required),
        ],
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(pixel_image_conversation_handler)
    application.add_handler(pixel_video_conversation_handler)
    application.add_handler(anonymization_conversation_handler)
    application.add_handler(MessageHandler(filters.COMMAND, unknown))

    application.add_error_handler(error)

    application.run_polling()
