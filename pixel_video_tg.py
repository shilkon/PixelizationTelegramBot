import logging
import os

from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler

import pixel_exception as pe
from pixel_video import VideoHandler

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

PIXEL_SIZE, PROCESS_VIDEO = range(2)


async def video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Вы можете отменить процесс пикселизации видео с помощью команды /cancel.\n"
        "Прежде чем обработать ваше видео, мне нужно узнать, "
        "во сколько раз нужно увеличить пиксели?"
    )

    return PIXEL_SIZE


async def pixel_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    text = update.message.text

    if not text.isdigit() or int(text) < 2 or int(text) > 1080:
        logger.warning(
            "In video pixelization: %s, User %s",
            pe.InvalidPixelSize().message,
            user.name,
        )
        await update.message.reply_text(
            "Размер пикселей задан неверно!\nВведите корректное значение."
        )
        return PIXEL_SIZE

    context.user_data["pixel_size_video"] = int(text)
    logger.info("Received pizel size for video pixelization, User %s", user.name)

    await update.message.reply_text("Отправьте видео для пикелизации.")

    return PROCESS_VIDEO


async def process_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user

    video_file = await update.message.effective_attachment.get_file()
    file_id = video_file.file_unique_id
    video = f"download/{file_id}.mp4"
    await video_file.download_to_drive(video)
    logger.info("Received video for video pixelization, User %s", user.name)

    reply = await update.message.reply_text(
        "Видео обрабатывается, пожалуйста подождите..."
    )

    pixel_size_video = context.user_data["pixel_size_video"]
    pixelized_video = VideoHandler(video).pixelize(pixel_size_video)

    with open(pixelized_video, "rb") as result:
        await context.bot.delete_message(reply.chat_id, reply.message_id)
        await update.message.reply_video(result)
        logger.info("Pixelized video sended, User %s", user.name)

    os.remove(video)
    os.remove(pixelized_video)

    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    logger.info("Canceled video pixelization, User %s", user.name)
    await update.message.reply_text("Пикселизация видео отменена.")

    return ConversationHandler.END


async def cancel_required(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    logger.info("Cancellation of video pixelization required, User %s", user.name)
    await update.message.reply_text(
        "Прежде чем использовать другие функции бота,"
        "отмените пикселизацию видео "
        "с помощью команды /cancel."
    )
