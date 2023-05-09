import logging
from io import BytesIO
from os import remove

import cv2
import numpy as np
from telegram import Document, Update
from telegram.ext import ContextTypes, ConversationHandler

from pixel_image import ImageHandler
from pixel_video import VideoHandler

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

ANONYMIZATION = 0


async def face(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Вы можете отменить процесс анонимизации с помощью команды /cancel.\n"
        "Отправьте изображение или видео для анонимизации лиц."
    )

    return ANONYMIZATION


async def anonymize_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user

    if isinstance(update.message.effective_attachment, Document):
        image_file = await update.message.effective_attachment.get_file()
    else:
        image_file = await update.message.photo[-1].get_file()

    image = await image_file.download_as_bytearray()
    image = cv2.imdecode(np.asarray(image), cv2.IMREAD_COLOR)
    logger.info("Received image for anonymization, User %s", user.name)

    if not ImageHandler(image).pixelize_faces():
        await update.message.reply_text("Лица не найдены.")
        logger.warning("Faces were not found in image, User %s", user.name)

        return ConversationHandler.END

    _, buffer = cv2.imencode(".jpg", image)
    buf = BytesIO(buffer)
    logger.info("Anonymized image, User %s", user.name)

    await update.message.reply_photo(buf)

    return ConversationHandler.END


async def anonymize_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user

    video_file = await update.message.effective_attachment.get_file()
    file_id = video_file.file_unique_id
    video = f"download/{file_id}.mp4"
    await video_file.download_to_drive(video)
    logger.info("Received video for anonymization, User %s", user.name)

    reply = await update.message.reply_text(
        "Видео обрабатывается, пожалуйста подождите..."
    )

    handler = VideoHandler(video)
    pixelized_video = handler.anonymize()
    if handler.faces_not_found():
        await context.bot.delete_message(reply.chat_id, reply.message_id)
        await update.message.reply_text("Лица не найдены.")
        logger.warning("Faces were not found in video, User %s", user.name)

        remove(video)
        remove(pixelized_video)

        return ConversationHandler.END

    with open(pixelized_video, "rb") as result:
        await context.bot.delete_message(reply.chat_id, reply.message_id)
        await update.message.reply_video(result)
        logger.info("Pixelized video sended, User %s", user.name)

    remove(video)
    remove(pixelized_video)

    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    logger.info("Canceled anonymization, User %s", user.name)
    await update.message.reply_text("Анонимизация лиц отменена.")

    return ConversationHandler.END


async def cancel_required(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    logger.info("Cancellation of anonymization required, User %s", user.name)
    await update.message.reply_text(
        "Прежде чем использовать другие функции бота,"
        "отмените анонимизацию лиц "
        "с помощью команды /cancel."
    )
