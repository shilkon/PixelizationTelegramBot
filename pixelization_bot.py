import logging
import os
import cv2
import numpy as np
import pixel_img
from io import BytesIO
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    filters,
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

COLOR_DEPTH, PIXEL_SIZE, CONVERT_IMAGE = range(3)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "I can pixelize your image.\n"
        "Use commands and follow the instructions to get result.\n\n"
        "Available commands:\n"
        "/image - pixelization of an image"
    )
    
async def image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Before I pixelize your image, I will ask you some questions.\n"
        "What will be the color depth?"
    )
    
    return COLOR_DEPTH

async def color_depth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    logger.info("Received color depth from User %s", user.username)
    context.user_data['color_depth'] = int(update.message.text)
    await update.message.reply_text(
        "What will be the pixel size?"
    )
    
    return PIXEL_SIZE

async def pixel_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    logger.info("Received pizel size from User %s", user.username)
    context.user_data['pixel_size'] = int(update.message.text)
    await update.message.reply_text(
        "Send the image to pixelize it"
    )
    
    return CONVERT_IMAGE
    
async def convert_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    context.args
    image_file = await update.message.photo[-1].get_file()
    image = await image_file.download_as_bytearray()
    image = cv2.imdecode(np.asarray(image), cv2.IMREAD_COLOR)
    logger.info("Received image from User %s", user.username)
    
    color_depth = context.user_data['color_depth']
    palette = pixel_img.create_palette(color_depth)
    pixel_size = context.user_data['pixel_size']
    
    converted_image = pixel_img.convert_image(image, palette, pixel_size)
    ret, buffer = cv2.imencode('.jpg', converted_image)
    buf = BytesIO(buffer)
    logger.info("Converted image for User %s", user.username)
    
    await update.message.reply_photo(buf)
    
    return ConversationHandler.END
    
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    logger.info("Canceled converting image for User %s", user.username)
    await update.message.reply_text(
        "Conversation canceled"
    )

    return ConversationHandler.END

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Sorry, I didn't understand that command."
    )

if __name__ == '__main__':
    load_dotenv()
    TOKEN = os.getenv('TOKEN')
    application = ApplicationBuilder().token(TOKEN).build()
    
    image_conversation_handler = ConversationHandler(
        entry_points=[CommandHandler("image", image)],
        states={
            COLOR_DEPTH: [MessageHandler(filters.TEXT, color_depth)],
            PIXEL_SIZE: [MessageHandler(filters.TEXT, pixel_size)],
            CONVERT_IMAGE: [MessageHandler(filters.PHOTO, convert_image)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(image_conversation_handler)
    application.add_handler(MessageHandler(filters.COMMAND, unknown))
    
    application.run_polling(timeout=20)