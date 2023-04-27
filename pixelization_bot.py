import logging
import os
import cv2
import numpy as np
import pixel_img
import pixel_exception as pe
from io import BytesIO
from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)
from telegram.ext import (
    filters,
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    CallbackQueryHandler
)
from warnings import filterwarnings
from telegram.warnings import PTBUserWarning

filterwarnings(action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

COLOR_LEVEL, PIXEL_SIZE, CONVERT_IMAGE = range(3)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Я могу пикселизировать изображения!\n"
        "Используйте команды и следуйте инструкциям "
        "для получения пикселизированного изображения\n\n"
        "Доступные команды:\n"
        "/image - пикселизация изображения"
    )
    
color_level_keyboard = [
    [
        InlineKeyboardButton("4", callback_data=4),
        InlineKeyboardButton("8", callback_data=8),
        InlineKeyboardButton("16", callback_data=16)
    ],
    [
        InlineKeyboardButton("32", callback_data=32),
        InlineKeyboardButton("64", callback_data=64),
        InlineKeyboardButton("128", callback_data=128),
    ]
]
color_level_kbd_markup = InlineKeyboardMarkup(color_level_keyboard)
    
async def image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Перед тем как пикселизировать изображения, я задам Вам несколько вопросов.\n"
        "Выберите количество уровней для каждого цветового канала RGB.",
        reply_markup=color_level_kbd_markup
    )
    
    return COLOR_LEVEL

async def color_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    
    await query.answer()
    
    # if query.data == 'cancel':
    #     await cancel_button(update, context)
    
    color_level = int(query.data)
    logger.info('Received color level, User %s', query.from_user.name)
    
    context.user_data['color_level'] = color_level
    
    await query.message.reply_text(
        "Вы выбрали {}.\nУкажите размер пикселей."
        .format(color_level)
    )
    
    return PIXEL_SIZE

async def pixel_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    
    if not update.message.text.isdigit():
        logger.warning('%s, User %s', pe.InvalidPixelSize().message, user.name)
        await update.message.reply_text(
            "Размер пикселей задан неверно!\n"
            "Введите корректное значение."
        )
        return PIXEL_SIZE
    
    pixel_size = int(update.message.text)
    logger.info('Received pizel size, User %s', user.name)
    
    if pixel_size < 2 or pixel_size > 1080:
        logger.warning('%s, User %s', pe.InvalidPixelSize().message, user.name)
        await update.message.reply_text(
            "Размер пикселей задан неверно!\n"
            "Введите корректное значение."
        )
        return PIXEL_SIZE
    
    context.user_data['pixel_size'] = pixel_size
    
    await update.message.reply_text(
        "Отправьте изображение для пикселизации."
    )
    
    return CONVERT_IMAGE
    
async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    
    image_file = await update.message.photo[-1].get_file()
    image = await image_file.download_as_bytearray()
    image = cv2.imdecode(np.asarray(image), cv2.IMREAD_COLOR)
    logger.info('Received image from User %s', user.name)
    
    color_level = context.user_data['color_level']
    pixel_size = context.user_data['pixel_size']
    
    try:
        palette = pixel_img.create_palette(color_level)
        converted_image = pixel_img.process_image(image, palette, pixel_size)
    
    except pe.InvalidPixelSize as e:
        logger.warning('%s, User %s', e.message, user.name)
        await update.message.reply_text(
            "Размер пикселей задан неверно!\n"
            "Введите корректное значение."
        )
        return PIXEL_SIZE
    
    ret, buffer = cv2.imencode('.jpg', converted_image)
    buf = BytesIO(buffer)
    logger.info('Converted image, User %s', user.name)
    
    await update.message.reply_photo(buf)
    
    return ConversationHandler.END

async def no_color_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Выберите количество уровней для каждого цветового канала RGB."
    )
    
    return COLOR_LEVEL

async def no_pixel_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Укажите размер пикселей."
    )
    
    return PIXEL_SIZE

async def no_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Отправьте изображение для пикселизации."
    )
    
    return CONVERT_IMAGE

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    logger.info('Canceled processing image, User %s', user.name)
    await update.message.reply_text(
        "Пикселизация изображения отменена."
    )

    return ConversationHandler.END
    
async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Извините, я не понял вашу команду."
    )

if __name__ == '__main__':
    load_dotenv()
    TOKEN = os.getenv('TOKEN')
    application = ApplicationBuilder().token(TOKEN).build()
    
    image_conversation_handler = ConversationHandler(
        entry_points=[CommandHandler("image", image)],
        states={
            COLOR_LEVEL: [
                CallbackQueryHandler(color_level),
                MessageHandler(filters.ALL & ~filters.COMMAND, no_color_level)
            ],
            PIXEL_SIZE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, pixel_size),
                MessageHandler(~filters.TEXT & ~filters.COMMAND, no_pixel_size)
            ],
            CONVERT_IMAGE: [
                MessageHandler(filters.PHOTO, process_image),
                MessageHandler(~filters.PHOTO & ~filters.COMMAND, no_image)
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(image_conversation_handler)
    application.add_handler(MessageHandler(filters.COMMAND, unknown))
    
    application.run_polling()