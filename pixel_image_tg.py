import logging
import cv2
import numpy as np
import pixel_exception as pe
import pixel_image
from io import BytesIO
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Document
)
from telegram.ext import (
    ContextTypes,
    ConversationHandler
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

COLOR_LEVEL, PIXEL_SIZE, PROCESS_IMAGE = range(3)

color_level_keyboard = [
    [
        InlineKeyboardButton("6-бит", callback_data=4),
        InlineKeyboardButton("9-бит", callback_data=8),
        InlineKeyboardButton("12-бит", callback_data=16)
    ],
    [
        InlineKeyboardButton("15-бит", callback_data=32),
        InlineKeyboardButton("18-бит", callback_data=64),
        InlineKeyboardButton("24-бит", callback_data=256),
    ]
]
color_level_kbd_markup = InlineKeyboardMarkup(color_level_keyboard)
    
async def image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Вы можете отменить процесс создания пиксель-арта с помощью команды /cancel.\n"
        "Прежде чем обработать ваше изображение, я задам Вам несколько вопросов.\n"
        "Выберите глубину цвета.",
        reply_markup=color_level_kbd_markup
    )
    
    return COLOR_LEVEL

async def color_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    
    await query.answer()
    
    context.user_data['color_level_image'] = int(query.data)
    logger.info('Received color level for image processing, User %s', query.from_user.name)
    
    await query.message.reply_text(
        "Отлично, теперь укажите, во сколько раз нужно увеличить пиксели?"
    )
    
    return PIXEL_SIZE

async def pixel_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    text = update.message.text
    
    if not text.isdigit() or int(text) < 2 or int(text) > 1080:
        logger.warning('In image processing: %s, User %s', pe.InvalidPixelSize().message, user.name)
        await update.message.reply_text(
            "Размер пикселей задан неверно!\n"
            "Введите корректное значение."
        )
        return PIXEL_SIZE
    
    context.user_data['pixel_size_image'] = int(text)
    logger.info('Received pizel size for image processing, User %s', user.name)
    
    await update.message.reply_text("Отправьте изображение для создания пиксель-арта.")
    
    return PROCESS_IMAGE
    
async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    
    if (type(update.message.effective_attachment) is Document):
        image_file = await update.message.effective_attachment.get_file()
    else:
        image_file = await update.message.photo[-1].get_file()
    
    image = await image_file.download_as_bytearray()
    image = cv2.imdecode(np.asarray(image), cv2.IMREAD_COLOR)
    logger.info('Received image for image processing, User %s', user.name)
    
    color_level = context.user_data['color_level_image']
    pixel_size = context.user_data['pixel_size_image']
    
    try:
        if (color_level != 256):
            palette = pixel_image.create_palette(color_level)
            converted_image = pixel_image.process_image(image, palette, pixel_size)
        else:
            converted_image = pixel_image.pixelize_image(image, pixel_size)
            
    except pe.InvalidPixelSize as e:
            logger.warning('In image processing: %s, User %s', e.message, user.name)
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

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    logger.info('Canceled image processing, User %s', user.name)
    await update.message.reply_text("Создание пиксель-арта отменено.")

    return ConversationHandler.END

async def cancel_required(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    logger.info('Cancellation of image processing required, User %s', user.name)
    await update.message.reply_text(
        "Прежде чем использовать другие функции бота,"
        "отмените создание пиксель-арта из изображения "
        "с помощью команды /cancel."
    )
