# IMPORTS

import telebot
import time
from random import choice
from TeleBotClass import StyleTransferModel
from BotPhrases import get_phrases


# MAIN PART

# Model preparation
print('Model loading ...')
STM = StyleTransferModel()
print('Model loaded successfully!')

# Bot preparation
API_TOKEN = 'SomeToken'
bot = telebot.TeleBot(API_TOKEN)
print('Bot is ready!')


# Get phrases
phrase = get_phrases()


# Let's create style transfer bot just in 4 steps
# Step 1: send 'hi!'

@bot.message_handler(commands=['help', 'start'])
def send_start_message(message):
    global chat_id
    chat_id = message.chat.id
    bot.send_chat_action(chat_id, 'typing')
    print('New user detected')

    # Send start message
    with open('help_image.jpg', 'rb') as help_image:
        msg = bot.send_photo(chat_id, help_image, caption=phrase['start'][0].format(choice(phrase['start_r'])))
        # Go for the next step
        bot.register_next_step_handler(msg, get_content_step)


# Step 2: get content image

def get_content_step(message):
    global chat_id
    bot.send_chat_action(chat_id, 'typing')
    time.sleep(1)
    try:
        content_info = bot.get_file(message.photo[-1].file_id)
        content_image = bot.download_file(content_info.file_path)
        with open('content_image.jpg', 'wb') as file:
            file.write(content_image)
        msg = bot.send_message(chat_id, phrase['get_cont'][0])
        print('Get content')
        # Go for the next step
        bot.register_next_step_handler(msg, get_style_step)
    except Exception:
        msg = bot.send_message(chat_id, phrase['err_cont'][0])
        print('Error content')
        # Go for the previous step
        bot.register_next_step_handler(msg, get_content_step)


# Step 3-4: get style image and send result

def get_style_step(message):
    global chat_id

    # Step 3: get style image

    bot.send_chat_action(chat_id, 'typing')
    time.sleep(1)

    try:
        style_info = bot.get_file(message.photo[-1].file_id)
        style_image = bot.download_file(style_info.file_path)
        with open('style_image.jpg', 'wb') as file:
            file.write(style_image)
        bot.send_message(chat_id, phrase['get_style'][0])
        print('Get style')

        # Step 4: send result

        try:
            result_name = STM.forward('content_image.jpg', 'style_image.jpg')
            bot.send_chat_action(chat_id, 'typing')
            with open(result_name, 'rb') as result:
                bot.send_photo(chat_id, result)
            bot.send_message(chat_id, phrase['get_result'][0])
            print('Photo send')
        except Exception:
            print('Error result')
            bot.send_message(chat_id, phrase['err_result'][0])

    except Exception:
        msg = bot.send_message(chat_id, phrase['err_style'][0])
        print('Error style')
        # Go for the previous step
        bot.register_next_step_handler(msg, get_style_step)


# Cycle command
bot.polling()
