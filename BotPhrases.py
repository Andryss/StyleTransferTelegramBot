# There are all bot's phrases

def get_phrases():
    return {
    'start': ['''Привет! Я StyleTransferBot. И я создан, чтобы {}! Шутка:)
    
На самом деле я бот для переноса стиля с одного изображения на другое (style transfer).
Чтобы все получилось, нужно следовать следующим шагам: 
- Вы даете мне две картинки (content и style, как в примере сверху);
- Я переношу стиль с картинки style на картинку content (ну или можно сказать рисую картинку content в стиле style);
- Показываю Вам результат (result) и Вы, надеюсь, остаетесь доволны.
  
Теперь, когда я все объяснил, жду от Вас картинку content.
    
(Если что-то осталось непонятным, то можно найти в интернете пару картинок по запросу Style Transfer. Хотя пример сверху должен все объяснить)
    '''],
    #
    'start_r': ['поработить человечество', 'манипулировать людьми', 'взорвать Землю', 'истребить всех живых существ',
                'устроить апокалипсис', 'узнать у вас номер кредитной карты', 'взломать ваш компьютер'],
    #
    'err_cont': ['''Так, так, так... Вы, кажется, меня не поняли. Вам нужно выбрать картинку и послать мне. Причем послать именно как картинку, а не как документ. 

(Просто пошлите мне картинку. Это не сложно)
    '''],
    #
    'get_cont': ['''Супер! Перейдем к следующему шагу. Теперь пошлите мне картинку style.'''],
    #
    'err_style': ['''Я конечно все понимаю. Но у меня возникает вопрос. Каким образом вы смогли отправить одну картинку, но не смогли отправить вторую?!
Просто сделайте все так же, как в прошлый раз.
    '''],
    #
    'get_style': ['''Замечательно! Теперь ждите результат. Он должен появится в течение 30-40 минут с момента отправки этого сообщения. 
Если результата нет, то, к сожалению, что-то пошло не так и разработчик вскоре постарается это исправить.


(ИНТЕРЕСНЫЙ ФАКТ. Если бы у разработчика было больше вычислительных ресурсов, то результат можно было бы увидеть через 1-2 минуты в хорошем качестве без всяких ошибок и вылетов.
Но поскольку платформа, на которой сейчас расположен бот, бесплатная, то ресурсов тут ооооочень мало, и к тому же есть ограничение по занимаемой памяти.
Поэтому придется немного подождать, а результат выйдет не в самом лучшем качестве) 
    '''],
    #
    'err_result': ['''Так. Спешу вам сообщить, что что-то пошло не так. Однако, тут уже ошибся разработчик. Скорее всего, эта неполадка будет скоро устранена.
    '''],
    #
    'get_result': ['''На этом моя работа закончена. Надеюсь, Вы удовлетворены результатом.    
Напишите /start или /help, чтобы начать заново.    
    ''']
}