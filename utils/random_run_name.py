import random
import nltk
import string
nltk.download('words', quiet=True)

def generate_random_word():
    words = nltk.corpus.words.words()
    return random.choice(words).lower()

def generate_run_name():
    word1 = generate_random_word()
    word2 = generate_random_word()
    number = random.randint(1, 100)

    return f'{word1}-{word2}-{number}'

def generate_run_id(length=8):
    letters = string.ascii_lowercase + string.digits

    return ''.join(random.choice(letters) for _ in range(length))