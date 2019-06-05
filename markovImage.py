from PIL import Image
from markovchain import JsonStorage
from markovchain.image import MarkovImage

markov = MarkovImage()

markov.data(Image.open('twitterbot.png'))

width = 32
height = 16
img = markov(width, height)
with open('output.png', 'wb') as fp:
    img.save(fp)

markov.save('markov.json')

markov = MarkovImage.from_file('markov.json')