import tweepy
import markovChain

print("This is my twitter bot") 

CONSUMER_KEY = ''
CONSUMER_SECRET = ''
ACCESS_KEY = ''
ACCESS_SECRET = ''

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
api = tweepy.API(auth) #api object used to read and write data into Twitter

mentions = api.mentions_timeline()

# for mention in mentions:
# 	print(str(mention.id) + " - " + mention.text)
# 	if('#hi' in mention.text):
# 		print('Found hi')
# 		print('responding back')

text_model = markovChain.getChain()

new_status = markovChain.generate(text_model)

api.update_status(new_status)