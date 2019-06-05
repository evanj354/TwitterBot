import tweepy
import markovChain

print("This is my twitter bot") 

CONSUMER_KEY = 'xBrF3oxTPbHZ3MBNUTb44pxfr'
CONSUMER_SECRET = 'j3aSNqKo6vEEAbKIgyEUKieTEiCQ7aQr4Hf5gpPsYzqnAgWmzE'
ACCESS_KEY = '930213357008822272-atSpInyEz33mRFBdNSS9gHkxqRNpwt7'
ACCESS_SECRET = 'yt6j7L9iDtXJwfyC8Rwx5JwqSFdBoxwrlzlTrJqnZ0MLM'

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