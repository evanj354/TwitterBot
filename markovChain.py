import markovify 

# markov = MarkovText()
index = []

def getChain():
	with open('quotes.txt', encoding='utf-8') as fp:
		for line in fp:
			if (line[0]).isalpha():
				index.append(line)
	# Build the model.
	text_model = markovify.Text(index)
	return text_model

# # # Print five randomly-generated sentences
def generate(text_model):
	return text_model.make_sentence()
		

