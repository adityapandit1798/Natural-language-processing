import nltk
nltk.download("all")
from nltk.tag import tnt
from nltk.corpus import indian

file_ids = indian.fileids()
print(file_ids)
'''
# Number of words for each language corpus
for f in file_ids:
	print(f)
	print(len(indian.raw(f))

# Number of sentences for each language corpus
for f in file_ids:
	print(f)
	print(len(indian.sents(f))
'''
text_hindi = "इराक के विदेश मंत्री ने अमरीका के उस प्रस्ताव का मजाक उड़ाया है , जिसमें अमरीका ने संयुक्त राष्ट्र के प्रतिबंधों को इराकी नागरिकों के लिए कम हानिकारक बनाने के लिए कहा है ।"
hindi_file = open(r'hindi.txt')
text_hindi = hindi_file.read()

train_data = indian.tagged_sents('hindi.pos')
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(train_data)

tagged_words_hindi = (tnt_pos_tagger.tag(nltk.word_tokenize(text_hindi)))
print(tagged_words_hindi)

with open('Tagged_data_hindi', 'w') as f:
	f.write('***************************')
	f.write('\nHINDI')
	f.write('\n***************************\n\n')
	for item in tagged_words_hindi:
		f.write("{0}\n".format(item))
	f.write('\n\n***************************')

text_marathi = "१७ नोकरीच्या संबंधाने असलेले अनेक प्रश्न, परिच्छेद १५ व १६ यांमध्ये चर्चा केलेल्या दोन प्रमुख प्रश्नांच्या उत्तरांचे काळजीपूर्वक परीक्षण केल्याने सोडवता येऊ शकतात."
marathi_file = open(r'marathi.txt')
text_marathi = marathi_file.read()

train_data  = indian.tagged_sents('marathi.pos')
tnt_pos_tagger.train(train_data)

tagged_words_marathi = (tnt_pos_tagger.tag(nltk.word_tokenize(text_marathi)))
print(tagged_words_marathi)

with open('Tagged_data_marathi', 'w') as f:
	f.write('***************************')
	f.write('\nMARATHI')
	f.write('\n***************************\n\n')
	for item in tagged_words_marathi:
		f.write("{0}\n".format(item))
	f.write('\n\n***************************')

