import nltk
from nltk.book import *

# searching text
text1.concordance("monstrous")
text1.similar("monstrous")
text2.common_contexts(["monstrous","very"])
text4.dispersion_plot(["citizen","democracy", "freedom", "duties", "America", "liberty"])
text3.generate(length=50)

# counting vocabulary
def lexical_diverisity(text):
    return len(text)/len(set(text))
lexical_diversity(text3)

# simple statistics
fdist1 = FreqDist(text1)
fdist1
fdist1["whale"]
list(fdist1.iteritems())[:50]
fdist1.plot(50,cumulative=True)
fdist1.hapaxes()

V = set(text1)
long_words = [w for w in V if len(w)>15]
sorted(long_words)
sorted([w for w in set(text1) if len(w)>10 and fdist1[w]>20])

fdist2 = FreqDist(len(w) for w in text1)
list(fdist2.iteritems())
fdist2.tabulate()
fdist2.max()
fdist2[3]
fdist2.freq(3)

# collocations and bigrams
list(bigrams(["more", "is","said","than","done"]))
text4.collocations(num=25, window_size=10)

# corpus: Gutenberg Corpus, webtext, reuters, brown, inaugural, universal declaration of human rights
nltk.corpus.gutenberg.fileids()
nltk.corpus.webtext.fileids()
nltk.corpus.reuters.categories()

raw = nltk.corpus.gutenberg.raw("burgess-busterbrown.txt")
words = nltk.corpus.gutenberg.words("burgess-busterbrown.txt")
sents = nltk.corpus.gutenberg.sents("burgess-busterbrown.txt")

from nltk.corpus import brown
nltk.corpus.brown.categories()
cfd = nltk.ConditionalFreqDist( (genre,word) for genre in brown.categories() for word in brown.words(categories=genre))
genres = ["news", "religion", "humor"]
modals = ["can", "could", "will", "may"]
cfd.tabulate(conditions=genres, samples=modals)

from nltk.corpus import inaugural
cfd = nltk.ConditionalFreqDist((target, fileid[:4]) 
                                for fileid in inaugural.fileids() 
                                for w in inaugural.words(fileid) 
                                for target in ['america', 'citizen'] 
                                if w.lower().startswith(target))
cfd.plot()


from nltk.corpus import udhr
languages = ["German_Deutsch", "English", "Chickasaw","French_Francais"]
cfd = nltk.ConditionalFreqDist((lang, len(word))
                                for lang in languages
                                for word in udhr.words(lang+'-Latin1'))
cfd.plot(cumulative=True)

# loading your own corpus
from nltk.corpus import PlaintextCorpusReader
corpus_root = '/usr/share/dict'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
wordlists.fileids()


# filtering 
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)    
unusual_words(nltk.corpus.nps_chat.words())

def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content)/len(text)
content_fraction(nltk.corpus.reuters.words())

# wordnet
from nltk.corpus import wordnet as wn
wn.synsets('motor')
wn.synset('car.n.01').lemma_names
motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()

wn.synset('whale.n.02').min_depth()
wn.synset('entity.n.01').min_depth()

# semantic similarity
right = wn.synset('right_whale.n.01')
orca = wn.synset('orca.n.01')
tortoise = wn.synset('tortoise.n.01')
right.lowest_common_hypernyms(orca)
right.lowest_common_hypernyms(tortoise)

# accessing text
# can use beautifulsoup or feedparser to parse html, rss
import urllib2
url = "http://www.gutenberg.org/files/2554/2554.txt"
response = urllib2.urlopen(url)
raw = response.read().decode('utf8')
type(raw)

from nltk import word_tokenize
tokens = word_tokenize(raw)
text = nltk.Text(tokens)
text.collocations()
raw.find('PART I')
raw.rfind('End of Project')

# regular expression
import re
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
[w for w in wordlist if re.search('ed$',w) and len(w)==3]
[w for w in wordlist if re.search('ed$',w) and len(w)==4]
[w for w in wordlist if re.search('^..j..t..$',w)]
[w for w in wordlist if re.search('^[abdc][efgh][ijkl]$',w)]

regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[AEIOUaeiou]'
def compress(word):
    pieces = re.findall(regexp,word)
    return ''.join(pieces)
english_udhr = nltk.corpus.udhr.words('English-Latin1')
print (nltk.tokenwrap(compress(w) for w in english_udhr[:100]))

# suffix, stemming
re.findall(r'^.*(ing|ly|ed|ious|ive|es|s|ment)$','processing')
re.findall(r'^.*(?:ing|ly|ed|ious|ive|es|s|ment)$','processing')
re.findall(r'^(.*)(ing|ly|ed|ious|ive|es|s|ment)$','processes')
re.findall(r'^(.*?)(ing|ly|ed|ious|ive|es|s|ment)?$','processes')                                    
def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp,word)[0]
    return stem
stem('processing')

# searching tokenized text
moby = nltk.Text(nltk.corpus.gutenberg.words('melville-moby_dick.txt'))
moby.findall(r"<a>(<.*>)<man>")
chat = nltk.Text(nltk.corpus.nps_chat.words())
chat.findall(r"<.*><.*><bro>")
chat.findall(r"<l.*>{3,}")
hobbies_learned = chat = nltk.Text(nltk.corpus.brown.words(categories=['hobbies','learned']))
hobbies_learned.findall(r'<\w*><and><other><\w*s>')

# normalizing text
class IndexedText(object):
    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                                 for (i, word) in enumerate(text))
    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = int(width/4)                # words of context
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i-wc:i])
            rcontext = ' '.join(self._text[i:i+wc])
            ldisplay = '%*s'  % (width, lcontext[-width:])
            rdisplay = '%-*s' % (width, rcontext[:width])
            print(ldisplay, rdisplay)
    def _stem(self, word):
        return self._stemmer.stem(word).lower() 
                                    
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter,grail)
text.concordance('lie')

wnl = nltk.WordNetLemmatizer()
#[wnl.lemmatize(t) for t in tokens]

# regular expressions for tokenizing text
re.split(r' ', raw)
re.split(r'[ \t\n]+', raw)
re.split(r'\W+', raw)
re.findall(r'\w+|\S\w*')
re.findall(r"\w+(?:[-']\w+)*|'|[-.()]+|\S\w*", raw)
# NLTK's regexp tokenizer
text = 'That U.S.A. poster-print costs $12.4...'
pattern = r'''(?x)    # set flag to allow verbose regexps
      ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*        # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
    | \.\.\.            # ellipsis
    | [][.,;"'?():-_`]  # these are separate tokens
  '''
nltk.regexp_tokenize(text, pattern)

# segementation
len(nltk.corpus.brown.words())/len(nltk.corpus.brown.sents())
text = nltk

# higher-order functions
def is_content_word(word):
    return word.lower() not in ['a', 'of', 'the', 'and', 'will', ',', '.']
sent = ['Take', 'care', 'of', 'the', 'sense', ',', 'and', 'the',
'sounds', 'will', 'take', 'care', 'of', 'themselves', '.']
list(filter(is_content_word, sent)) # equivalent to a list comprehension
list(map(lambda w: len(filter(lambda c: c.lower() in 'aeiou', w)), sent))
        
#named arguments
def generic(*args, **kwargs):
    print (args)
    print (kwargs)
generic(1, 'African swallow', monty="python")

song = [['four', 'calling', 'birds'],
        ['three', 'French', 'hens'],
        ['two', 'turtle', 'doves']]
list(zip(*song))==list(zip(song[0], song[1], song[2]))

# use python debugger
import pdb