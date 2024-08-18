from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import RegexpTokenizer
from gensim import corpora
from gensim.similarities import MatrixSimilarity
# pip install gensim==4.0.1 smart-open==1.9.0
import json

def get_stopwords():
    stop_file = open(r"D:\PPAudit\other_data\stopwords.txt", "r", encoding="utf8")
    stop_words = []
    for line in stop_file.readlines():
        stop_words.append(line.strip('\n'))
    stop_file.close()
    return stop_words

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def get_stems(tokens):
    tagged_sent = pos_tag(tokens)
    wnl = WordNetLemmatizer()
    lemmas_noun = []
    lemmas_verb = []
    lemmas_adj = []
    for tag in tagged_sent:
        tmp = get_wordnet_pos(tag[1])
        # print(tmp)
        wordnet_pos = tmp or wordnet.NOUN
        if tmp == 'n':
            lemmas_noun.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        if tmp == 'v':
            lemmas_verb.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        if tmp == 'a':
            lemmas_adj.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    return lemmas_noun, lemmas_verb, lemmas_adj


def get_stemmer_freq(tokens):
    tagged_sent = pos_tag(tokens) 
    # print(tagged_sent)
    wnl = WordNetLemmatizer()
    lemmas_noun = []
    lemmas_verb = []
    # print(tokens)
    for tag in tagged_sent:
        tmp = get_wordnet_pos(tag[1])
        wordnet_pos = tmp or wordnet.NOUN
        if tmp == 'n':
            lemmas_noun.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        if tmp == 'v':
            lemmas_verb.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        # lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    # freq_noun = nltk.FreqDist(lemmas_noun)
    # freq_verb = nltk.FreqDist(lemmas_verb)
    return lemmas_noun, lemmas_verb


def get_title_freq(tokens):
    tagged_sent = pos_tag(tokens)
    wnl = WordNetLemmatizer()
    lemmas_noun = []
    lemmas_adj = []
    for tag in tagged_sent:
        tmp = get_wordnet_pos(tag[1])
        # print(tmp)
        wordnet_pos = tmp or wordnet.NOUN
        if tmp == 'n':
            lemmas_noun.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        if tmp == 'a':
            lemmas_adj.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    return lemmas_noun, lemmas_adj


class DesSimSolver:

    def __init__(self):
        self.tokenizer = RegexpTokenizer('\\w+') 
        self.stop_words = get_stopwords()
        self.MODES = ['DesN', 'DesN+Title', 'All']

    def getWordList_single(self, content):
        tokens = self.tokenizer.tokenize(content)
        tokens = [word for word in tokens if word not in self.stop_words] 
        tokens = [word for word in tokens if not word.isnumeric()] 
        tokens = [word for word in tokens if len(word) > 1] 

        nouns, verbs, adjs = get_stems(tokens)
        return nouns, verbs, adjs

    def getWordCount_multiple(self, appNames, descriptions):
        des_nouns_list = []
        des_verbs_list = []
        title_nouns_list = []
        title_adjs_list = []
        for i in range(len(appNames)):
            des_nouns, des_verbs, des_adjs = self.getWordList_single(descriptions[i])
            title_nouns, title_verbs, title_adjs = self.getWordList_single(appNames[i])
            des_nouns_list.append(des_nouns)
            des_verbs_list.append(des_verbs)
            title_nouns_list.append(title_nouns)
            title_adjs_list.append(title_adjs)
        return des_nouns_list, des_verbs_list, title_nouns_list, title_adjs_list

    def getDesSim(self, app_link_hashes, appNames, descriptions, mode, threshold=0):
        # app_links_hashes / appNames / descriptions 
        # first element: appA; other elements: A's counterparts
        # return:
        #       ordered counterparts based on description similarity

        TITLE_RATIO_NOUN = 0.02
        TITLE_RATIO_ADJ = 0.01

        target_corpus = []
        for i in range(len(appNames)):
            des_nouns, des_verbs, des_adjs = self.getWordList_single(descriptions[i])
            title_nouns, title_verbs, title_adjs = self.getWordList_single(appNames[i])

            des_total_nouns_verbs = len(des_nouns) + len(des_verbs)
            title_nouns_factor = max(1, round(des_total_nouns_verbs * TITLE_RATIO_NOUN))
            title_adjs_factor = max(1, round(des_total_nouns_verbs * TITLE_RATIO_ADJ))
            title_nouns = title_nouns * title_nouns_factor
            title_adjs = title_adjs * title_adjs_factor

            if mode == 'DesN':
                target_corpus.append(des_nouns)
            elif mode == 'DesN+Title':
                target_corpus.append(des_nouns + title_nouns + title_adjs)
            elif mode == 'All':
                target_corpus.append(des_nouns + des_verbs + title_nouns + title_adjs)
            else:
                pass

        dictionary = corpora.Dictionary(target_corpus)  # word of bag
        vecs = [dictionary.doc2bow(doc) for doc in target_corpus] 
        vec_appA = vecs[0]
        sims_index = MatrixSimilarity(vecs, num_features=len(dictionary))
        sims = sims_index[vec_appA]

        sims_sorted = []
        sort_idx = sims.argsort()[::-1]
        for rank, idx in enumerate(sort_idx):
            if sims[idx] > threshold:
                sims_sorted.append([rank, app_link_hashes[idx], appNames[idx], float(sims[idx])])

        # print(sims_sorted)
        return sims_sorted

