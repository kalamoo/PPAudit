from pathlib import Path
import random
random.seed(2023)

import dill as pickle
import networkx as nx
import yaml
import xml.etree.ElementTree as ET

other_data_dir = Path(r"D:\PPAudit\other_data")


def extract_cus_templates():
    # separate them into CUS / Non-CUS (ncus) sentences, and save the corresponding labels
    # output:
        # cus_words: len=X=1383 [(sentence 1 words), (sentence 2 words), ...]
        # cus_labels: len=X [(labels corresponding to sentence 1), (labels corresponding to sentence 2), ...]
        # ncus_words len=Y=14865 [(ncus sentence 1 words), (ncus sentence 2 words), ...]

    cus_words = []
    cus_labels = []
    ncus_words = []
    for cus_type in ['CollectUse', 'Share']:
        for tf in ['false', 'true']:
            for data_split in ['train', 'validation']:
                conll_file = other_data_dir/'{}_{}'.format(cus_type, tf)/'{}.conll03'.format(data_split)
                with open(conll_file, 'r', encoding='utf-8') as fb:
                    words = []
                    labels = []
                    for line in fb:
                        line = line.strip('\n')
                        if line == '-DOCSTART- -X- O O':
                            # remove the header
                            pass
                        elif line =='':
                            # end of one sentence, save results for last sentence
                            if len(words) != 0:
                                if set(labels) == {'O'}:  # -> Non-CUS
                                    ncus_words.append(words)
                                else:
                                    cus_words.append(words)
                                    cus_labels.append(labels)
                                # init for processing next sentence
                                words = []
                                labels = []
                        else:
                            # e.g., information _ _ B/I-(NOT_)?COLLECT/SHARE 
                            contents = line.split(' ')
                            words.append(contents[0])
                            labels.append(contents[-1][0])  # only leave 'B'/'I'/'O
    
    return cus_words, cus_labels, ncus_words


def vr_dataobj_phrases():
    ovr_phrases = []  # #=267
    
     # vr phrases from ovrseen
    dataobj_onto_poli = pickle.load(open(other_data_dir/'data_ontology_policheck.pickle', 'rb'))
    nodes_poli = dataobj_onto_poli._node.keys()
    dataobj_onto_ovr = nx.read_gml(other_data_dir/'data_ontology.gml')
    nodes_ovr = dataobj_onto_ovr._node.keys()
    added_ovr_nodes = [node for node in nodes_ovr if node not in nodes_poli]
    with open(other_data_dir/'data_synonyms.yml', 'r') as rf:
        dataobj_synon = yaml.safe_load(rf)
        for added_node in added_ovr_nodes:
            if added_node in dataobj_synon.keys():
                ovr_phrases.append(added_node)
                ovr_phrases.extend(dataobj_synon[added_node])

    return ovr_phrases


def insert_phrase_to_template(phrase, words_template, labels_template):    
    # print('before', phrase, words_template, labels_template)
    # a template might have multiple dataobj slot ('B')，randomly choose one
    data_pos = len(labels_template) % labels_template.count('B')
    # print(data_pos)
    b_cnt, b_start, i_end, word_cnt = 0, 0, 0, 0
    while word_cnt < len(words_template):
        if labels_template[word_cnt] == 'B': 
            if b_cnt == data_pos:  # find the postition to insert the phrase
                b_start = word_cnt  # record the beginning and ending of this dataobj slot
                i_end = b_start + 1
                while i_end < len(words_template) and labels_template[i_end] == 'I':
                    i_end += 1
                i_end -= 1
                break
            else:
                b_cnt += 1
                word_cnt += 1
        else:
            word_cnt += 1

    # insert the phrase into [b_start, i_end] slot
    phrase_words = phrase.split(' ')
    words = words_template[:b_start] + phrase_words + words_template[i_end+1:]
    labels = labels_template[:b_start] + ['B'] + ['I'] * (len(phrase_words)-1) + labels_template[i_end + 1:]
    # print('end', words, labels)
    return words, labels


def synthetic_cus():
    # ratio of synthetic dataset
    # CUS positive [1 vr phrases (267*3) + the rest of classic templates] (1383) ~ CUS negative [ncus (14865)]

    cus_words_template, cus_labels_template, ncus_words = extract_cus_templates()
    template_shuffle = list(range(len(cus_words_template)))
    random.shuffle(template_shuffle)

    vr_phrases = vr_dataobj_phrases()
    
    cus_words = []
    cus_labels = []
    
    # insert vr phrases (267 phrases into 1383 templates)
    dup = 1 + int(len(template_shuffle) / 2 / len(vr_phrases))
    for i in range(len(vr_phrases)):
        for j in range(dup):
            template_idx = template_shuffle[dup*i+j]
            words, labels = insert_phrase_to_template(vr_phrases[i], cus_words_template[template_idx], cus_labels_template[template_idx])
            cus_words.append(words)
            cus_labels.append(labels)
    for template_idx in template_shuffle[dup*len(vr_phrases): ]:
        cus_words.append(cus_words_template[template_idx])
        cus_labels.append(cus_labels_template[template_idx])
    
    # shuffle the outputs
    cus_words_r = []
    cus_labels_r = []
    random_idx = list(range(len(cus_words)))
    random.shuffle(random_idx)
    labels_map = {
        'O': 0,
        'B': 1,
        'I': 2
    }
    for i in random_idx:
        cus_words_r.append(cus_words[i])
        # labels transform from OBI t0 012
        new_labels = [labels_map[label] for label in cus_labels[i]]
        cus_labels_r.append(new_labels)

    random.shuffle(ncus_words)
    
    return cus_words_r, cus_labels_r, ncus_words
    

if __name__ == '__main__':
    synthetic_cus()
