# Edited based on OVRSeen and PoliCheck / PolicyLint

import os
import re
import yaml
import spacy
from fuzzywuzzy import fuzz
import OntologyOps as ontutils
import json
from pathlib import Path

#####################################################################################################
# Pre-process

def fixWhitespace(text):
    text = re.sub(r'^\s+', '', text)
    text = re.sub(r'\s+$', '', text)
    return re.sub(r'\s+', ' ', text)


def stripEtc(term):
    term = re.sub(r'\b(etc)(\.)?$', '', term)
    return fixWhitespace(term)


def commonTermSubstitutions(term):
    # third-party --> third party
    term = re.sub(r'\b(third\-party)\b', 'third party', term)
    term = re.sub(r'\b(app(s)?|applications)\b', 'application', term)
    term = re.sub(r'\b(wi\-fi)\b', 'wifi', term)
    term = re.sub(r'\b(e\-\s*mail)\b', 'email', term)
    return fixWhitespace(term)


def stripIrrelevantTerms(term):
    pronRegex = re.compile(r'^(your|our|their|its|his|her|his(/|\s(or|and)\s)her)\b')
    irrevRegex = re.compile(r'^(additional|also|available|when\snecessary|obviously|technically|typically|basic|especially|collectively|certain|general(ly)?|follow(ing)?|important|limit(ed)?(\s(set|amount)\sof)?|more|most|necessary|only|optional|other|particular(ly)?|perhaps|possibl(e|y)|potential(ly)?|relate(d)?|relevant|require(d)?|select|similar|some(times)?|specific|variety\sof|various(\s(type|kind)(s)\sof)?)\b(\s*,\s*)?')
    while pronRegex.search(term) or irrevRegex.search(term):
        term = fixWhitespace(pronRegex.sub('', term))
        term = fixWhitespace(irrevRegex.sub('', term))
    return fixWhitespace(term)


def simpleSynonymSub(term):
    def isSafeSubstitution(term):  # Don't sub if there's a chance there are multiple terms in a noun phrase
        return False if re.search(r'\b(and|or)\b', term) or re.search(r'(,|;)', term) else True
    
    def isSimpleNonPersonalInfoTerm(term):
        if not isSafeSubstitution(term):
            return False
        if re.search(r'^(non-(pii|personally(\-|\s)identif(y|iable)\s(information|data|datum|detail)))$', term):
            return True
        return True if re.search(
            r'\b((information|data|datum|detail)\s.*\snot\sidentify\s(you|user|person|individual))\b', term) else False

    def isSimplePersonallyIdentifiableInfoTerm(term):
        if not isSafeSubstitution(term):
            return False
        #   if re.search(r'^((information|data|datum|detail)\sabout\syou)$', term):
        #       return True
        if re.search(r'^(pii|personally(\-|\s)identif(y|iable)\s(information|data|datum|detail))$', term):
            return True
        return True if re.search(
            r'\b((information|data|datum|detail)\s.*\sidentify\s(you(rself)?|user|person|individual))\b',
            term) else False

    def isSimpleIpAddr(term):
        if not isSafeSubstitution(term):
            return False
        return True if re.search(r'\b((ip|internet(\sprotocol)?)\saddress(es)?)\b', term) else False

    def isSimpleUsageInfoTerm(term):
        if not isSafeSubstitution(term):
            return False
        return True if re.search(
            r'^(information|data|datum|record|detail)\s+(about|regard|of|relate\sto)(\s+how)?\s+(you(r)?\s+)?(usage|use|uses|utilzation|activity)\s+(of|on|our)\s+.*$',
            term) else False

    if not isSafeSubstitution(term):
        return term
    if isSimpleNonPersonalInfoTerm(term):
        term = 'non-personally identifiable information'
    elif isSimplePersonallyIdentifiableInfoTerm(term):
        term = 'personally identifiable information'
    elif isSimpleIpAddr(term):
        term = 'ip address'
    elif isSimpleUsageInfoTerm(term):
        term = 'usage information'
    return term


def subInformation(text):
    text = re.sub(r'\b(info|datum|data)\b', 'information', text)
    #this can happen when subbing data for information
    return fixWhitespace(re.sub(r'\b(information(\s+information)+)\b', 'information', text))


def subOrdinals(term):
    term = re.sub(r'\b(1st)\b', 'first', term)
    term = re.sub(r'\b(3rd)\b', 'third', term)
    return fixWhitespace(term)


def stripQuotes(term):
    return fixWhitespace(re.sub(r'"', '', term))


def stripBeginOrEndPunct(term):
    punctRegex = re.compile(r'((^\s*(;|,|_|\'|\.|:|\-|\[|/)\s*)|((;|,|_|\.|:|\-|\[|/)\s*$))')
    andOrRegex = re.compile(r'^(and|or)\b')
    while punctRegex.search(term) or andOrRegex.search(term):
        term = fixWhitespace(punctRegex.sub('', term))
        term = fixWhitespace(andOrRegex.sub('', term))
    return term


def preprocess_term(term):

    # term = cleanupUnicodeErrors(term)  NOTE: Already fixed in the preprocessor (Hao Cui)

    # Strip unbalanced parentheses
    if not re.search(r'\)', term):
        term = re.sub(r'\(', '', term)
    if not re.search(r'\(', term):
        term = re.sub(r'\)', '', term)

    term = stripBeginOrEndPunct(term)
    term = stripEtc(term)
    term = stripBeginOrEndPunct(term)  #Do this twice since stripping etc may result in ending with punctuation...
    term = subOrdinals(term)
    term = stripQuotes(term)
    term = commonTermSubstitutions(term)
    term = stripIrrelevantTerms(term)

    term = fixWhitespace(term)
    term = simpleSynonymSub(term)
    term = subInformation(term)

    return term


####################################################################################################

class EntityHandler:
    def __init__(self, entity_synonyms_path, entity_ontology_path):
        
        # load synonyms file
        with open(entity_synonyms_path) as fin:
            d = yaml.safe_load(fin)
        ret = dict()
        for name, li in d.items():
            for synonym in (li or []):
                ret[synonym] = name
        self.entity_map = ret
        
        # init Entity ontology
        self.entity_onto = ontutils.loadEntityOntology(entity_ontology_path)
        
        self.entity_anyone = ['anyone']
        self.entity_we = ['first party', 'we']
        self.entity_cls = ['third party', 'law enforcement', 'payment processor', 'api', 'analytic provider',
                      'platform provider', 'social network', 'search engine', 'ad network', 'tech support', 'crm']
        self.entity_big_cpy = ['meta', 'google', 'microsoft', 'oracle', 'verizon', 'sony', 'adobe', 'amazon']
        self.entity_small_cpy = list(set(self.entity_onto.nodes).difference(
            set(self.entity_anyone + self.entity_we + self.entity_cls + self.entity_big_cpy)
        ))
        
        # mainly used for func fixLemma (i.e., tokenize and lemma form)
        self.nlp = spacy.load(r"D:\PPAudit\other_data\NlpFinalModel_Policheck")
        
    def fixLemma(self, txt):  # remove spacy.symbols.DET, and return lemma
        def getLemma(tok):
            return tok.lemma_ if tok.lemma_ != '-PRON-' else tok.text
    
        doc = self.nlp(txt)
        return ' '.join([getLemma(t) for t in doc if t.pos != spacy.symbols.DET])

    def get_synonym(self, text):
        # entity could be some entity class (entity_cls) or certain company name (entity_cpy)
        
        term = preprocess_term(self.fixLemma(text))

        # give preference to first attempt to match company's name
        for small_cpy in self.entity_small_cpy:
            if small_cpy in term:  # give preference to small company's name
                return small_cpy
        for big_cpy in self.entity_big_cpy:
            if big_cpy in term:
                return big_cpy
        
        # if not a certain company, then match entity class
        if term in self.entity_map:
            return self.entity_map[term]
        else:
            # Strip apostrophe and quotes
            term = re.sub(r'("|\'(\s*s)?)', '', term)
            # if still not found, return the result after strip as the synonym
            return self.entity_map.get(term, term)
    
    def process_e(self, e, firstPartyNames):
        eproc = self.get_synonym(e)
        
        if eproc.strip() == '' or eproc == 'IGNORE':
            return [], []
        if eproc in ['user', 'you', 'person', 'consumer', 'participant']:
            return [], []
        
        if eproc == 'third_party_implicit':
            eproc = 'third party'
            
        # 1st party
        if eproc in ['we', 'i', 'us', 'me', 'we_implicit'] \
                or eproc in ['app', 'mobile application', 'mobile app', 'application', 'service', 'website', 'web site', 'site'] \
                or (e.startswith('our') and eproc in ['app', 'mobile application', 'mobile app', 'application', 'service', 'company', 'business', 'web site', 'website', 'site']):
            eproc = 'we'
    
        # fix non-pronoun first party names
        for name in e, eproc:  # try `e` as well in case words in `eproc` are lemmatized
            tokens = re.split(r'\W+', name.strip().lower())
            if tokens and tokens[-1].strip('.') in ['inc', 'llc', 'ltd']:
                tokens.pop()
            test_name = " ".join(tokens).strip()
    
            for fp_name in firstPartyNames:  # compare with the first party name that we collected before
                test_fp_name = " ".join(re.split(r'\W+', fp_name.lower())).strip()
                if fuzz.partial_ratio(test_fp_name, test_name) > 90:
                    eproc = 'we'
                    break
        
        if eproc in self.entity_onto.nodes:
            return [eproc], []
            
        else:  # if still not found
            ents = []  # then split (if possible), and attemp to find the onto node again
            error_logs = []
            res = re.sub(r'\b(and|or|and/or|\/|&)\b', '\n', eproc)
            for e in res.split('\n'):
                e = e.strip()
                e = self.get_synonym(e)
                if e not in self.entity_onto.nodes:  # if still not found, log to error and ignore it
                    error_logs.append(e)
                    continue
                ents.append(e)

            if len(ents) == 0:
                return [], error_logs
            
            return ents, error_logs


################################################################################################

class DataObjHandler:
    def __init__(self, dataobj_synonyms_path, dataobj_ontology_path):

        # load synonyms file
        with open(dataobj_synonyms_path) as fin:
            d = yaml.safe_load(fin)
        ret = dict()
        for name, li in d.items():
            for synonym in (li or []):
                ret[synonym] = name
        self.dataobj_map = ret

        # init ontology
        self.data_onto = ontutils.loadDataOntology(dataobj_ontology_path)

        # mainly used for func fixLemma (i.e., tokenize and lemma form)
        self.nlp = spacy.load(r"D:\PPAudit\other_data\NlpFinalModel_Policheck")
    
    def fixLemma(self, txt):  # remove spacy.symbols.DET, and return after lemma
        def getLemma(tok):
            return tok.lemma_ if tok.lemma_ != '-PRON-' else tok.text
        
        doc = self.nlp(txt)
        return ' '.join([getLemma(t) for t in doc if t.pos != spacy.symbols.DET])
    
    def get_synonym(self, text):
        term = preprocess_term(self.fixLemma(text))
        
        if term in self.dataobj_map:
            return self.dataobj_map[term]
        else:
            # Strip apostrophe and quotes
            term = re.sub(r'("|\'(\s*s)?)', '', term)
            return self.dataobj_map.get(term, term)
    
    def process_d(self, d):
        dproc = self.get_synonym(d)

        if dproc.strip() == '' or dproc == 'IGNORE':
            return [], []

        if dproc in self.data_onto.nodes:
            return [dproc], []

        else:
            dataobjs = []
            error_logs = []
            res = re.sub(r'\b(and|or|and/or|\/|&)\b', '\n', dproc)
            for d in res.split('\n'):
                d = d.strip()
                d = self.get_synonym(d)
                if d not in self.data_onto.nodes:
                    error_logs.append(d)
                    continue
                dataobjs.append(d)

            if len(dataobjs) == 0:
                return [], error_logs
            
            return dataobjs, error_logs

##########################################################################################################

# load synonyms and ontologies
entity_handler = EntityHandler(
    entity_synonyms_path=r"D:\PPAudit\phrase_to_term\synonyms_and_ontologies\entity_synonyms.yml",
    entity_ontology_path=r'D:\PPAudit\phrase_to_term\synonyms_and_ontologies\entity_ontology.gml'
)

dataobj_handler = DataObjHandler(
    dataobj_synonyms_path=r"D:\PPAudit\phrase_to_term\synonyms_and_ontologies\data_synonyms.yml",
    dataobj_ontology_path=r'D:\PPAudit\phrase_to_term\synonyms_and_ontologies\data_ontology.gml'
)

# load first parties
with open(r'D:\PPAudit\crawl_data\app_info.json', 'r') as rf:
    app_info_json = json.load(rf)
with open(r'D:\PPAudit\output\first_party.json', 'r') as rf:
    first_parties = json.load(rf)

# output dir
term_output_dir = Path(r'D:\PPAudit\output\cus_term_tuples')
term_output_dir.mkdir(parents=False, exist_ok=True)

# process cus tuples to terms
def shouldIgnoreSentence(s):
    mentionsChildRegex = re.compile(r'\b(child(ren)?|kids|from\sminor(s)?|under\s1[0-9]+|under\s(thirteen|fourteen|fifteen|sixteen|seventeen|eighteen)|age(s)?(\sof)?\s1[0-9]+|age(s)?(\sof)?\s(thirteen|fourteen|fifteen|sixteen|seventeen|eighteen))\b', flags=re.IGNORECASE)
    mentionsUserChoiceRegex = re.compile(r'\b(you|user)\s(.*\s)?(choose|do|decide|prefer)\s.*\s(provide|send|share|disclose)\b', flags=re.IGNORECASE)
    mentionsUserChoiceRegex2 = re.compile(r'\b((your\schoice)|(you\sdo\snot\shave\sto\sgive))\b', flags=re.IGNORECASE)
    # TODO remove false positives that discuss "except as discussed in this privacy policy / below"
    mentionsExceptInPrivacyPol1 = re.compile(r'\b(except\sas\s(stated|described|noted))\b', flags=re.IGNORECASE)
    mentionsExceptInPrivacyPol2 = re.compile(r'\b(except\sin(\sthose\slimited)?\s(cases))\b', flags=re.IGNORECASE)

    if mentionsChildRegex.search(s) or mentionsUserChoiceRegex.search(s) or mentionsUserChoiceRegex2.search(s) or mentionsExceptInPrivacyPol1.search(s) or mentionsExceptInPrivacyPol2.search(s):
        return True
    return False


def formalize_cus_tuple(e, c, d, s, first_party): 
    # formalize for a phrase tuple
    # Return:
    #   formalized results
    #   unformalized entity
    #   unformalized data obj
    if c == 'not_collect' and shouldIgnoreSentence(s):
        return [], [], []
    else:
        formalized_e, error_logs_e = entity_handler.process_e(e, first_party)
        formalized_d, error_logs_d = dataobj_handler.process_d(d)
        formalized_tuples = []
        for ent in formalized_e:
            for dataobj in formalized_d:
                formalized_tuples.append((ent, dataobj))
        return formalized_tuples, error_logs_e, error_logs_d
    

def formalize_cus_tuple_for_app(app_link_hash, pp_cus_phrases_file:Path):
        
    print('-----------------------parsing {} in {}-----------------------------'.format(app_link_hash, pp_cus_phrases_file.name.split('.')[0]))
    
    formalized_tuples, error_logs_e, error_logs_d = [], [], []
    
    with open(pp_cus_phrases_file, 'r') as rf:
        cus_phrase_tuples = json.load(rf)[pp_cus_phrases_file.stem]

    first_party = first_parties[app_link_hash] if app_link_hash in first_parties.keys() else []

    for cus_phrase_tuple in cus_phrase_tuples:
        e = cus_phrase_tuple['entity_phrase']
        c = cus_phrase_tuple['cus_or_not']
        d = cus_phrase_tuple['dataobj_phrase']
        s = cus_phrase_tuple['sentence']
        term_tuples, errors_e, errors_d = formalize_cus_tuple(e, c, d, s, first_party)

        for term_tuple in term_tuples:
            formalized_tuples.append({
                'sentence': s,
                'entity_phrase': e,
                'entity_term': term_tuple[0],
                'cus_or_not': c,
                'dataobj_phrase': d,
                'dataobj_term': term_tuple[1],
                'cus_verb': cus_phrase_tuple['cus_verb']
            })
        
        for unrecognize_e in errors_e:
            error_logs_e.append({
                'sentence': s,
                'entity_phrase': e,
                'unterm_entity': unrecognize_e,
                'cus_or_not': c,
                'dataobj_phrase': d,
                'cus_verb': cus_phrase_tuple['cus_verb']
            })
        
        for unrecognize_d in errors_d:
            error_logs_d.append({
                'sentence': s,
                'entity_phrase': e,
                'cus_or_not': c,
                'dataobj_phrase': d,
                'unterm_dataobj': unrecognize_d,
                'cus_verb': cus_phrase_tuple['cus_verb']
            })

    with open(term_output_dir/'{}.json'.format(app_link_hash), 'w') as wf:
        wf.write(json.dumps(formalized_tuples, indent=4))
    with open(term_output_dir/'{}.error-e.json'.format(app_link_hash), 'w') as wf:
        wf.write(json.dumps(error_logs_e, indent=4))
    with open(term_output_dir/'{}.error-d.json'.format(app_link_hash), 'w') as wf:
        wf.write(json.dumps(error_logs_d, indent=4))


if __name__ == '__main__':
    for app_link_hash, this_app_info in app_info_json.items():
        if this_app_info['pp_html_file']:
            pp_cus_phrases_file = Path(r'D:\PPAudit\output\cus_phrase_tuples') / '{}.txt.json'.format(this_app_info['pp_html_file'])
            if pp_cus_phrases_file.is_file():
                formalize_cus_tuple_for_app(app_link_hash, pp_cus_phrases_file)
                break
