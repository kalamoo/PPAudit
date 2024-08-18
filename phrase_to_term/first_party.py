import re
import json
import time
import tldextract
# pip install tldextract
import spacy
import spacy.lang.en
from pathlib import Path

other_data_dir = Path(r'D:\PPAudit\other_data')
NLP_MODEL = other_data_dir / 'NlpFinalModel_Policheck'
crawl_data_dir = Path(r'D:\PPAudit\crawl_data')
app_info_json = crawl_data_dir / 'app_info.json'
output_dir = Path(r'D:\PPAudit\output')

with open(app_info_json, 'r') as rf:
    app_info = json.load(rf)

IGNORE_ENTITY_NAMES = set(spacy.lang.en.stop_words.STOP_WORDS)
IGNORE_ENTITY_NAMES.update({"com", "android", "free", "paid", "co"})
IGNORE_ENTITY_NAMES.update({"provider", "company", "website", "inc", "llc"})


def is_entity_valid(ent):
    flag = ent.label_ in {'PERSON', 'ORG'}
    flag = flag and ent.text.lower() not in IGNORE_ENTITY_NAMES
    flag = flag and re.search('[A-Z]', ent.text)
    return flag


def trim_entity(e):
    e = re.sub(r'\W+(ltd|llc|inc)[^\w]*$', '', e, flags=re.I)
    e = re.sub(r'^\W+', '', e)
    e = re.sub(r'\W+$', '', e)
    return e


def extract_first_party_name():
    nlp = spacy.load(NLP_MODEL)
    first_party_dict = dict()

    for app_link_url, item in app_info.items():
        first_party = []
        pp_url_hash = item['pp_html_file']

        # attempt to extract 1st-part only when we have to analyze their pp text
        if not (output_dir/'pp_txts'/'{}.txt'.format(pp_url_hash)).is_file():
            continue

        print('[+] {}'.format(pp_url_hash))
        
        # YZ: developer / app name
        if item['publisher_simple'] and item['publisher_simple'] != '': 
            first_party.append(item['publisher_simple'])
        if item['name_simple'] and item['name_simple'] != '': 
            first_party.append(item['name_simple'])
        
        # YZ: eSLD of website and pp link
        if item['app_link_home']:
            first_party.append(tldextract.extract(item['app_link_home']).domain)
        if item['pp_link']:
            first_party.append(tldextract.extract(item['pp_link']).domain)
        
        # from pp text file indication
        with open(output_dir/'pp_txts'/'{}.txt'.format(pp_url_hash), 'r', encoding='utf-8', errors='?') as rf:
            lines = rf.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Fix "xxx LLC" entity name
            line = line.replace("LLC", "Inc")
            
            if 'all rights reserved' in line.lower():
                # case 1: copyright <company name> all rights reserved
                doc = nlp(line)
                for sentence in doc.sents:
                    if 'all rights reserved' in sentence.text.lower():
                        for ent in doc.ents:
                            if is_entity_valid(ent):
                                first_party.append(
                                    re.sub(r'(?:copyright\s*)?20[12]\d\W*', '', ent.text, flags=re.I))
            
            if re.search(r"([\"'])(?:we|us|our)[^\W]*\1", line, re.I):
                # case 2: company name ("we"/"us"/"other names")
                doc = nlp(line)
                for sentence in doc.sents:
                    bra_pairs = []
                    bra_stack = []
                    for idx, tok in enumerate(sentence):
                        if tok.text == '(':
                            bra_stack.append(idx)
                        elif tok.text == ')' and len(bra_stack) > 0:
                            left_idx = bra_stack.pop()
                            if len(bra_stack) == 0:
                                bra_text = sentence[left_idx + 1: idx].text
                                # [NOTE] search "(we)" in this (..) span
                                if re.search(r"([\"'])(?:we|us|our)[^\W]*\1", bra_text, re.I):
                                    bra_pairs.append((left_idx, idx))
                    for left_idx, right_idx in bra_pairs:
                        for ent in sentence.ents:
                            # [NOTE] find XXX(we) or (we XXX)
                            if (left_idx >= ent.end and re.search(r'\w', sentence[ent.end:left_idx].text) is None) \
                                    or (ent.start >= left_idx and ent.end < right_idx):
                                if is_entity_valid(ent):
                                    first_party.append(ent.text)
            
            if line.endswith("Privacy Policy") or line.endswith("PRIVACY POLICY"):
                # case 3: extract company/app name from title ("XXX GAME PRIVACY POLICY")
                doc = nlp(line)
                if len(list(doc.sents)) > 1 or len(doc.ents) != 1:
                    continue
                if is_entity_valid(doc.ents[0]):
                    first_party.append(doc.ents[0].text)

        first_party = [e for e in list(set(first_party)) if e != '']
        print(first_party)
        first_party_dict[app_link_url] = first_party

    with open(output_dir/'first_party.json', 'w') as wf:
        wf.write(json.dumps(first_party_dict, indent=4))


if __name__ == '__main__':
    start_time = time.time()
    extract_first_party_name()
    end_time = time.time()
    print("[LOG] collect first party cost {} s".format(end_time - start_time))
