import sys
sys.path.append(r'D:\PPAudit')
from pathlib import Path
import time
import json
from minimization.des_sim import DesSimSolver

class CounterPartFinder:
    def __init__(self):
        with open(r'D:\PPAudit\minimization\recommend_ingredients.json', 'r') as rf:
            self.cp_ingredients = json.load(rf)

        with open(r'D:\PPAudit\crawl_data\app_info.json', 'r') as rf:
            self.app_info = json.load(rf)

        self.desSimSolver = DesSimSolver()
        self.counterpart_result = dict()

    def jaccard_similartiy(self, A, B):
        nominator = A.intersection(B) 
        denominator = A.union(B) 
        if len(denominator) == 0:
            return 0
        similarity = len(nominator) / len(denominator)
        return similarity

    def get_recommend_by_genres(self, app_link_hash, threshold=0.5):
        # FIXME genres may have semantically similar but not identical elements: e.g., history and historical

        genresA = self.cp_ingredients[app_link_hash]['genres']
        if not genresA or len(genresA)==0:
            return []
        recommend_by_genres = []
        for app_link_hashB in self.app_info.keys():
            if app_link_hashB == app_link_hash:
                continue
            genresB = self.cp_ingredients[app_link_hashB]['genres']
            if not genresB or len(genresB)==0:
                continue
            AB_sim = self.jaccard_similartiy(set(genresA), set(genresB))
            if AB_sim > threshold:
                recommend_by_genres.append(app_link_hashB)
        return recommend_by_genres

    def get_description(self, app_link_hash):
        filename = Path(r'D:\PPAudit\output\des_sentences')/'{}.txt'.format(app_link_hash)
        if filename.is_file():
            with open(filename, 'r', encoding='utf-8') as rf:
                description = rf.read()
        else:
            description = None
        return description

    def find_counterparts_single(self, app_link_hash):
        recommend_by_steamvr = [item[0] for item in self.cp_ingredients[app_link_hash]['steamvr-recommend']]
        recommend_by_steampeek = [item[0] for item in self.cp_ingredients[app_link_hash]['steampeek-recommend']]
        recommend_by_genres = self.get_recommend_by_genres(app_link_hash)

        # merge all recomendations, and chose top-k by des similarity
        recommend_all_hashes = list(set(recommend_by_steamvr + recommend_by_steampeek + recommend_by_genres) - {app_link_hash})
        
        recommend_all_appnames = []
        recommend_all_descriptions = []
        
        for recommend_app_hash in recommend_all_hashes:
            if self.get_description(recommend_app_hash):
                recommend_all_appnames.append(self.app_info[recommend_app_hash]['name_simple'])
                recommend_all_descriptions.append(self.get_description(recommend_app_hash))

        top_similar_apps = self.desSimSolver.getDesSim(
            [app_link_hash, ] + recommend_all_hashes,
            [self.app_info[app_link_hash]['name_simple'], ] + recommend_all_appnames,
            [self.get_description(app_link_hash), ] + recommend_all_descriptions,
            mode='All'
        )

        return top_similar_apps

    def find_counterparts_all(self):
        for app_link_hash in self.app_info.keys():        
            if not self.get_description(app_link_hash):
                continue
            print('[+] {}'.format(app_link_hash))
            counterparts = self.find_counterparts_single(app_link_hash)
            self.counterpart_result[app_link_hash] = counterparts
            print(json.dumps(counterparts, indent=4))
        with open(r'D:\PPAudit\minimization\counterparts.json', 'w') as wf:
            to_json = json.dumps(self.counterpart_result, indent=4)
            wf.write(to_json)


if __name__ == '__main__':
    cpFinder = CounterPartFinder()
    start_time = time.time()
    cpFinder.find_counterparts_all()
    end_time = time.time()
    print('[LOG] find counterparts cost {} s.'.format(end_time - start_time))
