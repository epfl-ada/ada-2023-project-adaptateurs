import wikipediaapi
from wikibaseintegrator import WikibaseIntegrator
from wikibaseintegrator.wbi_config import config as wbi_config
from wikibaseintegrator import wbi_helpers
import requests
wbi_config['USER_AGENT'] = 'MyWikibaseBot/1.0 (https://www.wikidata.org/wiki/User:Ada-2023-project-adaptateurs)'

def get_gender_wiki(wikiID, job):
    dictory = {
        'director' : 'P57',
        'producer' : 'P162',
        'writer' : 'P58'
    }
    key = dictory[job]
    wiki_wiki = wikipediaapi.Wikipedia('ada (ada@epfl.ch)', 'en')
    wbi = WikibaseIntegrator()
    page_id = wiki_wiki.page(wikiID)
    page_name = requests.get(f'http://en.wikipedia.org/w/api.php?action=query&pageids={wikiID}&format=json').json()['query']['pages'][wikiID]#['title']
    qid_movie = requests.get(f'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles={page_name}&format=json').json()['query']['pages'][wikiID]['pageprops']['wikibase_item']
    movie_item = wbi.item.get(entity_id=qid_movie)
    qid_directors = [] 
    for element in movie_item.get_json()['claims']['P57']:
        qid_directors.append(element['mainsnak']['datavalue']['value']['id'])
    genders = []
    for element in qid_directors:
        genders.append(wbi.item.get(entity_id=element).get_json()['claims']['P21'][0]['mainsnak']['datavalue']['value']['id'])
    return genders

def get_gender_imdb(imdb_id, job, provided_name):
    try:
        imdb_id = '"tt'+ imdb_id + '"'
        dictory = {
            'director' : 'P57',
            'producer' : 'P162',
            'writer' : 'P58'
        }
        key = dictory[job]
        wbi = WikibaseIntegrator()
        query = wbi_helpers.execute_sparql_query('SELECT ?film ?filmLabel WHERE {?film wdt:P345 '+imdb_id+'.SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}')
        query = query['results']['bindings'][0]['film']['value'].split('/')[-1]
        
        movie_item = wbi.item.get(entity_id=query)
        qid_directors = [] 

        for element in movie_item.get_json()['claims'][key]:
            qid_directors.append(element['mainsnak']['datavalue']['value']['id'])
        genders = []
        for element in qid_directors:
            item_get = wbi.item.get(entity_id=element).get_json()
            name = item_get['labels']['en']['value']
            item = item_get['claims']
            gender = item['P21'][0]['mainsnak']['datavalue']['value']['id']
            try: 
                if name.lower() == provided_name.lower():
                    return gender
            except:
                first_name_id = item['P735'][0]['mainsnak']['datavalue']['value']['id']
                last_name_id = item['P734'][0]['mainsnak']['datavalue']['value']['id']
                first_name = wbi.item.get(entity_id=first_name_id).get_json()['labels']['en']['value']
                last_name = wbi.item.get(entity_id=last_name_id).get_json()['labels']['en']['value']    
                name = first_name + ' ' + last_name
                provided_name = provided_name.split(' ')
                provided_name = provided_name[0] + ' ' + provided_name[-1]
                if name.lower() == provided_name.lower():
                    return gender
        return None
    except:
        print('error with ', provided_name)
        return None


def get_IMDB_ID(wikiID):
    wiki_wiki = wikipediaapi.Wikipedia('ada (ada@epfl.ch)', 'en')
    wbi = WikibaseIntegrator()
    page_id = wiki_wiki.page(wikiID)
    page_name = requests.get(f'http://en.wikipedia.org/w/api.php?action=query&pageids={wikiID}&format=json').json()['query']['pages'][wikiID]['title']
    qid_movie = requests.get(f'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles={page_name}&format=json').json()['query']['pages'][wikiID]['pageprops']['wikibase_item']
    movie_item = wbi.item.get(entity_id=qid_movie)
    IMDB_ID = movie_item.get_json()['claims']['P345'][0]['mainsnak']['datavalue']['value']
    return IMDB_ID

def get_gender_text(QID):
    dictory = {
        'male' : 'Q6581097',
        'female' : 'Q6581072',
        'intersex' : 'Q1097630',
        'trans women' : 'Q1052281',
        'trans men' : 'Q2449503',
        'non-binary' : 'Q48270',
        'fa\'afafine' : 'Q1399232',
        'mahu' : 'Q3277905',
        'kathoey' : 'Q746411',
        'fakaleiti' : 'Q350374',
        'hijra' : 'Q660882',
        'two-spirit' : 'Q301702',
        'transmasculine' : 'Q27679766',
        'transfeminine' : 'Q27679684',
        'muxe salu' : 'Q3177577',
        'agender' : 'Q505371',
        'genderqueer' : 'Q12964198',
        'genderfluid' : 'Q18116794',
        'neutrois' : 'Q1289754',
        'eunuch' : 'Q179294',
        'pangender' : 'Q7130936',
        'neutral sex' : 'Q52261234',
        'hermaphrodite' : 'Q16674976',
        'cisgender woman' : 'Q15145779',
        'cisgender man' : 'Q15145778',
        'third gender' : 'Q48279',
        'X-gender' : 'Q96000630',
        'demiboy' : 'Q93954933',
        'demigirl' : 'Q93955709',
        'bigender' : 'Q859614',
        'transgender' : 'Q189125',
        'travesti' : 'Q17148251',
        'akava\'ine' : 'Q4700377',
        'assigned female at birth' : 'Q56315990',
        'assigned male at birth' : 'Q25388691',
        'androgyne' : 'Q97595519',
        'yinyang ren' : 'Q8053770',
        'boi' : 'Q99519347',
        'intersex person' : 'Q104717073',
        'gynandromorphism' : 'Q430711',
        'takatapui' : 'Q7677449',
        'undisclosed' : 'Q113124952',
        'fakafifine' : 'Q112597587',
        'androgynos' : 'Q4759445',
        'intersex man' : 'Q121307094',
        'intersex woman' : 'Q121307100',
        'demimasc' : 'Q121368243'
    }
    inv_map = {v: k for k, v in dictory.items()}
    try:
        value = inv_map[QID]
        return value
    except KeyError:
        print("No gender found gender id")

def get_gender_id(QID):
    if QID == None:
        print('got none')
        return 0.
    dictory = {
        'Q6581097' : 2.0, 
        'Q6581072' : 1.0,
        'Q1052281' : 1.0,
        'Q2449503' : 2.0,
        'Q15145779' : 1.0,
        'Q15145778' : 2.0,
        'Q121307094' : 2.0,
        'Q121307100' : 1.0,
    }
    try:
        value = dictory[QID]
        print('got value ', value)
        return value
    except KeyError:
        print("No gender found from gender id.")