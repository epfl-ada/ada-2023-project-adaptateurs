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

def get_gender_id(QID):
    if QID == None:
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
        return value
    except KeyError:
        return 0.