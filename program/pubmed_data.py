from Bio import Entrez
from program.ds import Paper
import csv

import json
#  "Anatomy", "Organisms", "Diseases", "Chemicals and Drugs", "Analytical, Diagnostic and Therapeutic Techniques, and Equipment",
#  "Psychiatry and Psychology", "Phenomena and Processes", "Disciplines and Occupations", "Anthropology, Education, Sociology, and Social Phenomena",
# "

def bio_category():
    categories = list()
    return categories

def save_to_local_v1(papers):
    filename0 = '../dataset/pubmed_data/'
    with open(filename0, 'a+', encoding='utf-8') as f2:
        writer = csv.writer(f2)
        writer.writerow(['title', 'abstract', 'features', 'authorlist'])
        for paper in papers:
            writer.writerow([11, paper['mainTitle'], paper['abstractContent'], paper['collections'], paper['publisher']])
        f2.close()

def save_to_local_v2(papers):
    filename0 = 'E:/btd/mih/data/file4.csv'
    with open(filename0, 'a', encoding='utf-8') as f2:
        writer = csv.writer(f2)
        for paper in papers:
            writer.writerow([1, paper['mainTitle'], paper['abstractContent']])

def search(query):
    Entrez.email = 'hzythefirst@gmail.com'
    handle = Entrez.esearch(db='pubmed', sort='relevance', retmax='10000',retmode='xml', term=query)
    results = Entrez.read(handle)
    return results


def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'hzythefirst@gmail.com'
    handle = Entrez.efetch(db='pubmed',retmode='xml',id=ids)
    results = Entrez.read(handle)
    return results


if __name__ == '__main__':
    Json_lized = Paper()
    results = search('Publication Characteristics')
    id_list = results['IdList']
    papers = fetch_details(id_list)
    publications = list()
    for i, paper in enumerate(papers['PubmedArticle']):
        #print("%d) %s" % (i+1, paper['MedlineCitation']['Article']['ArticleTitle']))
        #print("%d) %s" % (i + 1, paper['MedlineCitation']['Article']['Abstract']))
        c = paper['MedlineCitation']
        b = Json_lized.pubmed(paper['MedlineCitation'])
        Json_lized.paper_list.append(b)
        #print(b)
    save_to_local_v1(Json_lized.paper_list)
