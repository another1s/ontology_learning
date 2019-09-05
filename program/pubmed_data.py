from Bio import Entrez
from program.ds import Paper
import csv
import sqlite3

import json
#  "Anatomy", "Organisms", "Diseases", "Chemicals and Drugs", "Analytical, Diagnostic and Therapeutic Techniques, and Equipment",
#  "Psychiatry and Psychology", "Phenomena and Processes", "Disciplines and Occupations", "Anthropology, Education, Sociology, and Social Phenomena",
# "

def bio_category():
    categories = list()
    return categories

def save_to_local_v1(papers):
    filename0 = '../dataset/pubmed_data/papers.csv'
    with open(filename0, 'a+', encoding='utf-8') as f2:
        writer = csv.writer(f2)
        writer.writerow(['title', 'abstract', 'features', 'authorlist'])
        for paper in papers:
            writer.writerow([0, paper['mainTitle'], paper['abstractContent'], paper['collections'], paper['publisher']])
        f2.close()

def save_to_local_v2(papers):
    conn = sqlite3.connect('test.db')
    cur = conn.cursor()
    # Create table
    c.execute('''CREATE TABLE papers
                 (date text, trans text, symbol text, qty real, price real)''')
    for paper in papers:
        c.execute("INSERT INTO papers VALUES ('2006-01-05','BUY','RHAT',100,35.14)")

    # Save (commit) the changes
    conn.commit()

    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    conn.close()


def search(query):
    Entrez.email = 'hzythesecond@gmail.com'
    handle = Entrez.esearch(db='pubmed', sort='relevance', retmax='1',retmode='xml', term=query)
    results = Entrez.read(handle)
    return results


def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'hzythesecond@gmail.com'
    handle = Entrez.efetch(db='pubmed',retmode='xml',id=ids)
    results = Entrez.read(handle)
    return results


if __name__ == '__main__':
    Json_lized = Paper()
    results = search('Anatomy')
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
