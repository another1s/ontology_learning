

class Paper:
    def __init__(self):
        self.paper_list = list()

    def pubmed(self, data):
        collection = list()
        pmid = str(data['PMID'])
        if len(data['Article']['ELocationID']):
            elocationid = str(data['Article']['ELocationID'][0])
        else:
            elocationid = "None"
        values = [pmid, elocationid]
        keys = ['PMID', 'ELocationID']
        otherid = dict(zip(keys, values))

        maintitle = str(data['Article']['ArticleTitle'])
        subtitle = 'None'
        abstract_content = ''
        abstract = list()
        if data['Article'].get('Abstract'):
            if data['Article']['Abstract'].get('AbstractText'):
                abstract = data['Article']['Abstract']['AbstractText']
            for part in abstract:
                abstract_content = abstract_content + part
        content = 'None'

        references = list()
        references.append("None")
        # url_key = ['pdf_url', 'abstract_url']
        # url_value = [data['pdf_url'], data['abstract_url']]
        originUrl = 'None'

        citation = list()
        citation.append('None')
        if len(data['KeywordList']):
            subjectareas = data['KeywordList']
            subjectarea = subjectareas[0]
            for area in subjectarea:
                collection.append(str(area))

        publishdate = ''
        submitdate = ''
        author = list()
        if data['Article'].get('AuthorList'):
            author = data['Article']['AuthorList']
        authorlist = list()
        if len(author):
            for a in author:
                forename = "None"
                lastname = "None"
                if a.get('ForeName') and a.get('LastName'):
                    lastname = a['LastName']
                    forename = a['ForeName']
                    authorlist.append(forename + ' ' + lastname)
                elif a.get('CollectiveName'):
                    authorlist.append((a['CollectiveName']))

        language = 'English'
        publication_type = ''
        for t in data['Article']['PublicationTypeList']:
            publication_type = publication_type + t

        key_entire = ['otherId', 'mainTitle', 'subTitle', 'abstractContent', 'content', 'references', 'originUrl',
                      'citation', 'collections', 'publishDate', 'submitDate', 'publisher', 'language',
                      'publicationType']
        value_entire = [otherid, maintitle, subtitle, abstract_content, content, references, originUrl, citation,
                        collection, publishdate, submitdate, authorlist, language, publication_type]

        data_modified = dict(zip(key_entire, value_entire))
        return data_modified


