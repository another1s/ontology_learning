import requests as req
import json

class trans_data:
    def __init__(self):
        self.header = dict()
        self.END_POINT = "http://35.236.122.104:8080"
        self.AUTH_PATH = "/api/test/auth/guest"
        self.TEST_PATH = "/api/test/crawler/publication/bulk"
        self.TEST_PATH0 = "/api/test/crawler/peoplenametopublication/bulk"

    def get_data(self, endpoint):
        response = req.get(url=endpoint + self.AUTH_PATH)
        return response.text

    def post_data(self, endpoint, data):
        t = json.loads((self.get_data(self.END_POINT)))
        self.header['X_Auth_Token'] = t['Token']
        #print(json.dumps(data[0], indent=4))
        url = endpoint+self.TEST_PATH0
        #d =data[0]
        response = req.post(url=endpoint + self.TEST_PATH, headers=self.header, json=data)
      #  print(json.loads(response.text))
        return response.text
