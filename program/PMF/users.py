import numpy as np
import sqlite3

class User:
    def __init__(self, config):
        self.initial = config['initial']
        self.userid = config['userid']
        self.preference = list()
        # self.save_to_database(self.userid, self.initial)

    # action: 0 for indifferent, 1 for viewed , 2 for interested
    @staticmethod
    def behaviour(paperid, userid, action):
        return np.array([paperid, userid, action])

    # @staticmethod
    # def save_to_database(user_id, user_type):
    #     conn = sqlite3.connect('../testdata.db')
    #     cursor = conn.cursor()
    #     user_id = int(user_id)
    #     try:
    #         cursor.execute("INSERT INTO users (userid, usertype) VALUES (?, ?)", (user_id, user_type))
    #         conn.commit()
    #         conn.close()
    #     except:
    #         cursor.execute('''CREATE TABLE IF NOT EXISTS users (userid text, usertype text)''')
    #         cursor.execute("INSERT INTO users (userid, usertype) VALUES (?, ?)", (user_id, user_type))
    #         conn.commit()
    #         conn.close()

class UserGroup(User):
    def __init__(self, user_num, config_list):
        self.user_list = list()
        for i in range(user_num):
            user = User.__init__(self, config=config_list[i])
            self.user_list.append(user)


def fake_users(user_num, paper_num, instances):
    user_id = np.random.random_integers(low=0, high=user_num, size=instances)
    user_set = set(user_id)
    paper_id = np.random.random_integers(low=0, high=paper_num, size=instances)
    item_set = set(paper_id)
    a = paper_id.T
    user_paper = np.vstack((user_id, paper_id.T))
    action = np.random.random_integers(low=0, high=2, size=instances)
    data = np.vstack((user_paper, action))
    print(data)
    data = data.T
    np.savetxt('fake.txt', data)
    return data, list(data)


# res1, res2 = fake_users(3000, 30000, 12000)
# print(1)
