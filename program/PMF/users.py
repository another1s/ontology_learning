import numpy as np


class Users:
    def __init__(self, config):
        self.initial = config['initial']
        self.userid = config['userid']
        self.preference = list()

    # action: 0 for indifferent, 1 for viewed , 2 for interested
    @staticmethod
    def behaviour(paperid, userid, action):
        return np.array([paperid, userid, action])


class UserGroup(Users):
    def __init__(self, user_num, config_list):
        self.user_list = list()
        for i in range(user_num):
            user = Users.__init__(self, config=config_list[i])
            self.user_list.append(user)


def fake_users(user_num, paper_num, instances):
    user_id = np.random.random_integers(low=0, high=user_num, size=instances)
    paper_id = np.random.random_integers(low=0, high=paper_num, size=instances)
    a = paper_id.T
    user_paper = np.vstack((user_id, paper_id.T))
    action = np.random.random_integers(low=0, high=2, size=instances)
    data = np.vstack((user_paper, action))
    print(data)
    data = data.T
    np.savetxt('fake.txt', data)
    return data, list(data)


res = ['wew', 'ewew', 'weww1']
print(str(res))
# res1, res2 = fake_users(3000, 30000, 12000)
# print(1)