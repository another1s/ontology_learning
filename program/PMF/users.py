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


def fake_users(user_num):

    return