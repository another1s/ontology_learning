import numpy as np
from program.PMF.users import fake_users, User, UserGroup
import sqlite3

class PMF:
    def __init__(self, pretrained_addr=None, user_set=None, item_set=None):
        self.model = dict()
        if pretrained_addr:
            self.pretrained = self.load_model(pretrained_addr)
        if user_set and item_set:
            self.userid = user_set
            self.itemid = item_set

    # U: [num_user, hidden_feature], V:[num_item, hidden_feature]
    def train(self, num_user, num_item, train, test, learning_rate, K, regu_u, regu_i, maxiter, update: 'bool'):
        if update is True and self.pretrained:
            U = self.pretrained['U']
            V = self.pretrained['V']
        else:
            U = np.random.normal(0, 0.1, (num_user, K))
            V = np.random.normal(0, 0.1, (num_item, K))

        # U_add = np.random.normal(0, 0.1, (1, K))
        # V_add = np.random.normal(0, 0.1, (1, K))
        # U = np.vstack((U, U_add))
        # V = np.vstack((V, V_add))

        pre_rmse = 100.0
        endure_count = 3
        patience = 0
        for iter in range(maxiter):
            loss = 0.0
            for data in train:
                user = data[0]
                item = data[1]
                rating = data[2]
                predict_rating = np.dot(U[user], V[item].T)
                error = rating-predict_rating
                loss += error**2
                U[user] += learning_rate*(error*V[item]-regu_u*U[user])
                V[item] += learning_rate*(error*U[user]-regu_i*V[item])
                loss += regu_u*np.square(U[user]).sum()+regu_i*np.square(V[item]).sum()
            loss = 0.5*loss
            rmse = self.eval_rmse(U, V, test)
            print('iter:%d loss:%.3f rmse:%.5f'%(iter, loss, rmse))
            if rmse < pre_rmse:   # early stop
                pre_rmse = rmse
                patience = 0
            else:
                patience += 1
            if patience >= endure_count:
                self.save_model(U=U, V=V)
                break

    # user's behaviour changed or added
    def iterative_training(self, num_user_added, num_item_added, train, test, learning_rate, K, regu_u, regu_i, maxiter):
        U = self.model['U']
        V = self.model['V']
        U_add = np.random.normal(0, 0.1, (num_user_added, K))
        V_add = np.random.normal(0, 0.1, (num_item_added, K))
        U = np.vstack((U, U_add))
        V = np.vstack((V, V_add))
        pre_rmse = 100.0
        endure_count = 3
        patience = 0
        for iter in range(maxiter):
            loss = 0.0
            for data in train:
                user = data[0]
                item = data[1]
                rating = data[2]
                predict_rating = np.dot(U[user], V[item].T)
                error = rating - predict_rating
                loss += error ** 2
                U[user] += learning_rate * (error * V[item] - regu_u * U[user])
                V[item] += learning_rate * (error * U[user] - regu_i * V[item])
                loss += regu_u * np.square(U[user]).sum() + regu_i * np.square(V[item]).sum()
            loss = 0.5 * loss
            rmse = self.eval_rmse(U, V, test)
            print('iter:%d loss:%.3f rmse:%.5f' % (iter, loss, rmse))
            if rmse < pre_rmse:  # early stop
                pre_rmse = rmse
                patience = 0
            else:
                patience += 1
            if patience >= endure_count:
                self.save_model(U=U, V=V)
                break

    # input: chosen_user(int) , itemid(list)
    # output: result [itemid, predict_rate]
    def recommendbyrank(self, chosen_user, itemid, rank_num):
        result = []
        for id in itemid:
            predict, user, item = self.predict([chosen_user, id])
            result.append((item, predict))

        sorted_result = sorted(result, key=lambda r: r[1])
        return sorted_result[0:rank_num]

    # input: [user_ind, paper_ind], output: predict_rate(float)
    def predict(self, data):
        user = data[0]
        item = data[1]
        predict_rate = np.dot(self.model['U'][user], self.model['V'][item].T)
        return predict_rate, user, item

    # input:  U [user_num, user_hidden_feature], V[item_num, item_hidden_feature]
    def save_model(self, U, V):
        self.model['U'] = U
        self.model['V'] = V
        np.savetxt('U.txt', U)
        np.savetxt('V.txt', V)
        return self.model

    def load_user_item_set(self, user_set, item_set):
        self.user_set = user_set
        self.item_set = item_set

    @staticmethod
    def eval_rmse(U, V, test):
        test_count = len(test)
        tmp_rmse = 0.0
        for te in test:
            user = te[0]
            item = te[1]
            real_rating = te[2]
            predict_rating = np.dot(U[user], V[item].T)
            tmp_rmse += np.square(real_rating-predict_rating)
        rmse = np.sqrt(tmp_rmse/test_count)
        return rmse

    @staticmethod
    def load_model(address):
        U = np.loadtxt(address + 'U.txt')
        V = np.loadtxt(address + 'V.txt')
        model = {'U': U, 'V': V}
        return model

    # @staticmethod
    # def save_user_behavior(userid, itemid, rate):
    #     conn = sqlite3.connect('testdata.db')
    #     cursor = conn.cursor()
    #     try:
    #         cursor.execute('''''')


def read_data(path, train_ratio):
    user_set = {}
    item_set = {}
    u_idx = 0
    i_idx = 0
    data = []
    with open(path) as f:
        for line in f.readlines():
            u, i, r, _ = line.split('::')
            if u not in user_set:
                config = {'initial': 'real', 'userid': u}
                new_user = User(config=config)
                user_set[u] = u_idx
                u_idx += 1
            if i not in item_set:
                item_set[i] = i_idx
                i_idx += 1
            data.append([user_set[u], item_set[i], float(r)])

    np.random.shuffle(data)
    train = data[0:int(len(data)*train_ratio)]
    test = data[int(len(data)*train_ratio):]
    return u_idx, i_idx, train, test, user_set, item_set


if __name__=='__main__':
    num_user, num_item, train, test, user_set, item_set = read_data('program/PMF/data/ratings.dat', 0.8)
    pmf = PMF()
    added_train, train_list = fake_users(user_num=3, paper_num=120, instances=1200)
    pmf.load_user_item_set(user_set, item_set)
    pmf.train(num_user, num_item, train, test, 0.01, 10, 0.01, 0.01, 100, False)
    user_added = 0
    item_added = 0
    for element in train_list:
        u = element[0]
        v = element[1]
        r = element[2]
        if u not in user_set:
            user_added = user_added + 1
        if v not in item_set:
            item_added = item_added + 1

    #for example in train_list:
    predict_rate, userid, paperid = pmf.predict([1, 3])
    print(predict_rate, ' ', userid, ' ', paperid)

    item_id = [n for n in range(0, 1000)]
    print(pmf.recommendbyrank(chosen_user=1, itemid=item_id, rank_num=10))
    pmf.iterative_training(num_user_added=user_added, num_item_added=item_added, train=train_list, test=test,
                           learning_rate=0.01, K=10, regu_u=0.01, regu_i=0.01, maxiter=100)
    print(pmf.recommendbyrank(chosen_user=1, itemid=item_id, rank_num=10))
    # predict_rate, userid, paperid = pmf.predict([1, 3])
    # print(predict_rate, ' ', userid, ' ', paperid)


