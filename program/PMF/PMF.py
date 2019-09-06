import numpy as np
from program.PMF.users import fake_users


class PMF:
    def __init__(self, pretrained_addr=None, userid=None, itemid=None):
        self.model = dict()
        if pretrained_addr:
            self.pretrained = self.load_model(pretrained_addr)
        if userid and itemid:
            self.userid = userid
            self.itemid = itemid

    # U: [num_user, hidden_feature], V:[num_item, hidden_feature]
    def train(self, num_user, num_item, train, test, learning_rate, K, regu_u, regu_i, maxiter, update: 'bool'):
        if self.model:
            U = self.model['U']
            V = self.model['V']
        elif update is True and self.pretrained:
            U = self.pretrained['U']
            V = self.pretrained['V']
        else:
            U = np.random.normal(0, 0.1, (num_user, K))
            V = np.random.normal(0, 0.1, (num_item, K))
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


def read_data(path,train_ratio):
    user_set = {}
    item_set = {}
    u_idx = 0
    i_idx = 0
    data = []
    with open(path) as f:
        for line in f.readlines():
            u, i, r, _ = line.split('::')
            if u not in user_set:
                user_set[u] = u_idx
                u_idx += 1
            if i not in item_set:
                item_set[i] = i_idx
                i_idx += 1
            data.append([user_set[u], item_set[i], float(r)])

    np.random.shuffle(data)
    train = data[0:int(len(data)*train_ratio)]
    test = data[int(len(data)*train_ratio):]
    return u_idx, i_idx, train, test


if __name__=='__main__':
    num_user, num_item, train, test = read_data('./data/ratings.dat', 0.8)
    pmf = PMF()
    pmf.train(num_user, num_item, train, test, 0.01, 10, 0.01, 0.01, 100, False)
    predict_rate, userid, paperid = pmf.predict([2312, 3])
    print(predict_rate, ' ', userid, ' ', paperid)
    added_train, train_list = fake_users(user_num=1000, paper_num=5000, instances=3000)
    #pmf.train(num_user=1000, num_item)
    #pmf.train(num_user=1000)


