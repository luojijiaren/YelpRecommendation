# -*- coding: utf-8 -*-
"""
Created on Tue Apr  20 13:59:51 2018

@author: fzhan
"""

import numpy as np
import csv
from sklearn.metrics import confusion_matrix
class Collab_Filtering(object):
    def __init__(self):
        self._training_set = []
        self._dict_business = {}
        self._num_business = 0
        self._dict_user = {}
        self._num_user = 0
        self._UBRR = {} ### User-Business Reviews Rating
        self._BURR = {} ### Business-User Reviews Rating
        self._dict_business_avg = {}
        self._dict_user_avg = {}
        self._sorted = {}
        self._sim_dict = {}
        self._mu = 0

    def dict_business_lookup(self,business_id):
        if not business_id in self._dict_business.keys():
            self._dict_business[business_id] = self._num_business
            business = self._num_business
            self._num_business += 1
        else:
            business = self._dict_business[business_id]
        return business

    def dict_user_lookup(self,user_id):
        if not user_id in self._dict_user.keys():
            self._dict_user[user_id] = self._num_user
            user = self._num_user
            self._num_user += 1
        else:
            user = self._dict_user[user_id]
        return user

    def load_dataset_csv(self,filename):
        input = []
        with open(filename+'.csv') as csvfile:
            file = csv.reader(csvfile, delimiter=',', quotechar='|')
            co = 0
            header_passed = False
            for row in file:
                if not header_passed:
                    header_passed = True
                    continue
                user_id = row[0]
                business_id = row[1]
                rating = int(row[2])
                user = self.dict_user_lookup(user_id)
                business = self.dict_business_lookup(business_id)
                input.append([user,business,rating])
        input = np.array(input)
        np.save(filename+'.npy',input)
        return input

    def load_dataset_npy(self, filename):
        return np.load(filename+'.npy')

    def get_avg_rating(self,dict,id):
        t = dict[id]
        return t[0]/t[1]

    def get_baseline_estimate(self,user,business):
        bu = 0
        if user in self._dict_user_avg.keys():
            bu = self.get_avg_rating(self._dict_user_avg,user)-self._mu
        bb = 0
        if business in self._dict_business_avg.keys():
            bb = self.get_avg_rating(self._dict_business_avg,business)-self._mu
        return self._mu+bu+bb

    def train(self,input):
        self._training_set = input
        for i in range(self._training_set.shape[0]):
            user = int(self._training_set[i][0])
            business = int(self._training_set[i][1])
            rating = int(self._training_set[i][2])

            if not business in self._dict_business_avg:
                self._dict_business_avg[business] = np.array([0,0])
            self._dict_business_avg[business] += [rating,1]
            if not user in self._dict_user_avg:
                self._dict_user_avg[user] = np.array([0,0])
            self._dict_user_avg[user] += [rating,1]

            if not user in self._UBRR.keys():
                self._UBRR[user] = {}
            self._UBRR[user][business] = rating
            if not business in self._BURR.keys():
                self._BURR[business] = {}
            self._BURR[business][user] = rating
        print('users',len(self._dict_user_avg.keys()))
        print('business',len(self._dict_business_avg.keys()))
        print('reviews',input.shape[0])
        s = 0
        co=0
        for business in self._dict_business_avg.keys():
            s+=self.get_avg_rating(self._dict_business_avg,business)
            co+=1
        self._mu = s/co
        ss_dict = {}
        for b_key, b_list in self._UBRR.items():
            for id1, val1 in b_list.items():
                for id2, val2 in b_list.items():
                    if id1 < id2:
                        avg1 = self.get_avg_rating(self._dict_business_avg,id1)
                        avg2 = self.get_avg_rating(self._dict_business_avg,id2)
                        key = (id1, id2)
                        if not key in ss_dict.keys():
                            ss_dict[key] = np.array([0., 0., 0., 0.])
                        xy = (val1 - avg1) * (val2 - avg2)
                        xx = (val1 - avg1) * (val1 - avg1)
                        yy = (val2 - avg2) * (val2 - avg2)
                        ss_dict[key] += np.array([xy, xx, yy, 1])
        result = np.zeros((len(ss_dict.keys()), 6))
        co = 0
        for key, val in ss_dict.items():
            result[co] = [key[0], key[1], val[0], val[1], val[2], val[3]]
            co += 1
        ss_dict= result
        ss_dict[ss_dict[:, 3] == 0, 3] = 1
        ss_dict[ss_dict[:, 4] == 0, 4] = 1
        sim_list = ss_dict[:, 2] / np.sqrt(ss_dict[:, 3] * ss_dict[:, 4])
        self._sim_dict = {}
        for i in range(ss_dict.shape[0]):
            b1 = ss_dict[i][0]
            b2 = ss_dict[i][1]
            self._sorted[b1] = False
            self._sorted[b2] = False
            val = sim_list[i]
            if not b1 in self._sim_dict.keys():
                self._sim_dict[b1] = []
                self._sim_dict[b1].append([b2, val])
            if not b2 in self._sim_dict.keys():
                self._sim_dict[b2] = []
                self._sim_dict[b2].append([b1, val])

    def predict(self,test_set,k=20):
        n_tests = test_set.shape[0]  # test_set.shape[0]
        rating = np.zeros(n_tests)
        rating_with_gb = np.zeros(n_tests)
        actual_rating = np.zeros(n_tests)
        dist = 0
        for i in range(n_tests):
            user = int(test_set[i][0])
            business = int(test_set[i][1])
            actual_rating[i] = int(test_set[i][2])
            rating[i] = 3
            rating_with_gb[i] = self._mu

            if not business in self._sorted.keys():
                if not user in self._dict_user_avg.keys():
                    pass
                else:
                    rating[i] = self.get_avg_rating(self._dict_user_avg,user)
                    rating_with_gb[i] = self.get_baseline_estimate(user,business)
            else:
                if not self._sorted[business]:
                    val = np.array(self._sim_dict[business])
                    val = val[np.argsort(val[:, 1])[::-1]]
                    self._sorted[business] = True
                    self._sim_dict[business] = val
                else:
                    val = self._sim_dict[business]
                # print(val.shape[0])
                sum_sr = 0
                sum_s = 0
                sum_sr_gb = 0
                co = 0
                for j in range(self._sim_dict[business].shape[0]):
                    b2 = val[j, 0]
                    if val[j, 1] <= 0:
                        break
                    if user in self._BURR[b2].keys():
                        r = self._BURR[b2][user]
                        co += 1
                        sum_sr += val[j, 1] * r
                        sum_sr_gb += val[j,1] * (r - self.get_baseline_estimate(user,val[j,0]))
                        sum_s += val[j, 1]
                        if co == k:
                            break
                if sum_s == 0:
                    rating[i] = self.get_avg_rating(self._dict_business_avg, business)
                    rating_with_gb[i] = self.get_baseline_estimate(user,business)
                else:
                    rating[i] = sum_sr / sum_s
                    rating_with_gb[i] = self.get_baseline_estimate(user,business) + sum_sr_gb/sum_s
            dist += (actual_rating[i] - rating[i]) ** 2
        # np.save('rating', rating)
        # np.save('actual_rating', actual_rating)
        return rating, rating_with_gb

    def get_mu(self):
        return self._mu

    def calculating_RMSE(self,predicted,actual):
        return np.sqrt(sum((predicted-actual)**2)/predicted.shape[0])

    def get_business_avg(self,id):
        if not id in self._dict_business_avg.keys():
            return self._mu
        return self.get_avg_rating(self._dict_business_avg,id)

    def get_user_avg(self,id):
        if not id in self._dict_user_avg.keys():
            return self._mu
        return self.get_avg_rating(self._dict_user_avg,id)

    def get_eval_metrics(self,predicted,actual):
        predicted[(predicted < 1.5)] = 1
        predicted[(predicted >= 1.5) & (predicted < 2.5)] = 2
        predicted[(predicted >= 2.5) & (predicted < 3.5)] = 3
        predicted[(predicted >= 3.5) & (predicted < 4.5)] = 4
        predicted[(predicted >= 4.5)] = 5
        confusion = confusion_matrix(actual, predicted)
        accuracy = sum([confusion[i, i] for i in range(5)])/np.sum(confusion)
        precision = 0
        co=0
        for i in range(5):
            if sum(confusion[:,i])!=0:
                precision+=confusion[i, i]/sum(confusion[:,i])
                co+=1
        precision/=co
        recall = sum([confusion[i, i] / sum(confusion[i, :]) for i in range(5)]) / 5
        return accuracy,precision, recall

def main(args):
    cf = Collab_Filtering()
    training_set = cf.load_dataset_csv('train')
    test_set = cf.load_dataset_csv('validate')
    #training_set = np.concatenate((training_set,test_set))
    cf.train(training_set)
    rating, rating_with_gb = cf.predict(test_set)
    actual_rating = test_set[:,2]
    methods = {"cf": rating,
               "cf global": rating_with_gb,
               "always return 3": (np.ones(actual_rating.shape[0]) * 3),
               "global avg": (np.ones(actual_rating.shape[0]) *cf.get_mu()),
               "user avg": (np.array([cf.get_user_avg(id) for id in test_set[:,0]])),
               "business avg": (np.array([cf.get_business_avg(id) for id in test_set[:,1]])),
               }
    for name, data in methods.items():
        print(name, ":")
        print("   RMSE =", np.sqrt(sum((data - actual_rating) ** 2) / actual_rating.shape[0]))
        evals = cf.get_eval_metrics(data, actual_rating)
        print("   accuracy =", evals[0])
        print("   precision =", evals[1])
        print("   recall =", evals[2])
        #print(np.sqrt(sum((data - actual_rating) ** 2) / actual_rating.shape[0]),"\t",evals[0],"\t",evals[1],"\t",evals[2])

if __name__ == "__main__":
    import sys
    main(sys.argv)