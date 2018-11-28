from __future__ import division
import random
import pandas as pd


def split_text_train(text_in_label):
    data = []
    data_dict ={'0': [], '1': []}
    vocabulary = 0
    Ps = 0
    PnS = 0
    sumdata = 0

    for row in text_in_label:
        sumdata += 1

        if row[1] == 1:
            split = row[0].split()
            Ps += 1

            for i in split:
                data.append(i)
                if i not in data_dict['1']:
                    vocabulary += 1
                data_dict['1'].append(i)

        if row[1] == 0:
            split = row[0].split()
            PnS += 1

            for i in split:
                data.append(i)
                if i not in data_dict['0']:
                    vocabulary += 1
                data_dict['0'].append(i)


    return data, data_dict, vocabulary, Ps, PnS, sumdata


def count_class_freq(dictionary):
    data_dict_spam = {}
    data_dict_not_spam = {}

    for word in dictionary['1']:
        if word not in data_dict_spam:
            data_dict_spam[word] = 0

    for word in dictionary['0']:
        if word not in data_dict_not_spam:
            data_dict_not_spam[word] = 0

    return data_dict_spam, data_dict_not_spam


def training(dataset, test_data, data_dictionary, dict_spam_words, dict_not_spam_words, vocabulary, alpha = 1):
    prob_spam_words = {}
    prob_not_spam_words = {}
    spam_words = {}
    not_spam_words = {}
    num_of_word_in_spam = len(data_dictionary['1'])
    num_of_word_in_not_spam = len(data_dictionary['0'])

    for word in dataset:
        if word not in dict_spam_words:
            spam_words[word] = 0

        if word in dict_spam_words:
            if word in spam_words:
                spam_words[word] += 1
            if word not in spam_words:
                spam_words[word] = 0

        if word not in dict_not_spam_words:
            not_spam_words[word] = 0

        if word in dict_not_spam_words:
            if word in not_spam_words:
                not_spam_words[word] += 1
            if word not in not_spam_words:
                not_spam_words[word] = 0

    for word in test_data:

        if word not in dict_spam_words:
            spam_words[word] = 0

        if word not in dict_not_spam_words:
            not_spam_words[word] = 0

    for word in spam_words:
        prob_spam_words[word] = (spam_words[word] + alpha)/(num_of_word_in_spam + vocabulary)

    for word in not_spam_words:
        prob_not_spam_words[word] = (not_spam_words[word] + alpha)/(num_of_word_in_not_spam + vocabulary)

    return prob_spam_words, prob_not_spam_words


def predict(test_data, dict_prob_spam, dict_prob_not_spam, Pspam, PnotSpam, sumdata):
    probability_spam = []
    probability_not_spam = []
    prediction = []
    spam_meter = 1
    not_spam_meter = 1

    for word in test_data:
        if word in dict_prob_spam:
            probability_spam.append(dict_prob_spam[word])
        if word not in dict_prob_spam:
            probability_spam.append(dict_prob_spam[word])
        if word in dict_prob_not_spam:
            probability_not_spam.append(dict_prob_not_spam[word])
        if word not in dict_prob_not_spam:
            probability_not_spam.append(dict_prob_not_spam[word])

    for value in probability_spam:
        spam_meter *= value
    spam_meter = spam_meter * (PSpam/sumdata)

    # print('Spam meter:  %s' %(spam_meter))

    for value in probability_not_spam:
        not_spam_meter *= value
    not_spam_meter = not_spam_meter * (PnotSpam/sumdata)

    # print('Not Spam meter: %s' %(not_spam_meter))

    if spam_meter > not_spam_meter:
        prediction.append(1)
    if not_spam_meter > spam_meter:
        prediction.append(0)
    if spam_meter == not_spam_meter:
        prediction.append(None)

    prediction_val = max(spam_meter, not_spam_meter)

    return prediction, prediction_val


df = pd.read_csv('data_politik.csv')
x = df.iloc[:, 0]
y = df.iloc[:, 1]

data = []
for i in range(len(df)):
    data.append([x[i], y[i]])

dataset, data_dictionary, vocab, PSpam, PnotSpam, sumdata = split_text_train(data)
dict_spam, dict_not_spam = count_class_freq(data_dictionary)


def execute_single_test():
    comment = "dasar cina"
    forbidden_words = ['dan', 'atau', 'yang', 'ber', 'kan', 'mem', 'me', 'men', 'di', 'i']
    # print(type(comment))
    test_data = comment.split()

    for count in range(len(test_data)):
        for i, filter in enumerate(test_data):
            if filter in forbidden_words:
                test_data.pop(i)

    print(test_data)

    prob_spam, prob_not_spam = training(dataset, test_data, data_dictionary, dict_spam, dict_not_spam, vocab)
    pred, val = predict(test_data, prob_spam, prob_not_spam, PSpam, PnotSpam, sumdata)


    print(('\nPREDICTION:  %s \nprobability measure:  %s') %(pred, val))


    print(len(data_dictionary['1']), ':1   =    0:', len(data_dictionary['0']))
    # print(PSpam, PnotSpam, sumdata)
    # print(vocab)
    # print(len(dataset))
    # print(prob_not_spam)


def execute_multiple_test():
    dg = pd.read_csv('data_politik_test.csv')
    x_test = dg.iloc[:, 0]
    y_test = dg.iloc[:, 1]
    label = []
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    testing = []
    for i in range(len(dg)):
        testing.append([x_test[i], y_test[i]])

    for i, row in enumerate(testing):
        split_data = row[0].split()
        label.append(row[1])
        prob_spam, prob_not_spam = training(dataset, split_data, data_dictionary, dict_spam, dict_not_spam, vocab)
        pred, val = predict(split_data, prob_spam, prob_not_spam, PSpam, PnotSpam, sumdata)

        if pred[0] == label[i]:
            if pred[0] == 1:
                TP += 1
            if pred[0] == 0:
                TN += 1

        if pred[0] != label[i]:
            if pred[0] == 1:
                FP += 1
            if pred[0] == 0:
                FN += 1

    Acc = ((TP + TN)/(TP + TN + FP + FN)* 100)
    prec = (TP/(TP+FP)*100)
    rec = (TP/(TP+TN)*100)
    Fmeasure = ((2*TP)/(2*(TP*TN*FP))*100)
    print('Accuracy is %s%s' %(Acc, '%'))
    print('Precision is %s%s' %(prec, '%'))
    print('Recall is %s%s' %(rec, '%'))
    print('F-measure is %s%s' %(Fmeasure, '%'))



execute_multiple_test()