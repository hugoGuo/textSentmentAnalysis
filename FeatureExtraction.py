# encoding:utf-8
import itertools
import os
import pickle
from random import shuffle

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC

fileFolder = os.path.join(os.path.dirname(os.getcwd()), 'DataSource')


def get_pos_data():
    pos_data = list(pickle.load(open(fileFolder + "/pos.p", "rb")))
    return pos_data


def get_neg_data():
    neg_data = list(pickle.load(open(fileFolder + "/neg.p", "rb")))
    return neg_data


def get_pos_words():
    temp_data = get_pos_data()
    temp_data.remove(temp_data[0])
    pos_words = list(itertools.chain(*temp_data))
    return pos_words


def get_neg_words():
    neg_data = get_neg_data()
    neg_data.remove(neg_data[0])
    neg_words = list(itertools.chain(*neg_data))

    return neg_words


def create_word_scores():
    # FileUtils.save(os.path.join(fileFolder, 'pos'), 'pos.p')
    # FileUtils.save(os.path.join(fileFolder, 'neg'), 'neg.p')

    word_fd = nltk.FreqDist()  # 可统计所有词的词频
    cond_word_fd = nltk.ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词频
    for word in get_pos_words():
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in get_neg_words():
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N()  # 积极词的数量
    neg_word_count = cond_word_fd['neg'].N()  # 消极词的数量
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count),
                                               total_word_count)  # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count),
                                               total_word_count)  # 同理
        word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量

    return word_scores


def create_word_bigram_scores():
    bigram_finder = BigramCollocationFinder.from_words(get_pos_words())
    bigram_finder = BigramCollocationFinder.from_words(get_neg_words())
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = get_pos_words() + posBigrams  # 词和双词搭配
    neg = get_neg_words() + negBigrams

    word_fd = nltk.FreqDist()  # 可统计所有词的词频
    cond_word_fd = nltk.ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词频
    for word in pos:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in neg:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores


def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[
                :number]  # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_words = set([w for w, s in best_vals])
    return dict([(word, True) for word in best_words])


def build_features():
    feature = find_best_words(create_word_scores(), 300)
    pos_features = []
    pos_data = get_pos_data()
    pos_data.remove(pos_data[0])
    for items in pos_data:
        a = {}
        for item in items:
            if item in feature.keys():
                a[item] = 'True'
        pos_words = [a, 'pos']
        pos_features.append(pos_words)

    neg_features = []
    neg_data = get_neg_data()
    neg_data.remove(neg_data[0])
    for items in neg_data:
        b = {}
        for item in items:
            if item in feature.keys():
                b[item] = 'True'
        neg_words = [b, 'neg']
        neg_features.append(neg_words)

    return pos_features, neg_features


posFeatures, negFeatures = build_features()
shuffle(posFeatures)  # 把文本的排列随机化
shuffle(negFeatures)  # 把文本的排列随机化

train = posFeatures[720:] + negFeatures[720:]  # 训练集(80%)

test = posFeatures[:720] + negFeatures[:720]  # 预测集(验证集)(20%)

data, tag = zip(*test)  # 分离测试集合的数据和标签，便于验证和测试


def score(classifier):
    classifier = nltk.SklearnClassifier(classifier) #在nltk 中使用scikit-learn 的接口
    classifier.train(train) #训练分类器
    pred = classifier.classify_many(data) #对开发测试集的数据进行分类，给出预测的标签
    return accuracy_score(tag, pred)


print('BernoulliNB`s accuracy is %f' % score(BernoulliNB()))
print('MultinomiaNB`s accuracy is %f' % score(MultinomialNB()))

print('LogisticRegression`s accuracy is  %f' % score(LogisticRegression()))

print('SVC`s accuracy is %f' % score(SVC()))

# print('LinearSVC`s accuracy is %f' % score(LinearSVC()))
#
# print('NuSVC`s accuracy is %f' % score(NuSVC()))
