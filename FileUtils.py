# encoding:utf-8
import codecs
import os
import pickle
import re

import jieba


def get_seg_word(file_pos):
    try:
        f = codecs.open(file_pos, 'r', 'gbk')
        file_content = f.read()
    except:
        print(file_pos)
    finally:
        content = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——!！，,.。？?、~@#￥%……&*（）]+[\r\n]+[A-Za-z0-9]", "", file_content)
        return list(jieba.cut(content))


def save(file_pos, name):
    result_file = open(file_pos + '/../' + name, 'wb')
    files = os.listdir(file_pos)
    obj = [files]
    for f in files:
        obj.append(get_seg_word(os.path.join(file_pos, f)))

    pickle.dump(obj, result_file, 2)


def load(path):
    f = open(path + "/" + 'result.i2', 'rb')

    obj = pickle.load(f)
    for i in range(1, len(obj)):
        f = open(path + "/" + obj[0][i - 1], 'wb')
        f.writelines(obj[i])
        f.close()
