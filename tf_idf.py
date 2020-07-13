import glob
import random
import jieba


def get_content(path):
    """
    数据读取
    :param path: 文件存储路径
    :return content: 去掉了首尾空格的字符
    """
    with open(path, 'r', encoding='gbk', errors='ignore') as f:
        content = ''
        for l in f:
            l = l.strip()
            content += l

        return content


def get_TF(words, top_K=10):
    """
    高频词统计
    :param words: 词的数组
    :param top_K: 读取最高频词的数量
    :return: 出现频率最高的top_K词
    """

    tf_dic = {}
    for w in words:
        tf_dic[w] = tf_dic.get(w, 0) + 1

    return sorted(tf_dic.items(), key=lambda x: x[1], reverse=True)[: top_K]


def stop_words(path):
    """
    整理停用词，比如','，'.'等
    :param path: 停用词词典路径
    :return:
    """
    with open(path, encoding='UTF-8') as f:
        return [l.strip() for l in f]


def main():
    files = glob.glob('./data/news/C000013/*.txt')
    corpus = [get_content(x) for x in files]

    sample_inx = random.randint(0, len(corpus))
    split_words = [x for x in jieba.cut(corpus[sample_inx]) if x not in stop_words('./data/stop_words.utf8')]
    print('样本之一:', corpus[sample_inx])
    print('样本分词效果:', '/ '.join(split_words))
    print('样本的top_K词:', str(get_TF(split_words)))


main()
