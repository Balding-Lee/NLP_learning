import os
import pickle


class HiddenMarkovModel(object):
    """
    隐含马尔可夫模型
    从可观察的参数中确定该过程的隐含参数

    目标: 判断每个字在构造一个特定词语时占据的确定构词位置

    本例子中的可观察参数: 每个句子中的字
    本例子中的隐含参数: 每个字的构词位置

    Attribute:
        model_file: 存储算法中间结果的文件
        state_list: 状态值集合，包括:
            B: 词首; M: 词中; E: 词尾; S: 单独成词
        load_para: 用于判断是否需要重新加载model_file的字段
    """
    def __init__(self):
        # 提取文件hmm_model.pkl
        # 主要用于存取算法中间结果，不用每次都训练模型
        self.model_file = './data/hmm_model.pkl'

        # 状态值集合
        self.state_list = ['B', 'M', 'E', 'S']

        # 参数加载，用于判断是否需要重新加载model_file
        self.load_para = False

    def try_load_model(self, trained):
        """
        判别加载中间文件结果。
        当直接加载中间结果时，可以不通过语料库训练，直接进行分词调用。
        否则该函数用于初始化初始概率、转移概率以及发射概率等信息。(当需要重新训练时，需要初始化清空结果)
        :param trained: 是否需要直接加载中间结果:
                        True: 加载; False: 初始化清空结果
        :return:
        """
        if trained:
            with open(self.model_file, 'rb') as f:
                self.A_dic = pickle.load(f)
                self.B_dic = pickle.load(f)
                self.Pi_dic = pickle.load(f)
                self.load_para = True
        else:
            # 状态转移概率 P(ok | ok-1)
            self.A_dic = {}
            # 发射概率 P(λk | ok)
            self.B_dic = {}
            # 状态的初始概率 P(λ1 = ok)
            self.Pi_dic = {}
            self.load_para = False

    def train(self, path):
        """
        通过给定的分词语料进行训练，计算转移概率、发射概率和初始概率
        语料格式为每行一句话，逗号隔开也算依据，每个词以空格分隔
        :param path: 训练文件所在路径
        :return self: 返回该类的实例
        """

        # 重置概率矩阵
        self.try_load_model(False)

        # 统计每个标签的出现次数，求P(o)
        count_dic = {}

        def init_parameters():
            """
            初始化参数

            Pi_di: 初始概率，为列向量，直接赋值为0
            A_dic: 转移概率，因为转移情况只有16种，只是概率不同，所以可以先把标签给上并赋值0.0
            B_dic: 发射概率，因为并没有训练，所以里面为空
            count_dic: 统计每个标签出现的次数

            转移概率表明了: 从某个隐含状态转移到另一个(包括自己)隐含状态的概率
            输出概率表明了: 从某个隐含状态输出可见状态的概率
            :return:
            """
            for state in self.state_list:
                self.A_dic[state] = {s: 0.0 for s in self.state_list}
                self.Pi_dic[state] = 0.0
                self.B_dic[state] = {}

                count_dic[state] = 0

        def make_label(text):
            """
            根据传入文本的字数来给这个小词语的每个字赋予一个标签
            :param text: 传入的词语
            :return out_text: 该词语中每个字出现位置的标签列表
            """
            out_text = []
            if len(text) == 1:
                # 如果只有一个字，则归类为S
                out_text.append('S')
            else:
                # 否则返回B，M，E
                # 因为长度大于1，所以至少2个字，如果是多个字，那么在这个词中第一个字肯定是B，最后一个肯定是E，其余的为M
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']

            return out_text

        init_parameters()
        line_num = -1

        # 观察者集合，主要是字以及标点等
        words = set()  # 无序不重复的元素序列
        with open(path, encoding='utf8') as f:
            for line in f:
                line_num += 1  # 计算一共有多少行，用于后面计算初始概率

                line = line.strip()  # 除去首位空格
                if not line:
                    # 如果本行为空，则跳过本轮循环
                    continue

                word_list = [i for i in line if i != ' ']  # 除去空格后的每一行句子的列表

                # 更新字的集合 words = words | set(word_list)，
                # 即更新words，使words和set(word_list)中的字全部去重后加入到words中
                words |= set(word_list)

                line_list = line.split()  # 以空格为分隔符，将文本切片为一个列表
                line_state = []

                for w in line_list:
                    # 循环列表中的每一个字或词，获得make_label(w)中的结果，追加到line_state中
                    line_state.extend(make_label(w))

                assert len(word_list) == len(line_state)  # 断言，如果词语长度不等于状态长度，则报异常

                for k, v in enumerate(line_state):  # 将列表组合为一个索引序列，包括数据下标和数据本身，比如(0, 'B')
                    count_dic[v] += 1  # 该标签出现次数 + 1
                    if k == 0:
                        self.Pi_dic[v] += 1  # 每个句子的第一个字的状态，用于计算初始状态概率
                    else:
                        self.A_dic[line_state[k - 1]][v] += 1  # 用于计算转移概率，即从上一个标签转移到这个标签发生了多少次
                        # 用于计算发射概率，在该状态下每出现一个这个字，这个字的次数 + 1，如果没有，则加入该字典中
                        self.B_dic[line_state[k]][word_list[k]] = self.B_dic[line_state[k]].get(word_list[k], 0) + 1.0

        # 第一个字每个标签出现次数 / 句子个数 = 初始概率
        self.Pi_dic = {k: v * 1.0 / line_num for k, v in self.Pi_dic.items()}


        # 统计所有转移的次数，转移次数 / 这个标签出现次数 = 转移概率
        self.A_dic = {k: {k1: v1 / count_dic[k] for k1, v1 in v.items()} for k, v in self.A_dic.items()}

        # 数据稀疏问题: 由于训练样本不足而导致所估计的分布不可靠的问题
        # 有可能出现某个数据(比如名字)就导致整个词语出现的概率为0
        # 问题提出:
        # 研究表明，语言中只有很少的常用词，大部分词都是低频词。
        # 将语料库的规模扩大，主要是高频词词例的增加，大多数词(n元组)在语料中的出现是稀疏的，
        # 因此扩大语料规模不能从根本上解决稀疏问题。
        # 解决方案:
        # 平滑: 1. 把在训练样本中出现过的事件的概率适当减小；
        #       2. 把减小得到的概率质量分配给训练语料中没有出现过的事件；
        #       3. 这个过程有时候也称为减值法(discounting)。
        # 但是最简单的策略是"加1平滑"，
        # 加1平滑: 规定n元组比真实出现次数多一次，没有出现过的n元组的概率不再是0，而是一个较小的概率值，实现了概率质量的重新分配
        # 统计在每个标签中出现每个字的次数，该次数 / 这个标签出现的次数 = 发射概率
        self.B_dic = {k: {k1: (v1 + 1) / count_dic[k] for k1, v1 in v.items()} for k, v in self.B_dic.items()}  # 序列化

        # print(self.B_dic.items())
        # print("'我'字出现的情况:")
        # for i in self.state_list:
        #     print("state:", i, "  frequency:", self.B_dic[i]['我'])
        # print("'秃'字出现的情况:")
        # for i in self.state_list:
        #     print("state:", i, "  frequency:", self.B_dic[i]['秃'])  # 报错，训练集没有该字
        # print("'然'字出现的情况:")
        # for i in self.state_list:
        #     print("state:", i, "  frequency:", self.B_dic[i]['然'])

        # 保存数据到pkl文件中
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.A_dic, f)
            pickle.dump(self.B_dic, f)
            pickle.dump(self.Pi_dic, f)

        return self

    def viterbi(self, text, states, start_p, trans_p, emit_p):
        """
        维特比算法: 动态规划方法

        如果最终的最优路径经过某个oi，那么从初始节点到oi-1点的路径必然也是一个最优路径
        因为每一个节点oi只会影响前后两个P(oi-1 | oi)和P(oi | oi+1)
        :param text: 输入的需要切分的文本内容
        :param states: 状态值集合
        :param start_p: 初始概率 Pi_dic
        :param trans_p: 转移概率 A_dic
        :param emit_p: 发射概率 B_dic
        :return prob: 最佳路径的概率
        :return path[state]: 最佳路径
        """
        V = [{}]  # 记录输入的文本中每个字属于每个标签的概率
        path = {}  # 标签
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)  # 计算这个文本的第一个字属于4个标签的概率
            path[y] = [y]

        for t in range(1, len(text)):
            # 循环这个文本的第二个字到最后(因为第一个字是属于初始概率)

            V.append({})  # 列表中新增一个字典(用于存放第二个字以及之后的字出现标签的概率)
            new_path = {}

            # 检查训练的发射概率矩阵中是否有该字
            never_seen = text[t] not in emit_p['S'].keys() and \
                text[t] not in emit_p['M'].keys() and \
                text[t] not in emit_p['E'].keys() and \
                text[t] not in emit_p['B'].keys()

            for y in states:
                # 循环4个标签

                # 设置未知字单独成词，未知字的发射概率设置为1
                # 这句话翻译为: 如果这个字在训练结果中没有没见过，那么发射概率为这个字的发射概率，否则为1.0
                p_emit = emit_p[y].get(text[t], 0) if not never_seen else 1.0

                # 如果t - 1的字的y0标签出现过，那么t这个字取: t - 1字y0标签出现的概率 * 从y0转移到y的转移概率 * 发射概率 中最大值
                # 即state是t这个字从y0转移到y最有可能出现的t - 1时刻的标签，prob是t这个字从y0转移到y取到state标签的概率
                # y0是t - 1的字出现过的标签
                # 如果最终的最优路径经过某个oi，那么从初始节点到oi-1点的路径必然也是一个最优路径
                (prob, state) = max(
                    [(V[t - 1][y0] * trans_p[y0].get(y, 0) * p_emit, y0)
                     for y0 in states if V[t - 1][y0] > 0
                     ])

                V[t][y] = prob  # 从y0转移到y，第t个字取到第y个标签的最有可能的概率，每次添加的概率又成为下一轮循环的前一个节点概率
                new_path[y] = path[state] + [y]  # 从t - 1字到t字的路径，如:如果t字是B标签，那么t - 1字最优结果是S标签

            path = new_path  # 更新路径

        # 个人理解: 最后一个字如果出现M的概率比S大(因为按理来说处于中间位置的标签不该最后出现)，
        #          那么极有可能因为是二元模型，历史信息较少判断出错，所以需要重新判断这个字标签为E和M的概率，
        #          而如果最后一个字单独成词，那么要再看看有没有其他的可能性
        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            # 如果最后一个字的标签为M的发射概率大于S,那么就取最后一个字的E或者M中最大的概率以及标签
            (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E', 'M')])
        else:
            # 如果最后一个字的标签为S的发射概率大于M，那么就取最后一个字所有标签中的最大概率和标签
            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])

        return prob, path[state]

    def cut(self, text):
        """
        切词，通过加载中间文件，调用维特比算法完成。
        :param text: 输入的文本
        :return:
        """
        if not self.load_para:
            # 如果load_para为False，那么判断文件是否存在以决定是否需要重新训练
            self.try_load_model(os.path.exists(self.model_file))

        # 获取维特比算法返回的最佳路径概率与最佳路径列表
        prob, pos_list = self.viterbi(text, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)
        begin, next = 0, 0  # begin是一个词的开始，next是下一个词开始的索引

        for i, char in enumerate(text):
            # 将输入的文本组合为一个索引序列，i为索引，char为每个字
            pos = pos_list[i]  # 路径中的第i个节点
            if pos == 'B':
                # 如果这一节点为"B"，那么begin为该索引，意思就是这一节点是这个词的开始
                begin = i
            elif pos == 'E':
                # 如果这一节点为"E", 那么生成器生成从begin到i的内容，并且next为下一个字的开始，
                # 意思就是，这个词结束了，并且next指针指向下个词开始的位置
                yield text[begin: i + 1]
                next = i + 1
            elif pos == 'S':
                # 如果这一节点为"S"，那么生成器生成这个字，并且next为下一个字的开始，
                # 意思就是，这个字单独成词，所以下个词开始
                yield char
                next = i + 1

        if next < len(text):
            # 如果next指针的位置比整个文本长度小，那么生成器生成后面的内容，意思就是后面的内容整体为一个词
            yield text[next:]


if __name__ == "__main__":
    hmm = HiddenMarkovModel()
    hmm.train('./data/trainCorpus.txt_utf8')

    text = '我秃然聪明绝顶'
    res = hmm.cut(text)
    print(text)
    print(str(list(res)))

