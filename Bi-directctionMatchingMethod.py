class BiDirectctionMatchingMethod(object):
    """
    双向最大匹配法

    算法思想:
        1. 如果正反向分词结果词数不同，则取分词数量较少的那个
        2. 如果分词结果词数相同：
            2.1 分词结果相同，说明没有歧义，可返回任意一个
            2.2 分词结果不同，返回其中单字较少的那个

    Attribute:
        window_size: 机器词典最长词条字符数
        dic: 机器词典
        text: 需要匹配的字符串(文本)
    """

    def __init__(self, text):
        self.window_size = 3
        self.dic = ['研究', '研究生', '生命', '命', '的', '起源']
        self.text = text

    def MM_cut(self):
        """
        正向最大匹配法的方法

        算法思想:
        1. 从左向右取待切分汉语句的m个字符作为匹配字符, m为机器词典中最长词条的字符数
        2. 查找机器词典并进行匹配，若匹配成功, 则将这个匹配字段作为一个词切分出来。
           若匹配不成功, 则将这个匹配字段的最后一个字去掉, 剩下的字符串作为新的匹配字段,
           进行再次匹配, 重复以上过程, 直到切分出所有词为止。

        :return MM_result: 正向最大匹配法匹配结果
        """

        MM_result = []
        MM_index = 0
        MM_text_length = len(self.text)
        MM_piece = None

        while MM_index < MM_text_length:
            # MM的循环
            for size in range(self.window_size + MM_index, MM_index, -1):
                # 每一轮循环从新的字符串的"索引位置(起始位置) + 机器词典中最长的词条字符数"位置开始匹配字符
                # 如果这一轮循环匹配失败，则将要匹配的字符数进行-1操作，进行新一轮的匹配
                # 最后一轮匹配为一个字符匹配
                MM_piece = self.text[MM_index: size]
                if MM_piece in self.dic:
                    # 如果这串字符在机器词典中，那么移动索引至匹配了的字符串的最后一个字符的下标处(将匹配了的字符串移出这个线性表)
                    MM_index = size - 1
                    break

            # 将索引移动到下一轮匹配的开始字符位置，即如果匹配成功，将之前成功匹配的字符移除线性表
            # 如果匹配失败，则是将第一个字符移除线性表
            MM_index += 1
            MM_result.append(MM_piece)

        return MM_result

    def RMM_cut(self):
        """
        逆向最大匹配法

        RMM的算法思想:
        1.
        先将文档进行倒排处理(reverse)，生成逆序文档，然后根据逆序词典，对逆序文档用正向最大匹配法处理
        2.
        从左向右取待切分汉语句的m个字符作为匹配字符, m为机器词典中最长词条的字符数
        3.
        查找机器词典并进行匹配，若匹配成功, 则将这个匹配字段作为一个词切分出来。
        若匹配不成功, 则将这个匹配字段的最后一个字去掉, 剩下的字符串作为新的匹配字段,
        进行再次匹配, 重复以上过程, 直到切分出所有词为止。

        该应用的算法思想:
        没有使用reverse处理，而是直接从后向前匹配，只是匹配的结果进行了reverse处理
        (因为匹配的结果第一个是"起源"，最后一个是"研究")

        :return RMM_result: 逆向最大匹配法匹配结果
        """
        RMM_result = []
        RMM_index = len(self.text)
        RMM_piece = None

        while RMM_index > 0:
            # RMM的循环
            for size in range(RMM_index - self.window_size, RMM_index):
                # 匹配最后的3个字符串，如果匹配就进行下一轮while循环，否则字符数-1，进行下一轮for循环
                RMM_piece = self.text[size: RMM_index]
                if RMM_piece in self.dic:
                    # 如果这串字符在机器词典中，那么移动索引至成功匹配的第一个字符的下标处(将匹配了的字符串移出这个线性表)
                    RMM_index = size + 1
                    break

            # 将索引移动到下一轮匹配的开始字符位置，即如果匹配成功，将之前成功匹配的字符移除线性表
            # 如果匹配失败，则是将最后一个字符移除线性表
            RMM_index -= 1
            RMM_result.append(RMM_piece)

        RMM_result.reverse()

        return RMM_result


def get_best_matching_result(MM_result, RMM_result):
    """
    比较两个分词方法分词的结果

    比较方法:
        1. 如果正反向分词结果词数不同，则取分词数量较少的那个
        2. 如果分词结果词数相同：
            2.1 分词结果相同，说明没有歧义，可返回任意一个
            2.2 分词结果不同，返回其中单字较少的那个

    :param MM_result: 正向最大匹配法的分词结果
    :param RMM_result: 逆向最大匹配法的分词结果
    :return:
        1.词数不同返回词数较少的那个
        2.词典结果相同，返回任意一个(MM_result)
        3.词数相同但是词典结果不同，返回单字最少的那个
    """
    if len(MM_result) != len(RMM_result):
        # 如果两个结果词数不同，返回词数较少的那个
        return MM_result if (len(MM_result) < len(RMM_result)) else RMM_result
    else:
        if MM_result == RMM_result:
            # 因为RMM的结果是取反了的，所以可以直接匹配
            # 词典结果相同，返回任意一个
            return MM_result
        else:
            # 词数相同但是词典结果不同，返回单字最少的那个
            MM_word_1 = 0
            RMM_word_1 = 0
            for word in MM_result:
                # 判断正向匹配结果中单字出现的词数
                if len(word) == 1:
                    MM_word_1 += 1

            for word in RMM_result:
                # 判断逆向匹配结果中单字出现的词数
                if len(word) == 1:
                    RMM_word_1 += 1

            return MM_result if (MM_word_1 < RMM_word_1) else RMM_result


if __name__ == '__main__':
    text = '研究生命的起源'
    tokenizer = BiDirectctionMatchingMethod(text)
    MM_result = tokenizer.MM_cut()
    RMM_result = tokenizer.RMM_cut()
    best_result = get_best_matching_result(MM_result, RMM_result)
    print("MM_result:", MM_result)
    print("RMM_result:", RMM_result)
    print("best_result:", best_result)
