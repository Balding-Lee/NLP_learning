class MaximumMatchMethod(object):
    """
    正向最大匹配法的类

    Attributes:
        window_size: 机器词典中最长词条的字符数
    """
    def __init__(self):
        self.window_size = 3

    def cut(self, text):
        """
        正向最大匹配法的方法

        算法思想:
        1. 从左向右取待切分汉语句的m个字符作为匹配字符, m为机器词典中最长词条的字符数
        2. 查找机器词典并进行匹配，若匹配成功, 则将这个匹配字段作为一个词切分出来。
           若匹配不成功, 则将这个匹配字段的最后一个字去掉, 剩下的字符串作为新的匹配字段,
           进行再次匹配, 重复以上过程, 直到切分出所有词为止。

        :param text: 输入的字符串
        """
        result = []
        index = 0
        text_length = len(text)
        dic = ['研究', '研究生', '生命', '命', '的', '起源']  # 机器词典
        piece = None

        while text_length > index:
            # 当文本长度大于索引值的时候，就一直循环
            for size in range(self.window_size + index, index, - 1):
                # 每一轮循环从新的字符串的"索引位置(起始位置) + 机器词典中最长的词条字符数"位置开始匹配字符
                # 如果这一轮循环匹配失败，则进行-1操作，进行新一轮的匹配
                # 最后一轮匹配为一个字符匹配
                piece = text[index: size]
                if piece in dic:
                    # 如果这串字符在机器词典中，那么移动索引至匹配了的字符串的最后一个字符的下标处(将匹配了的字符串移出这个线性表)
                    index = size - 1
                    break

            index += 1  # 将索引移动到下一轮匹配的开始字符位置
            result.append(piece + '----')
        print(result)


if __name__ == '__main__':
    text = '研究生命的起源'
    tokenizer = MaximumMatchMethod()
    print(tokenizer.cut(text))


