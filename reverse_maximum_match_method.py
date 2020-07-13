class ReverseMaximumMatchMethod(object):
    """
    逆向最大匹配法

    Attribute:
        window_size: 机器词典最长词条字符数
    """

    def __init__(self):
        self.window_size = 3

    def cut(self, text):
        """
        逆向最大匹配法

        RMM的算法思想:
            1. 先将文档进行倒排处理(reverse)，生成逆序文档，然后根据逆序词典，对逆序文档用正向最大匹配法处理
            2. 从左向右取待切分汉语句的m个字符作为匹配字符, m为机器词典中最长词条的字符数
            3. 查找机器词典并进行匹配，若匹配成功, 则将这个匹配字段作为一个词切分出来。
               若匹配不成功, 则将这个匹配字段的最后一个字去掉, 剩下的字符串作为新的匹配字段,
               进行再次匹配, 重复以上过程, 直到切分出所有词为止。

        该应用的算法思想:
            没有使用reverse处理，而是直接从后向前匹配，只是匹配的结果进行了reverse处理
            (因为匹配的结果第一个是"起源"，最后一个是"研究")
        :param text: 输入的字符串
        """
        result = []
        index = len(text)
        dic = ['研究', '研究生', '生命', '命', '的', '起源']
        piece = None

        while index > 0:
            # 只要索引没有到第一个字符，匹配就一直进行
            for size in range(index - self.window_size, index):
                # 匹配最后的3个字符串，如果匹配就进行下一轮while循环，否则字符数-1，进行下一轮for循环
                piece = text[size: index]
                if piece in dic:
                    index = size + 1
                    break

            index -= 1
            result.append(piece + '----')

        result.reverse()
        print(result)


if __name__ == '__main__':
    text = '研究生命的起源'
    tokenizer = ReverseMaximumMatchMethod()
    print(tokenizer.cut(text))
