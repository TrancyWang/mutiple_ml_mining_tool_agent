# -*-encoding:utf-8-*-
################################################################################

################################################################################
"""
Brief: 根据关键字进行一些正则表达式的过滤和抽取
Date: 2021-12-08 17:22:25
Author: TrancyWang
"""
import re
class ContentFilter(object):
    """
      内容过滤
    """

    def __init__(self):
        pass

    def html_fitler(self, text):
        """
            过滤只有html或者<img>的贴子
        """
        re_tag = re.compile('</?\w+[^>]*>')
        new_text = re.sub(re_tag, '', text)
        return new_text

    def special_char_filter(self, text):
        """
          过滤文本里面特殊字符
        """
        new_text = re.sub(",+", ",", text)
        new_text = re.sub("[http]{4}\\:\\/\\/[a-z]*(\\.[a-zA-Z]*)*(\\/([a-zA-Z]|[0-9])*)*\\s?", "", new_text)
        new_text = re.sub("[https]{4}\\:\\/\\/[a-z]*(\\.[a-zA-Z]*)*(\\/([a-zA-Z]|[0-9])*)*\\s?", "", new_text)
        # new_text = re.sub('\\"', "", new_text)
        new_text = re.sub('\n', "", new_text)
        new_text = re.sub('\t', "", new_text)
        new_text = re.sub("\\[", "", new_text)
        new_text = re.sub("\\]", "", new_text)
        new_text = re.sub(" +", " ", new_text)  # 合并空格
        new_text = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", new_text)
        return new_text

    def single_char_filter(self, text):
        """
        过滤掉只有一个字的内容
        :param text: 输入文本
        :return: 如果只有一个字返回True，否则返回False
        """
        # 去除空格后判断长度
        content = text.strip()
        if len(content) <= 1:
            return True
        return False

    def remove_digits(self, text):
        """
        过滤出纯数字的
        :param text:
        :return:
        """
        text = re.sub(u"[1-9]\d*|-[1-9]\d*|[1-9]\d*\.\d*|0\.\d*[1-9]\d*$", "", text)
        if len(text) == 0 or len(text)<=2:
            return True
        return False

    def negativeContentRetainsByRule(self, content):
        """
          使用关键字规则过滤
        """
        res = False
        if len(content) == 0:
            return res
        regex_str = (".*?([\u4E00-\u9FA5]+谨慎|风险|慎重|小心|慎重点|理智|不严谨|盲目|隐私|别神化|不智能|太复杂了|替代|慌得很|太可怕|可怕|"
                     +"拿捏不定|不准|不实用)")
        match_obj = re.match(regex_str, content)
        if match_obj:
            res = True
        return res

    def positiveContentRetainsByRule(self, content):
        """
          使用关键字规则过滤
        """
        res = False
        if len(content) == 0:
            return res
        regex_str = (".*?([\u4E00-\u9FA5]+不错|不错啊|专业|很专业|靠谱|很靠谱|好用|超好用|真不错|炸裂|太人性化|很丝滑|真好|真不错|太厉害"
                     +"|越来越牛|太赞|神器|期待|觉得可以|蛮方便|好厉害|厉害|太快了|很有效|值得信赖|挺准|用起来|支棱起来|有点东西|智能|看好"
                      +"|很有效|有效|很棒|太棒了|太强了|很强|值得试试|速度快|值得信赖|太强了|很强|很棒|简单易懂|越来越好|棒棒哒"
                       +"|不想换|超级精准|更简单|理财搭子|很实用|想试用|太牛|超级精准|牛|超前|很方便|方便|太方便|不错哦|变简单|很方便|很实用|福音|"
                        +"必须能用|帮大忙了|干正事的chatgpt|看得懂|必须支持一下啊)")
        match_obj = re.match(regex_str, content)
        if match_obj:
            res = True
        return res

    def general_filter_content(self, text, lt, gt):
        """
        :param text:  输入的文本数据
        :param lt:    最低阈值
        :param gt:    最低阈值
        :param is_filter: 是否通过正则过滤
        :return:
        """
        content = str(text)
        content = self.html_fitler(content)
        content = self.special_char_filter(content)

        content = content.strip()
        if len(content) <= lt or len(content) > gt:
            content = "其他"
            return content
        isPureDigital = self.remove_digits(content)
        if isPureDigital:
            content = "其他"
            return content
        return str(content)

    # 测试

#text="{""imageUrl"":""https://download...fengyouhui...net/fyh_img/dc0350e0--5666--4fd9--a330--f335368368d91625133814624...jpg"

# filter = ContentFilter()
# text = "理财还是要很慎重才行"
# res =filter.negativeContentRetainsByRule(text)
# print(res)
#
# text2 = "真的很强"
# res =filter.positiveContentRetainsByRule(text2)
# print(res)
#
#
# res = filter.filterContentByKeyWord(text)
# print(res)
# res = filter.filter_chinese(text)
# print(res)
#
# text = "1234566中国-1234444"
# filter.filter_pure_digital(text)