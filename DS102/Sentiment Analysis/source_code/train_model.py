import pandas as pd
import numpy as np
import regex as re
import nltk

nltk.download("perluniprops")
import enchant
import demoji
from underthesea import word_tokenize
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold


import pandas as pd
import numpy as np
import json
import requests
import re
import os
import sys
from datetime import datetime
import dateutil.parser
import traceback
import time
from underthesea import word_tokenize


def clf_report(model, X_test, y_test):
    from sklearn.metrics import classification_report

    y_pred = model.predict(X_test)
    target_names = ["positive", "negative", "neutral"]
    print(classification_report(y_test, y_pred, target_names=target_names))


def mean_scores(model, X_train, y_train):

    scorings = {"accuracy": "accuracy", "f1-macro": "f1_macro"}

    cv = StratifiedKFold(n_splits=5)

    cross_val = cross_validate(
        model, X_train, y_train, scoring=scorings, n_jobs=-1, cv=cv
    )
    results = np.mean([cross_val["test_" + key] for key in scorings.keys()], axis=1)
    result_dict = {key: results[i] for i, key in enumerate(scorings.keys())}
    return result_dict




def remove_dub_char(sentence):
  
  teencode_df = pd.read_csv('/content/teencode.txt', names=['teencode', 'map'], sep='\t')
  teencode_list = teencode_df['teencode'].to_list()
  eng = enchant.Dict("en_US")

  sentence = str(sentence)
  words = []
  for word in sentence.strip().split():
    if word in teencode_list:
      words.append(word)
      continue
    if eng.check(str(word)):
      words.append(word)
      continue
    words.append(re.sub(r"([A-Z])\1+", lambda m: m.group(1), word, flags=re.IGNORECASE))
  return " ".join(words)


def remove_icon(document):
    return demoji.replace(str(document), "")


def remove_punctuation(document):
    return re.sub("[%s]" % re.escape(string.punctuation), "", str(document))



# Chuẩn hoá tiếng việt
uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"


def loaddicchar():
    dic = {}
    char1252 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        "|"
    )
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        "|"
    )
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic


dicchar = loaddicchar()

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def covert_unicode(txt):
    return re.sub(
        r"à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ",
        lambda x: dicchar[x.group()],
        txt,
    )


# Chuẩn hoá kiểu gõ dấu

# logger = LogEventSourcing()


def call_api(data, url, method, timeout=3):
    headers = {
        "content-type": "application/x-www-form-urlencoded",
        "cache-control": "no-cache",
        "postman-token": "6a410524-a8e2-79c7-bd9d-53e4b68c84c7",
    }
    response = requests.request(
        method, url, data=data, headers=headers, timeout=timeout
    )
    return response


uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"


def loaddicchar():
    dic = {}
    char1252 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        "|"
    )
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        "|"
    )
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic


dicchar = loaddicchar()


def convertwindown1525toutf8(txt):
    return re.sub(
        r"à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ",
        lambda x: dicchar[x.group()],
        txt,
    )


"""
    Start section: Chuyển câu văn về kiểu gõ telex khi không bật Unikey
    Ví dụ: thủy = thuyr, tượng = tuwowngj
"""
bang_nguyen_am = [
    ["a", "à", "á", "ả", "ã", "ạ", "a"],
    ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "aw"],
    ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "aa"],
    ["e", "è", "é", "ẻ", "ẽ", "ẹ", "e"],
    ["ê", "ề", "ế", "ể", "ễ", "ệ", "ee"],
    ["i", "ì", "í", "ỉ", "ĩ", "ị", "i"],
    ["o", "ò", "ó", "ỏ", "õ", "ọ", "o"],
    ["ô", "ồ", "ố", "ổ", "ỗ", "ộ", "oo"],
    ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ", "ow"],
    ["u", "ù", "ú", "ủ", "ũ", "ụ", "u"],
    ["ư", "ừ", "ứ", "ử", "ữ", "ự", "uw"],
    ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ", "y"],
]
bang_ky_tu_dau = ["", "f", "s", "r", "x", "j"]

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)


def vn_word_to_telex_type(word):
    dau_cau = 0
    new_word = ""
    for char in word:
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            new_word += char
            continue
        if y != 0:
            dau_cau = y
        new_word += bang_nguyen_am[x][-1]
    new_word += bang_ky_tu_dau[dau_cau]
    return new_word


def vn_sentence_to_telex_type(sentence):
    """
    Chuyển câu tiếng việt có dấu về kiểu gõ telex.
    :param sentence:
    :return:
    """
    words = sentence.split()
    for index, word in enumerate(words):
        words[index] = vn_word_to_telex_type(word)
    return " ".join(words)


"""
    End section: Chuyển câu văn về kiểu gõ telex khi không bật Unikey
"""

"""
    Start section: Chuyển câu văn về cách gõ dấu kiểu cũ: dùng òa úy thay oà uý
    Xem tại đây: https://vi.wikipedia.org/wiki/Quy_t%E1%BA%AFc_%C4%91%E1%BA%B7t_d%E1%BA%A5u_thanh_trong_ch%E1%BB%AF_qu%E1%BB%91c_ng%E1%BB%AF
"""


def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == "q":
                chars[index] = "u"
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == "g":
                chars[index] = "i"
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = (
                        bang_nguyen_am[5][dau_cau]
                        if chars[1] == "i"
                        else bang_nguyen_am[9][dau_cau]
                    )
            return "".join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            # for index2 in nguyen_am_index:
            #     if index2 != index:
            #         x, y = nguyen_am_to_ids[chars[index]]
            #         chars[index2] = bang_nguyen_am[x][0]
            return "".join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
        else:
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
        # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
        # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
    return "".join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def chuan_hoa_dau_cau_tieng_viet(sentence):
    """
    Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
    :param sentence:
    :return:
    """
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        words[index] = chuan_hoa_dau_tu_tieng_viet(word)
    return " ".join(words)


"""
    End section: Chuyển câu văn về cách gõ dấu kiểu cũ: dùng òa úy thay oà uý
    Xem tại đây: https://vi.wikipedia.org/wiki/Quy_t%E1%BA%AFc_%C4%91%E1%BA%B7t_d%E1%BA%A5u_thanh_trong_ch%E1%BB%AF_qu%E1%BB%91c_ng%E1%BB%AF
"""
# Chuẩn hoá teencode
# teencode = pd.read_excel("teencode.xlsx")


def chuan_hoa_teen_code(sentence):
    code = [
        "ctrai",
        "khôg",
        "bme",
        "cta",
        "mih",
        "mqh",
        "cgai",
        "nhữg",
        "mng",
        "svtn",
        "r",
        "qtam",
        "thươg",
        "qtâm",
        "chug",
        "trườg",
        "thoy",
        "đki",
        "atsm",
        "ạk",
        "cv",
        "vch",
        "cùg",
        "pn",
        "pjt",
        "thjk",
        "keke",
        "ktra",
        "nek",
        "cgái",
        "nthe",
        "chúg",
        "kái",
        "tìh",
        "phòg",
        "lòg",
        "từg",
        "rằg",
        "sốg",
        "thuj",
        "thuơng",
        "càg",
        "đky",
        "bằg",
        "sviên",
        "ák",
        "đág",
        "nvay",
        "nhjeu",
        "xg",
        "zồi",
        "trag",
        "zữ",
        "atrai",
        "kte",
        "độg",
        "lmht",
        "gắg",
        "đzai",
        "thgian",
        "plz",
        "đồg",
        "btrai",
        "nthê",
        "hìhì",
        "vọg",
        "hihe",
        "đôg",
        "răg",
        "thườg",
        "tcảm",
        "đứg",
        "ksao",
        "dz",
        "hjxhjx",
        "cmày",
        "xuốg",
        "nkư",
        "lquan",
        "tiếg",
        "hajz",
        "xih",
        "hìh",
        "thàh",
        "ngke",
        "dzậy",
        "teencode",
        "tnào",
        "tưởg",
        "ctrinh",
        "phog",
        "hôg",
        "zìa",
        "kũg",
        "ntnao",
        "trọg",
        "nthế",
        "năg",
        "ngđó",
        "lquen",
        "riêg",
        "ngag",
        "hêhê",
        "bnhiu",
        "ngốk",
        "kậu",
        "highland",
        "kqua",
        "htrc",
        "địh",
        "gđình",
        "giốg",
        "csống",
        "xug",
        "zùi",
        "bnhiêu",
        "cbị",
        "kòn",
        "buôg",
        "csong",
        "chàg",
        "chăg",
        "ngàh",
        "llac",
        "nkưng",
        "nắg",
        "tíh",
        "khoảg",
        "thík",
        "ngđo",
        "ngkhác",
        "thẳg",
        "kảm",
        "dàh",
        "júp",
        "lặg",
        "vđê",
        "bbè",
        "bóg",
        "dky",
        "dòg",
        "uốg",
        "tyêu",
        "snvv",
        "đthoại",
        "qhe",
        "cviec",
        "tượg",
        "qà",
        "thjc",
        "nhưq",
        "cđời",
        "bthường",
        "zà",
        "đáh",
        "xloi",
        "zám",
        "qtrọng",
        "bìh",
        "lzi",
        "qhệ",
        "đhbkhn",
        "hajzz",
        "kủa",
        "lz",
        "đhkhtn",
        "đóg",
        "cka",
        "lgi",
        "nvậy",
        "qả",
        "đkiện",
        "nèk",
        "tlai",
        "bsĩ",
        "hkì",
        "đcsvn",
        "vde",
        "chta",
        "òy",
        "ltinh",
        "ngyeu",
        "đthoai",
        "snghĩ",
        "nặg",
        "họk",
        "dừg",
        "hphúc",
        "hiha",
        "wtâm",
        "thíck",
        "chuện",
        "lạh",
        "fây",
        "ntnày",
        "lúk",
        "haj",
        "ngía",
        "mớj",
        "hsơ",
        "ctraj",
        "nyêu",
        "điiiiiii",
        "rồii",
        "c",
        "kih",
        "kb",
        "hixxx",
        "dthương",
        "nhiềuuu",
        "ctrình",
        "mìnk",
        "mjh",
        "ng",
        "vc",
        "uhm",
        "thỳ",
        "nyc",
        "tks",
        "nàg",
        "thôii",
        "đjên",
        "bgái",
        "vớii",
        "xink",
        "hđộng",
        "đhọc",
        "mk",
        "bn",
        "thik",
        "cj",
        "mn",
        "nguoi",
        "nógn",
        "hok",
        "ko",
        "bik",
        "vs",
        "cx",
        "mik",
        "wtf",
        "đc",
        "cmt",
        "ck",
        "chk",
        "ngta",
        "gđ",
        "oh",
        "vk",
        "ctác",
        "sg",
        "ae",
        "ah",
        "ạh",
        "rì",
        "ms",
        "vn",
        "nhaa",
        "cũg",
        "đag",
        "ơiii",
        "hic",
        "ace",
        "àk",
        "uh",
        "cmm",
        "cmnr",
        "ơiiii",
        "hnay",
        "ukm",
        "tq",
        "ctr",
        "đii",
        "nch",
        "trieu",
        "hahah",
        "nta",
        "ngèo",
        "kêh",
        "ak",
        "ad",
        "j",
        "ny",
        "dc",
        "qc",
        "baoh",
        "zui",
        "zẻ",
        "tym",
        "aye",
        "eya",
        "fb",
        "insta",
        "z",
        "thich",
        "vcl",
        "đt",
        "acc",
        "lol",
        "loz",
        "lozz",
        "trc",
        "chs",
        "đhs",
        "qá",
        "ntn",
        "wá",
        "zậy",
        "zô",
        "ytb",
        "vđ",
        "vchg",
        "sml",
        "xl",
        "cmn",
        "face",
        "hjhj",
        "vv",
        "ns",
        "iu",
        "vcđ",
        "in4",
        "qq",
        "sub",
        "kh",
        "zạ",
        "oy",
        "jo",
        "clmm",
        "bsvv",
        "troai",
        "wa",
        "hjx",
        "e",
        "ik",
        "ji",
        "ce",
        "lm",
        "đz",
        "sr",
        "ib",
        "hoy",
        "đbh",
        "k",
        "vd",
        "a",
        "cũng z",
        "z là",
        "unf",
        "my fen",
        "fen",
        "cty",
        "on lai",
        "u hai ba",
        "kô",
        "đtqg",
        "hqua",
        "xog",
        "uh",
        "uk",
        "nhoé",
        "biet",
        "quí",
        "stk",
        "hong kong",
        "đươc",
        "nghành",
        "nvqs",
        "ngừoi",
        "trog",
        "tgian",
        "biêt",
        "fải",
        "nguời",
        "tđn",
        "bth",
        "vcđ",
        "tgdd",
        "khg",
        "nhưg",
        "thpt",
        "thằg",
        "đuợc",
        "dc",
        "đc",
        "ah",
        "àh",
        "ku",
        "thým",
        "onl",
        "zô",
        "zú",
        "cmnd",
        "sđt",
        "klq",
    ]
    chuanhoa = [
        "con trai",
        "không",
        "bố mẹ",
        "chúng ta",
        "mình",
        "mối quan hệ",
        "con gái",
        "những",
        "mọi người",
        "sinh viên tình nguyện",
        "rồi",
        "quan tâm",
        "thương",
        "quan tâm",
        "chung",
        "trường",
        "thôi",
        "đăng ký",
        "ảo tưởng sức mạnh",
        "ạ",
        "công việc",
        "vãi chưởng",
        "cùng",
        "bạn",
        "biết",
        "thích",
        "ce ce",
        "kiểm tra",
        "nè",
        "con gái",
        "như thế",
        "chúng",
        "cái",
        "tình",
        "phòng",
        "lòng",
        "từng",
        "rằng",
        "sống",
        "thôi",
        "thương",
        "càng",
        "đăng ký",
        "bằng",
        "sinh viên",
        "á",
        "đáng",
        "như vậy",
        "nhiều",
        "xuống",
        "rồi",
        "trang",
        "dữ",
        "anh trai",
        "kinh tế",
        "động",
        "liên minh huyền thoại",
        "gắng",
        "đẹp trai",
        "thời gian",
        "pờ ly",
        "đồng",
        "bạn trai",
        "như thế",
        "hì hì",
        "vọng",
        "hi he",
        "đông",
        "răng",
        "thường",
        "tình cảm",
        "đứng",
        "không sao",
        "đẹp trai",
        "hix hix",
        "chúng mày",
        "xuống",
        "như",
        "liên quan",
        "tiếng",
        "hai",
        "xinh",
        "hình",
        "thành",
        "nghe",
        "dậy",
        "tin cốt",
        "thế nào",
        "tưởng",
        "chương trình",
        "phong",
        "không",
        "gì",
        "cũng",
        "như thế nào",
        "trọng",
        "như thế",
        "năng",
        "người đó",
        "làm quen",
        "riêng",
        "ngang",
        "hê hê",
        "bao nhiêu",
        "ngốc",
        "cậu",
        "hai lừn",
        "kết quả",
        "hôm trước",
        "định",
        "gia đinh",
        "giống",
        "cuộc sống",
        "xùng",
        "rồi",
        "bao nhiêu",
        "chuẩn bị",
        "còn",
        "buông",
        "cuộc sống",
        "chàng",
        "chăng",
        "ngành",
        "liên lạc",
        "nhưng",
        "nắng",
        "tính",
        "khoảng",
        "thích",
        "người đó",
        "người khác",
        "thẳng",
        "cảm",
        "dành",
        "giúp",
        "lặng",
        "vấn đề",
        "bạn bè",
        "bóng",
        "đăng ký",
        "dòng",
        "uống",
        "tình yêu",
        "sinh nhật vui vẻ",
        "điện thoại",
        "quan hệ",
        "công việc",
        "tượng",
        "quà",
        "thích",
        "nhưng",
        "cuộc đời",
        "bình thường",
        "già",
        "đánh",
        "xin lỗi",
        "dám",
        "quan trọng",
        "bình",
        "làm gì",
        "quan hệ",
        "đại học bách khoa hà nội",
        "hai",
        "của",
        "làm gì",
        "đại học khoa học tự nhiên",
        "đóng",
        "cha",
        "làm gì",
        "như vậy",
        "quả",
        "điều kiện",
        "nè",
        "tương lai",
        "bác sĩ",
        "học kỳ",
        "đảng cộng sản việt nam",
        "vấn đề",
        "chúng ta",
        "rồi",
        "linh tinh",
        "người yêu",
        "điện thoại",
        "suy nghĩ",
        "nặng",
        "học",
        "dừng",
        "hạnh phúc",
        "hi ha",
        "quan tâm",
        "thích",
        "chuyện",
        "lạnh",
        "phây",
        "như thế này",
        "lúc",
        "hai",
        "nghía",
        "mới",
        "hồ sơ",
        "con trai",
        "người yêu",
        "đi",
        "rồi",
        "chị",
        "kinh",
        "kết bạn",
        "hích",
        "dễ thương",
        "nhiều",
        "chương trình",
        "mình",
        "mình",
        "người",
        "vợ chồng",
        "ừm",
        "thì",
        "người yêu cũ",
        "thanks",
        "nàng",
        "thôi",
        "điên",
        "bạn gái",
        "với",
        "xinh",
        "hành động",
        "đại học",
        "mình",
        "bạn",
        "thích",
        "chị",
        "mọi người",
        "người",
        "nóng",
        "không",
        "không",
        "biết",
        "với",
        "cũng",
        "mình",
        "what the fuck",
        "được",
        "comment",
        "chồng",
        "chồng",
        "người ta",
        "gia đình",
        "ồ",
        "vợ",
        "công tác",
        "sài gòn",
        "anh em",
        "à",
        "ạ",
        "gì",
        "mới",
        "việt nam",
        "nha",
        "cũng",
        "đang",
        "ơi",
        "hích",
        "anh chị em",
        "à",
        "ừ",
        "con mẹ mày",
        "con mẹ nó rồi",
        "ơi",
        "hôm nay",
        "ừm",
        "trung quốc",
        "chương trình",
        "đi",
        "nói chuyện",
        "triệu",
        "ha ha",
        "người ta",
        "nghèo",
        "kênh",
        "à",
        "admin",
        "gì",
        "người yêu",
        "được",
        "quảng cáo",
        "bao giờ",
        "vui",
        "vẻ",
        "tim",
        "anh yêu em",
        "em yêu anh",
        "facebook",
        "instagram",
        "vậy",
        "thích",
        "vờ cờ lờ",
        "điện thoại",
        "account",
        "lồn",
        "lồn",
        "lồn",
        "trước",
        "chẳng hiểu sao",
        "đéo hiểu sao",
        "quá",
        "như thế nào",
        "quá",
        "vậy",
        "vô",
        "youtube",
        "vãi đái",
        "vãi chưởng",
        "sấp mặt lờ",
        "xin lỗi",
        "con mẹ nó",
        "facebook",
        "hi hi",
        "vui vẻ",
        "nói",
        "yêu",
        "vãi cải đái",
        "info",
        "quằn què",
        "subcribe",
        "không",
        "vậy",
        "rồi",
        "giờ",
        "cái lồn mẹ mày",
        "buổi sáng vui vẻ",
        "trai",
        "quá",
        "hix",
        "em",
        "ý",
        "gì",
        "chị em",
        "làm",
        "đẹp giai",
        "sorry",
        "inbox",
        "thôi",
        "đéo bao giờ",
        "không",
        "ví dụ",
        "anh",
        "cũng vậy",
        "vậy là",
        "unfriend",
        "my friend",
        "friend",
        "công ty",
        "online",
        "u23",
        "không",
        "đội tuyển quốc gia",
        "hôm qua",
        "xong",
        "ừ",
        "ừ",
        "nhé",
        "biết",
        "quý",
        "số tài khoản",
        "hồng kông",
        "được",
        "ngành",
        "nghĩa vụ quân sự",
        "người",
        "trong",
        "thời gian",
        "biết",
        "phải",
        "người",
        "thế đéo nào",
        "bình thường",
        "vãi cả đái",
        "thế giới di động",
        "không",
        "nhưng",
        "trung học phổ thông",
        "thằng",
        "được",
        "được",
        "được",
        "à",
        "à",
        "cu",
        "thím",
        "online",
        "dô",
        "vú",
        "chứng minh nhân dân",
        "số điện thoại",
        "không liên quan",
    ]
    data = {"code": code, "chuanhoa": chuanhoa}
    teencode = pd.DataFrame(data)
    result = [x.strip() for x in sentence.split()]
    for i in range(0, len(result)):
        for j in range(0, len(teencode)):
            if result[i] == teencode.at[j, "code"]:
                result[i] = teencode.at[j, "chuanhoa"]
    x = " ".join(result)
    x.strip()
    return x


stopword = list(
    [
        "đến_nỗi",
        "việc",
        "các",
        "bị",
        "cho",
        "đây",
        "vẫn",
        "đang",
        "nếu",
        "để",
        "khi",
        "lên",
        "rằng",
        "đó",
        "vì",
        "sự",
        "lúc",
        "của",
        "sau",
        "lại",
        "và",
        "so",
        "cùng",
        "nên",
        "như",
        "vào",
        "cứ",
        "gì",
        "điều",
        "cần",
        "từng",
        "vậy",
        "với",
        "chuyện",
        "một_cách",
        "chiếc",
        "do",
        "càng",
        "cả",
        "chỉ",
        "bởi",
        "cái",
        "qua",
        "ra",
        "những",
        "nữa",
        "tại",
        "sẽ",
        "có_thể",
        "nơi",
        "là",
        "rồi",
        "từ",
        "phải",
        "theo",
        "mà",
        "chứ",
        "thì",
        "này",
        "cũng",
    ]
)


def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    return " ".join(words)


# Hàm tiền xử lí tổng hợp


def text_preprocess(document):
    # đưa về lower
    document = document.lower()
    # chuẩn hóa unicode
    document = covert_unicode(document)
    # chuẩn hóa cách gõ dấu tiếng Việt
    document = chuan_hoa_dau_cau_tieng_viet(document)
    # xóa các ký tự không cần thiết
    document = re.sub(
        r"[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]",
        " ",
        document,
    )
    # chuẩn hoá teencode
    document = chuan_hoa_teen_code(document)
    # tách từ
    document = word_tokenize(document, format="text")
    # loại bỏ stopword
    document = remove_stopwords(document)
    # xóa khoảng trắng thừa
    document = re.sub(r"\s+", " ", document).strip()
    return document


def clean_text(sentence):
    sentence = remove_dub_char(sentence)
    sentence = remove_icon(sentence)
    sentence = text_preprocess(sentence)
    return sentence
