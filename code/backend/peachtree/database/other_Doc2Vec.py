import pandas as pd
from pandas import Series, DataFrame
from konlpy.tag import Okt
from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer

okt = Okt()

data = pd.read_csv('./dataDB.csv', encoding = 'cp949')
# other을 column을 하나의 변수로 선언
other = data.other
# string으로 채우기
other.fillna('',inplace=True)

# 단어의 빈도수 구하기
# 단어들을 하나로 묶어주기
other_list = ''
for i in other:
    other_list += i +' '
# 전체 문서 명사기준 tokeninzing
other_list_tokens = okt.nouns(other_list)
# 명사-단어별로 구분 해놓은 것
word_list = pd.Series(other_list_tokens)
word_count = word_list.value_counts()
word_count.head(10)

# '지급 졸업 성적 소득 장학생' 이 중요한 명사들이라고 판단. 이들에 대한 문장 별 유사도를 좌표로 사용한다.
sent_important = '지급 졸업 성적 소득 장학생'

# vectorizer 생성
tfidf_vectorizer = TfidfVectorizer()

# sent_important 를 추가할 other_text_add 생성
other_text_add = other_text
other_text_add.append(sent_important)

# 명사(noun) 단위로 tokenizing 실행
other_tokens_add=[]
for i in (other_text_add):
  tokens = okt.nouns(i)
  other_tokens_add.append(tokens)

# 토큰화 된 단어들을 문장으로 묶어주기
other_for_vectorize_add = []
for content in other_tokens_add:
    sentence=''
    for word in content:
        sentence = sentence + ' ' + word
    other_for_vectorize_add.append(sentence)

feature_vect_simple = tfidf_vectorizer.fit_transform(other_for_vectorize_add)
feature_vect_dense = feature_vect_simple.todense()

# 코사인 유사도 함수 선언
def cos_similarity(v1,v2):
    dot_product = np.dot(v1,v2)
    l2_norm = (np.sqrt(sum(np.square(v1)))*np.sqrt(sum(np.square(v2))))
    similarity = dot_product / l2_norm
    return similarity

# 코사인 유사도 계산하기
onehot_similarity = []
vect_importance = np.array(feature_vect_dense[940]).reshape(-1,)
for i in feature_vect_dense:
    vect1 = np.array(i).reshape(-1,)
    similarity = cos_similarity(vect1,vect_importance)
    onehot_similarity.append(similarity)

# 코사인 유사도가 빈 값에 0으로 채워주기
onehot_similarity=pd.DataFrame(onehot_similarity)
onehot_similarity.fillna(0, inplace=True)

# 마지막 열은 sentence important는 지워준다
onehot_similarity.drop(940)

# 기존 데이터 프레임에 열(onehot_similarity)추가
data['other_similarity'] = onehot_similarity

# 새로 저장하기
data.to_csv('dataDB_other.csv',mode='w')

