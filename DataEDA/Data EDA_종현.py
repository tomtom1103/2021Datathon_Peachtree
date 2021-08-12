#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import sweetviz as sv
from autoviz.AutoViz_Class import AutoViz_Class
from pandasgui import show
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 데이터 전처리, EDA by Thomas

# In[3]:


data = pd.read_excel('first_final_cha.xlsx')


# In[4]:


data.head(3)


# In[5]:


data.isnull().sum()


# ### Column 별 결측치 채우기

# #### Major

# In[6]:


data['major'] #학과가 상관없는 장학금이 많기때문에 결측치가 많다.


# In[7]:


all_majors = '경영학과 국어국문학과 영어영문학과 철학과 한국사학과 사학과 사회학과 독어독문학과 불어불문학과 중어중문학과 노어노문학과 일어일문학과 서어서문학과 한문학과 언어학과 식품자원경제학과 생명과학부 생명공학부 식품공학과 환경생태공학부 정치외교학과 경제학과 행정학과 통계학과 수학과 물리학과 화학과 지구환경과학과 화공생명공학과 신소재공학부 건축사회환경공학부 건축학과 기계공학부 산업경영공학부 전기전자공학부 융합에너지공학과 반도체공학과 교육학과 체육교육과 가정교육과 수학교육과 국어교육과 영어교육과 지리교육과 역사교육과 간호학과 디자인조형학부 국제학부 국제학부 바이오의공학부 바이오시스템의과학부 보건환경융합과학부 보건정책관리학부 컴퓨터학과 데이터과학과 사이버국방학과 심리학부'
print(all_majors) #넣어주기 위해 고려대 모든 학과를 변수에 지정


# In[8]:


data['major'].fillna(all_majors, inplace=True)


# In[10]:


data.head(2)


# #### Activity

# In[11]:


data[data['activity'].isnull()]


# In[12]:


data.at[726, 'activity'] = 1
data.at[802, 'activity'] = 1
data.at[845, 'activity'] = 1
data[data['activity'].isnull()]


# #### Characteristic

# In[13]:


data[data['characteristic'].isnull()]


# In[14]:


data.at[258, 'characteristic'] = 0
data.at[634, 'characteristic'] = 2
data.at[715, 'characteristic'] = 2
data.at[759, 'characteristic'] = 2
data.at[793, 'characteristic'] = 0
data.at[802, 'characteristic'] = 0
data.at[825, 'characteristic'] = 0
data.at[836, 'characteristic'] = 2
data.at[845, 'characteristic'] = 3
data.at[901, 'characteristic'] = 3
data.at[905, 'characteristic'] = 2
data.at[922, 'characteristic'] = 3
data.at[935, 'characteristic'] = 0
data[data['characteristic'].isnull()]


# #### Sex

# In[15]:


data[data['sex'].isnull()]


# In[16]:


data.at[793, 'sex'] = 2
data.at[836, 'sex'] = 2
data.at[845, 'sex'] = 2
data[data['sex'].isnull()]


# #### Age_max

# In[17]:


len(data[data['age_max'].isnull()]) #age_max의 결측치는 지원나이 상관없다는 뜻.


# In[18]:


data['age_max'].fillna(50, inplace=True)
len(data[data['age_max'].isnull()])


# #### last_grade

# In[19]:


len(data[data['last_grade'].isnull()]) # 이전학기 학점이 상관없으면 0.00 4.50 으로 통일


# In[20]:


data['last_grade'].fillna('0.00 4.50', inplace=True)
len(data[data['last_grade'].isnull()])


# In[ ]:


# 추가적으로 지원금액 시급인거 만원단위로 안바뀐거 하나 수정


# In[21]:


data.at[649, 'amount'] = 1.115


# In[22]:


data.columns


# In[ ]:


pr = data.profile_report()
pr.to_file('./pr_report.html')


# In[ ]:


sweet_report = sv.analyze(data)


# In[ ]:


sweet_report.show_html('sweet_report.html')


# In[ ]:


AV = AutoViz_Class()


# In[ ]:


df = AV.AutoViz('afterfirsteda_1.xlsx')


# In[ ]:


data.to_excel('dataDB.xlsx', encoding = 'UTF-8') #필요하면 저장.


# In[ ]:


# pandas report, sweetviz, autoviz, pandasgui 사용해서 EDA 방식 지정.


# ### EDA

# #### 지원금액에 대한 insight 구하기

# In[38]:


data.head(1)


# In[39]:


data_formoney = data.drop(['year','in_school','activity',                    'characteristic','major','sem_min','sem_max','sex',                   'age_min','age_max','pause','characteristic_money','region',                    'recommendation','link','date_start','date_end',                   'feature_integer','feature','feature_specified','other','original_id'], axis=1)


# In[40]:


data_formoney #지원금액에 대한 insight 도출하기 위해 새로운 df 생성


# In[41]:


data_formoney.info() #datatype 확인


# In[42]:


indexnames = data_formoney[data_formoney['grade_min'] == 0.00].index
#최소요구학점이 0.00이면 학점이 상관없는 장학금이기 때문에 row 삭제


# In[ ]:


data_formoney.drop(indexnames, inplace=True)


# In[45]:


data_formoney


# In[46]:


data_formoney['added_grade'] = np.where(data_formoney['grade_min'] == True,                                        (data_formoney['grade_min'])+(data_formoney['grade_max']),                                        (data_formoney['grade_min'])+(data_formoney['grade_max']))

# 코드 설명하자면 added_grade 라는 칼럼을 생성.
# np.where 함수 args 참고!
# grade_min == True 는 grade_min 이 존재하는지 판단. 당연히 존재하니깐 뒤 args 통과. 약간 편법쓴거다
# 뒤 두개의 args는 똑같다. 최소학점과 최대학점의 합을 구하는 코든데 np.where 함수의
# 2번째, 3번째 args 가 1번째 args 의 조건에 따라 bool인것.


# In[47]:


data_formoney #added_grade 칼럼 생성


# In[48]:


data_formoney['added_grade'] = data_formoney['added_grade'].div(2) #2로나눠서 평균값


# In[49]:


data_formoney


# In[ ]:


data_formoney.to_csv('data_formoney.csv') #저장용


# In[51]:


data_moneyregression = data_formoney.drop(data_formoney.                                          columns.difference(['added_grade','amount']), 1)


# In[52]:


data_moneyregression #지원금액과 학점평균만 남기고 나머지 날려줬다.


# In[ ]:


# 여기서 data_formoney, data_moneyregression을 pandasgui의 Show()함수에 넣으면 interactive GUI 가 실행된다
# gui 통해서 그래프 생성!


# #### 각 학과와 지원금액 insight 구하기

# In[55]:


data_majormoney = data.drop(data.columns.difference(['major','amount']), 1)


# In[56]:


data_majormoney #각 장학금이 어떤 과를 지원해주는지, 그리고 그에 따른 금액을 보여주는 데이터프레임.


# In[57]:


from pandas import DataFrame


# In[58]:


df = data_majormoney


# In[59]:


def splitDataFrameList(df,target_column,separator): #explode 함수 생성. df, 터트리고 싶은 칼럼과 세퍼레이터를 인자로 받는다.
    
    def splitListToRows(row,row_accumulator,target_column,separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


# In[60]:


data_explode = splitDataFrameList(df, 'major', ' ')


# In[61]:


data_explode #데이터 익스플로딩 잘 된다!


# In[ ]:


data_explode.to_csv('data_explode.csv', encoding = 'UTF-8') #필요하면 저장.


# In[62]:


import plotly.express as px


# In[63]:


grouped_data_explode = data_explode.groupby('major') #학과로 그룹바이
mean_data_explode = grouped_data_explode.mean() #금액이 더해지기 때문에 평균
mean_data_explode = mean_data_explode.reset_index() #인덱스 재설정


# In[64]:


mean_data_explode #각 학과 학생에 대한 받을 수 있는 평균 지원금액!


# In[65]:


mean_data_explode = mean_data_explode.sort_values('amount') #값으로 오름차순.


# In[66]:


mean_data_explode #ㅠㅠ황족고경


# In[ ]:


mean_data_explode.to_csv('mean_data_explode.csv', encoding = 'UTF-8') #필요하면 저장


# In[67]:


fig = px.bar(mean_data_explode, x='major', y='amount', color = 'amount') #이게 plotly 라이브러리!


# In[68]:


fig.show() #아이이뻐~


# In[ ]:


### 1차 EDA 끝. 추가적인 것들 진행예정.

