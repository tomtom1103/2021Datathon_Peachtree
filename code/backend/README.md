# 최적반의 도원결의 - 당장!

"당장!" 의 Backend Engine 인 peachtree 의 내부 구성은 다음과 같다:

### >peachtree
#### main.py
#### >>backend
##### __init__.py
##### process_part.py

#### >>database
##### dataDB.csv
##### db_instruct.py
##### peachtree.db
##### DBSCAN.py

# process_part.py
process_part.py 는 당장! 의 엔진을 구현하는데 있어 필요한 함수들의 모음이다.


#### Library Import
```python
import sqlite3
import numpy as np
import pandas as pd
import os
```
필요한 패키지 import.


#### student_info 함수
```python
def student_info(line):
    conn = sqlite3.connect(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/database/peachtree.db")
    cur = conn.cursor()
    sql = "INSERT INTO students VALUES('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')"
    cur.execute(sql.format(line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9]))
    conn.commit()
    conn.close()
```
student_info 함수는 사용자가 웹에서 입력한 값들을 DB의 students 테이블에 입력하는 sql 함수다. 인수로는 main.py 의 line 을 사용한다.


#### student_val 함수
```python
def student_val(line):
    # 평균학점,직전학점,소득분위 데이터 받음
    st_grade, st_lgrade, st_income = line[5], line[6], line[8]
    st_grade_temp, st_lgrade_temp = 0.0, 0.0

    # 그와 관련된 장학금의 column들 db에서 불러옴 (dataframe으로 저장)
    conn = sqlite3.connect(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/database/peachtree.db")
    cur = conn.cursor()
    cur.execute(f"SELECT grade_min, last_grade_min, income_max, label_1, label_2, label_3 FROM scholarship")
    df = cur.fetchall()
    conn.close()
    df = pd.DataFrame(df)
    df.columns = ["grade_min", "last_grade_min", "income_max", "label_1", "label_2", "label_3"]

    # 각 값들이 어느 라벨과 대응되는지 **_label 리스트에 똑같은 index에 저장
    # 평균성적은 평균성적의 min cut
    grade_min_set = list(set(df['grade_min']))
    grade_min_set = sorted(grade_min_set)
    '''grade_min_label=[]
    for _ in grade_min_set:
        grade_min_label.append(int(df[df['grade_min']==_]['label_1'].head(1)))'''
    # 계산 복잡도를 줄이기 위해 결과 바로 입력
    grade_min_label = [2, -1, -1, -1, -1, -1, -1, -1, -1, 3, 3, 3, 3, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, -1, -1]

    # 직전성적은 직전성적의 mincut
    last_grade_min_set = list(set(df['last_grade_min']))
    last_grade_min_set = sorted(last_grade_min_set)
    '''last_grade_min_label=[]
    for _ in last_grade_min_set:
        last_grade_min_label.append(int(df[df['last_grade_min']==_]['label_2'].head(1)))'''
    # 계산 복잡도를 줄이기 위해 결과 바로 입력
    last_grade_min_label = [0, -1, -1, -1, -1, -1, 5, 6, -1, 3, 3, 3, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 7, 7]

    # 소득분위는 소득분위의 max cut
    income_max_set = list(set(df['income_max']))
    income_max_set = sorted(income_max_set)
    '''income_max_label=[]
    for _ in income_max_set:
        income_max_label.append(int(df[df['income_max']==_]['label_3'].head(1)))'''
    # 계산 복잡도를 줄이기 위해 결과 바로 입력
    income_max_label = [1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0]

    # 학생들의 정보가 장학금 column의 값들의 어디에 설정되는지 정의
    # 예를들어 학생의 직전 성적 3.75 면 3.7이 같거나 작은 최대의 수이므로 3.7로 입력됨
    for i in grade_min_set:
        if st_grade < i:
            break
        st_grade_temp = i
    for i in last_grade_min_set:
        if st_lgrade < i:
            break
        st_lgrade_temp = i
    st_grade, st_lgrade = st_grade_temp, st_lgrade_temp

    st_grade_label = grade_min_label[grade_min_set.index(st_grade)]
    st_lgrade_label = last_grade_min_label[last_grade_min_set.index(st_lgrade)]
    st_income_label = income_max_label[income_max_set.index(st_income)]

    return ([st_grade_label, st_lgrade_label, st_income_label])

```
line 에 있는 학생의 평균학점, 직전학점 소득분위 데이터를 받는다. 이 데이터를 조합해서 3개의 DBSCAN 클러스터에 대해 클러스터의 하위클러스터 인덱스를 저장한 리스트를 반환해준다. 내부에 있는 Docstring 을 없애고 함수를 돌리면 직접 입력한 값들이 나오는 것을 볼 수 있다.

#### first_filter 함수
```python
def first_filter(year=2021, id=None):
    conn = sqlite3.connect(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/database/peachtree.db")
    curs = conn.cursor()

    curs.execute('SELECT * FROM scholarship')
    result_scholarship = curs.fetchall()
    scholarship = pd.DataFrame(result_scholarship)
    scholarship.columns = ['id_scholarship', 'scholarship_name', 'year', 'in_school', 'activity',
                           'characteristic', 'major', 'sem_min', 'sem_max', 'sex', 'age_min',
                           'age_max', 'grade_min', 'grade_max', 'last_grade_min', 'last_grade_max',
                           'pause', 'income_min', 'income_max', 'characteristic_money',
                           'recommendation', 'region', 'link', 'date_start', 'date_end',
                           'scholarship_price', 'paybyhour', 'feature_integer', 'feature',
                           'feature_specified', 'other', 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9',
                           'l10',
                           'l11', 'l12', 'l13', 'l14']

    scholarship['date_start'] = pd.to_datetime(scholarship['date_start'])
    scholarship['date_end'] = pd.to_datetime(scholarship['date_end'])

    curs.execute(f"SELECT * FROM students WHERE id_students == '{id}'")
    result_student = curs.fetchall()
    result_student = pd.DataFrame(result_student)
    result_student.columns = ['name', 'age', 'sex', 'id_students', 'major', 'last_score', 'avg_score', 'place',
                              'income', 'semester']
    user = result_student

    if user['sex'][0] == '남자':
        global filtered_scholarship
        filtered_scholarship = scholarship[
            (scholarship['sex'] != 0) &  # 성별
            (scholarship['major'].str.contains(user['major'][0])) &  # 학과
            (scholarship['grade_min'] <= user['avg_score'][0]) &  # 평균학점
            (scholarship['grade_max'] >= user['avg_score'][0]) &
            (scholarship['last_grade_min'] <= user['last_score'][0]) &  # 직전학점
            (scholarship['last_grade_max'] >= user['last_score'][0]) &
            (scholarship['income_min'] <= user['income'][0]) &  # 소득분위
            (scholarship['income_max'] >= user['income'][0]) &
            (scholarship['sem_min'] <= user['semester'][0]) &
            (scholarship['sem_max'] >= user['semester'][0])]

    else:
        filtered_scholarship = scholarship[
            (scholarship['sex'] != 1) &
            (scholarship['major'].str.contains(user['major'][0])) &
            (scholarship['grade_min'] <= user['avg_score'][0]) &
            (scholarship['grade_max'] >= user['avg_score'][0]) &
            (scholarship['last_grade_min'] <= user['last_score'][0]) &
            (scholarship['last_grade_max'] >= user['last_score'][0]) &
            (scholarship['income_min'] <= user['income'][0]) &
            (scholarship['income_max'] >= user['income'][0]) &
            (scholarship['sem_min'] <= user['semester'][0]) &
            (scholarship['sem_max'] >= user['semester'][0])]

    global thisyear
    thisyear = pd.to_datetime(f'{year}-01-01')  # 지정된 연도부터 출력

    filtered_scholarship = filtered_scholarship[(filtered_scholarship['date_start'] >= thisyear)]
    filtered_scholarship = filtered_scholarship.replace('nan', np.NaN)
    global temp_scholarship
    temp_scholarship = filtered_scholarship[~(pd.isna(filtered_scholarship['region']))]
    temp_scholarship = filtered_scholarship.drop(list(temp_scholarship.index))
    return_list = list(np.array(temp_scholarship['id_scholarship'].tolist()))

    return return_list

```
first_filter은 인자로 지정 연도와 사용자의 PK 값인 학번을 입력받는다. Sqlite 로 DB와 커넥션을 열고, 전체 DB (scholarship table, students table) 을 불러온다. if 문부터 실질적인 함수가 돌아가는데, 학생의 입력 정보 토대로 해당하지 않는 장학금을 거르고 지원할 수 있는 장학금의 PK 값인 id_scholarship 을 리스트로 반환해준다.

#### cos_similarity 함수

```python
def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)
```
cos_similarity 함수는 인자로 두개의 list vector을 받아 numpy의 np.dot() 함수를 사용하여 두 벡터의 코사인 유사도를 구한다. 

#### similarity_scholarship 함수
```python
def similarity_scholarship(student_value, ff_list):
    onehot_column_names = ('labels_1_-1', 'labels_1_0', 'labels_1_1', 'labels_1_2', 'labels_1_3',
                           'labels_2_-1', 'labels_2_0', 'labels_2_1', 'labels_2_2', 'labels_2_3',
                           'labels_2_4', 'labels_2_5', 'labels_2_6', 'labels_2_7', 'labels_3_0',
                           'labels_3_1', 'labels_3_2')

    conn = sqlite3.connect(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/database/peachtree.db")
    curs = conn.cursor()

    curs.execute('SELECT id_scholarship, label_1, label_2, label_3 FROM scholarship')
    scholarship = curs.fetchall()
    conn.close()
    scholarship = pd.DataFrame(scholarship)
    scholarship.columns = ['id_scholarship', 'labels_1 : grade_min', 'labels_2 : last_grade_min',
                           'labels_3 : income_max']

    scholarship_onehot = scholarship[scholarship['id_scholarship'].isin(ff_list)]
    scholarship_onehotlist = scholarship_onehot.values.tolist()
    scholarship_clustset = pd.DataFrame(0, index=np.arange(len(scholarship_onehotlist)),
                                        columns=onehot_column_names)

    for i in range(len(scholarship_onehotlist)):
        for j in range(1, len(scholarship_onehotlist[i])):
            scholarship_clustset[f'labels_{j}_{scholarship_onehotlist[i][j]}'][i] += 1

    xuser_clustset = pd.DataFrame(columns=onehot_column_names)
    xuser_clustset = xuser_clustset.append(pd.Series(0, index=xuser_clustset.columns), ignore_index=True)

    for i in range(len(student_value)):
        xuser_clustset[f'labels_{i + 1}_{student_value[i]}'][0] += 1

    x = xuser_clustset.to_numpy()
    x = x[0]
    z = scholarship_clustset.to_numpy()

    similarity_list = []

    for _ in z:
        similarity_list.append(cos_similarity(x, _))

    lst1 = similarity_list
    lst2 = ff_list

    best_similarity = pd.DataFrame(
        {'similarity_list': lst1,
         'return_list': lst2
         })
    best_similarity = best_similarity.sort_values(by='similarity_list', ascending=False)
    best_similarity_pk = best_similarity['return_list'].tolist()

    return best_similarity_pk

```
similarity_scholarship 은 DBSCAN 으로 진행한 장학금에 대한 클러스터링의 인덱스와 학생의 클러스터링 인덱스의 코사인 유사도를 구해주는 함수다. student_val의 결과값인 리스트와 first_filter의 결과값인 리스트를 인자로 받는다. DBSCAN 클러스터 결과물의 인덱스값을 각 장학금에 대해 정의하고, 원핫 인코딩을 통해 dataframe을 만들어준다. 동시에 학생의 입력 정보에 대해 클러스터링 결과물의 인덱스값을 원핫 인코딩한 dataframe 도 만들어준다. 학생의 원핫 인코딩 한 클러스터링 벡터와 각 장학금에 대한 원핫 인코딩 한 클러스터링 벡터를 for 문으로 이전에 정의한 cos_similarity 함수를 통해 코사인 유사도를 구해준다. 코사인 유사도가 가장 높은 장학금의 PK 값인 id_scholarship 을 리스트로 반환해 준다.

#### first_engine 함수

```python
def filter_engine(ss_list):
    conn = sqlite3.connect(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/database/peachtree.db")
    curs = conn.cursor()
    sql = 'SELECT * FROM scholarship WHERE id_scholarship in ({seq})'.format(seq=','.join(['?'] * len(ss_list)))
    curs.execute(sql, ss_list)
    scholarship = curs.fetchall()
    curs.execute('SELECT * FROM scholarship')
    names = list(map(lambda x: x[0], curs.description))
    conn.close()
    scholarship = pd.DataFrame(scholarship, columns=names)
    result_dic = []
    # ai가 추천하는 상위 5개 장학금
    ai_list = ss_list[0:5]
    result_dic.append({"cluster_name": "ai_recommendation", "cluster_contents": cluster_info(ai_list)})

    # feature integer 기준 1:경제상황, 2:개인/가족신분, 3:특정단체,  4:기타, 0: nan 값
    fi_scholarhip = scholarship[scholarship.feature_integer != 0]  # 0 즉 nan 제거
    if fi_scholarhip['feature_integer'].value_counts().max() >= 3:  # 같은 기준이 3개 이상 되면 출력
        fi_val = fi_scholarhip['feature_integer'].value_counts().idxmax()
        if fi_val == 1:
            fi_val = "경제 상황"
        elif fi_val == 2:
            fi_val = "개인/가족 신분"
        elif fi_val == 3:
            fi_val = "특정 단체"
        else:
            fi_val = "기타"
        result_dic.append({"cluster_name": "feature", "cluster_contents": cluster_info(
            list(np.array(fi_scholarhip[fi_scholarhip.feature_integer == fi_val]['id_scholarship'].tolist()))),
                           "cluster_specific": fi_val})

    # 지역기준
    rg_scholarship = scholarship.replace('nan', np.NaN)
    rg_scholarship = rg_scholarship[rg_scholarship['region'].notna()]
    if len(rg_scholarship) >= 2:
        result_dic.append({"cluster_name": "region",
                           "cluster_contents": cluster_info(list(np.array(rg_scholarship['id_scholarship'].tolist())))})

    # activity 기준 : 활동성(0) 수혜성(1)
    result_dic.append({"cluster_name": "activity_0", "cluster_contents": cluster_info(
        list(np.array(scholarship[scholarship['activity'] == 0]['id_scholarship'].tolist())))})
    result_dic.append({"cluster_name": "activity_1", "cluster_contents": cluster_info(
        list(np.array(scholarship[scholarship['activity'] == 1]['id_scholarship'].tolist())))})

    # characteristic 기준: 소득연계(0) 성적연계(1) 소득연계&성적연계(2) nan(3)
    ch_scholarship = scholarship[scholarship.characteristic != 3]
    if ch_scholarship['characteristic'].value_counts().max() >= 3:
        result_dic.append({"cluster_name": "characteristic_0", "cluster_contents": cluster_info(
            list(np.array(scholarship[scholarship['characteristic'] == 0]['id_scholarship'].tolist())))})
        result_dic.append({"cluster_name": "characteristic_1", "cluster_contents": cluster_info(
            list(np.array(scholarship[scholarship['characteristic'] == 1]['id_scholarship'].tolist())))})

    # charcteristic_money 기준 : 등록금(0), 그 외(1), 둘다(2)
    result_dic.append({"cluster_name": "characteristic_money_0", "cluster_contents": cluster_info(
        list(np.array(scholarship[scholarship['charcteristic_money'] == 0]['id_scholarship'].tolist())))})
    result_dic.append({"cluster_name": "characteristic_money_1", "cluster_contents": cluster_info(
        list(np.array(scholarship[scholarship['charcteristic_money'] == 1]['id_scholarship'].tolist())))})
    result_dic.append({"cluster_name": "characteristic_money_2", "cluster_contents": cluster_info(
        list(np.array(scholarship[scholarship['charcteristic_money'] == 2]['id_scholarship'].tolist())))})

    result_dic = {i: result_dic[i] for i in range(len(result_dic))}

    return result_dic
```
first_engine 함수는 first_filter의 return 값인 리스트를 인자로 받는다. 학생과 유사도가 가장 높은 장학금들에 대해 그 장학금이 DB에 있는 다른 장학금들과의 유사도를 구하는 함수다.


# main.py

main.py는 당장! 의 엔진이다. 전체적인 흐름을 설명하고자 한다.

```python from flask import Flask, request, jsonify
from flask_restx import Api
from flask_cors import CORS
from backend.process_part import student_info, big_cluster_info, first_filter, student_val, similarity_scholarship
import sqlite3
import os
```
필요한 패키지 import.

```python
conn = sqlite3.connect(os.path.abspath("") + "/database/peachtree.db")
cur = conn.cursor()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)
CORS(app)
```
peachtree 의 database 폴더엔 생성한 .db 파일이 있다. sqlite로 커넥션을 생성 후 커서를 정의하여 sql문을 사용 가능하게 만들어준다. 추가적으로 Flask library 로 웹앱을 정의했다.

```python
@app.route('/main_request', methods=['GET'])
def main_request():
    name = request.args.get('name')
    age = int(request.args.get('age'))
    sex = request.args.get('sex')
    id_students = int(request.args.get('id_students'))
    major = request.args.get('major')
    last_score = float(request.args.get('last_score'))
    avg_score = float(request.args.get('avg_score'))
    place = request.args.get('place')
    income = int(request.args.get('income'))
    semester = int(request.args.get('semester'))

    line = [name, age, sex, id_students, major, avg_score, last_score, place, income, semester]

```
학생이 입력하는 정보들을 GET 방식으로 불러오는 부분이다. 모든 정보를 불러와 line 이란 리스트에 저장한다.


```python
    # 입력값 업로드
    student_info(line)

    # 학생의 장학금 VALUE 생성 엔진
    student_value = student_val(line)

    # 기본 정보로 지원가능한 장학금 1차 필터링
    ff_list = first_filter(year=2021, id=id_students)

    # 유사도가 높은 장학금 코드 리턴 엔진
    ss_list = similarity_scholarship(student_value, ff_list)

    # 추천 장학금 id 리턴 (클러스터 별)
    result = filter_engine(ss_list)

    # 이를 json으로 묶어서 보내주기
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
```
process_part.py 에서 정의한 함수들을 순차적으로 돌리는 부분이다. 최종 result를 jsonify library 를 사용하여 json 형식으로 다시 웹으로 response 를 보내준다.

# dataDB.csv
dataDB.csv는 당장!의 원천데이터다.


# db_instruct.py
db_instruct.py 는 초기에 .db 를 생성하기 위한 sql query문이다. dataDB.csv를 query 문으로 .db에 append 한다.

# peachtree.db
db_instructy.py 를 통해 생성한 local db다. 모든 함수와 엔진은 이 .db 파일에서 데이터를 query 문으로 불러오는 형식으로 구동된다.


# DBSCAN.py

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv('dataDB_other.csv')
idvars=['Unnamed: 0.1','Unnamed: 0','id_scholarship','scholarship_name','link','other']
unusedvars=['major','region','date_start','date_end','feature','feature_specified',
            'age_max','grade_max','last_grade_max','recommendation','year']
data_input = data.drop(idvars, axis = 1)
data_input = data_input.drop(unusedvars, axis = 1)

scaler = MinMaxScaler()

data_input_array = scaler.fit_transform(data_input)
data_input = pd.DataFrame(data_input_array, columns=data_input.columns)

data_input_1=data_input[['grade_min']]
X=np.array(data_input_1)

db_1=DBSCAN(eps = 0.03, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_1.labels_, dtype=bool)
core_samples_mask[db_1.core_sample_indices_] = True
labels_1 = db_1.labels_
data['labels_1 : grade_min'] = labels_1

n_clusters_1 = len(set(labels_1)) - (1 if -1 in labels_1 else 0)
n_noise_1 = list(labels_1).count(-1)

print('Variable : grade_min')
print('Estimated number of clusters: %d' % n_clusters_1)
print('Estimated number of noise points: %d' % n_noise_1)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_1))
print()

data_input_2=data_input[['last_grade_min']]
X=np.array(data_input_2)
db_2=DBSCAN(eps = 0.03, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_2.labels_, dtype=bool)
core_samples_mask[db_2.core_sample_indices_] = True
labels_2 = db_2.labels_
data['labels_2 : last_grade_min'] = labels_2

n_clusters_2 = len(set(labels_2)) - (1 if -1 in labels_2 else 0)
n_noise_2 = list(labels_2).count(-1)

print('Variable : last_grade_min')
print('Estimated number of clusters: %d' % n_clusters_2)
print('Estimated number of noise points: %d' % n_noise_2)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_2))
print()

data_input_3=data_input[['last_grade_min','grade_min']]
X=np.array(data_input_3)
db_3=DBSCAN(eps = 0.05, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_3.labels_, dtype=bool)
core_samples_mask[db_3.core_sample_indices_] = True
labels_3 = db_3.labels_
data['labels_3 : last_grade_min, grade_min'] = labels_3

n_clusters_3 = len(set(labels_3)) - (1 if -1 in labels_3 else 0)
n_noise_3 = list(labels_3).count(-1)

print('Variable : last_grade_min, grade_min')
print('Estimated number of clusters: %d' % n_clusters_3)
print('Estimated number of noise points: %d' % n_noise_3)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_3))
print()

unique_labels = set(labels_3)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_3 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.suptitle('other_similarity vs. scholarship_price')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_3, metrics.silhouette_score(X, labels_3)))
plt.show()

data_input_4=data_input[['paybyhour']]
X=np.array(data_input_4)
db_4=DBSCAN(eps = 0.05, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_4.labels_, dtype=bool)
core_samples_mask[db_4.core_sample_indices_] = True
labels_4 = db_4.labels_
data['labels_4 : paybyhour'] = labels_4

n_clusters_4 = len(set(labels_4)) - (1 if -1 in labels_4 else 0)
n_noise_4 = list(labels_4).count(-1)

print('Variable : paybyhour')
print('Estimated number of clusters: %d' % n_clusters_4)
print('Estimated number of noise points: %d' % n_noise_4)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_4))
print()

data_input_5=data_input[['characteristic','characteristic_money','scholarship_price']]
X=np.array(data_input_5)
db_5=DBSCAN(eps = 0.1, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_5.labels_, dtype=bool)
core_samples_mask[db_5.core_sample_indices_] = True
labels_5 = db_5.labels_
data['labels_5 : characteristic, characteristic_money, scholarship_price'] = labels_5

n_clusters_5 = len(set(labels_5)) - (1 if -1 in labels_5 else 0)
n_noise_5 = list(labels_5).count(-1)

print('Variable : characteristic, characteristic_money, scholarship_price')
print('Estimated number of clusters: %d' % n_clusters_5)
print('Estimated number of noise points: %d' % n_noise_5)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_5))
print()

unique_labels = set(labels_5)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_5 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')

x = data_input_5['characteristic']
y = data_input_5['characteristic_money']
z = data_input_5['scholarship_price']
ax.scatter(x,y,z,c=labels_5, s = 20, alpha = 0.5, cmap='rainbow')
plt.xlabel('characteristic')
plt.ylabel('characteristic_money')
ax.set_zlabel('scholarship_price')
plt.suptitle('characteristic(X) vs. characteristic_money(Y) vs. scholarship_price(Z)')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' %(n_clusters_5, metrics.silhouette_score(X, labels_5)))
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')

x = data_input_5['characteristic']
y = data_input_5['characteristic_money']
z = data_input_5['scholarship_price']
ax.scatter(x,y,z,c=labels_5, s = 20, alpha = 0.5, cmap='rainbow')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_5, metrics.silhouette_score(X, labels_5)))
plt.suptitle('characteristic(X) vs. characteristic_money(Y) vs. scholarship_price(Z)')
plt.show()

data_input_6=data_input[['income_max']]
X=np.array(data_input_6)
db_6=DBSCAN(eps = 0.1, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_6.labels_, dtype=bool)
core_samples_mask[db_6.core_sample_indices_] = True
labels_6 = db_6.labels_
data['labels_6 : income_max'] = labels_6

n_clusters_6 = len(set(labels_6)) - (1 if -1 in labels_6 else 0)
n_noise_6 = list(labels_6).count(-1)

print('Variable : income_max')
print('Estimated number of clusters: %d' % n_clusters_6)
print('Estimated number of noise points: %d' % n_noise_6)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_6))
print()

data_input_onehot_1=data_input[['other_similarity']]
X=np.array(data_input_onehot_1)
db_onehot_1=DBSCAN(eps = 0.015, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_1.labels_, dtype=bool)
core_samples_mask[db_onehot_1.core_sample_indices_] = True
labels_onehot_1 = db_onehot_1.labels_
data['labels_onehot_1 : other_similarity'] = labels_onehot_1

n_clusters_onehot_1 = len(set(labels_onehot_1)) - (1 if -1 in labels_onehot_1 else 0)
n_noise_onehot_1 = list(labels_onehot_1).count(-1)

print('Variable : other_similarity')
print('Estimated number of clusters: %d' % n_clusters_onehot_1)
print('Estimated number of noise points: %d' % n_noise_onehot_1)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_1))
print()

data_input_onehot_2=data_input[['other_similarity', 'scholarship_price']]
X = np.array(data_input_onehot_2)
db_onehot_2=DBSCAN(eps = 0.05, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_2.labels_, dtype=bool)
core_samples_mask[db_onehot_2.core_sample_indices_] = True
labels_onehot_2 = db_onehot_2.labels_
data['labels_onehot_2 : other_similarity'] = labels_onehot_2

n_clusters_onehot_2 = len(set(labels_onehot_2)) - (1 if -1 in labels_onehot_2 else 0)
n_noise_onehot_2 = list(labels_onehot_2).count(-1)

print('Variable : other_similarity, scholarship_price')
print('Estimated number of clusters: %d' % n_clusters_onehot_2)
print('Estimated number of noise points: %d' % n_noise_onehot_2)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_2))
print()

import matplotlib.pyplot as plt
unique_labels = set(labels_onehot_2)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_2 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('other_similarity vs. scholarship_price')
plt.suptitle('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_2, metrics.silhouette_score(X, labels_onehot_2)))
plt.show()

data_input_onehot_3=data_input[['other_similarity', 'grade_min']]
X = np.array(data_input_onehot_3)
db_onehot_3=DBSCAN(eps = 0.06, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_3.labels_, dtype=bool)
core_samples_mask[db_onehot_3.core_sample_indices_] = True
labels_onehot_3 = db_onehot_3.labels_
data['labels_onehot_3 : other_similarity, grade_min'] = labels_onehot_3

n_clusters_onehot_3 = len(set(labels_onehot_3)) - (1 if -1 in labels_onehot_3 else 0)
n_noise_onehot_3 = list(labels_onehot_3).count(-1)

print('Variable : other_similarity, grade_min')
print('Estimated number of clusters: %d' % n_clusters_onehot_3)
print('Estimated number of noise points: %d' % n_noise_onehot_3)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_3))
print()

unique_labels = set(labels_onehot_3)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_3 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.suptitle('other_similarity vs. grade_min')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_3, metrics.silhouette_score(X, labels_onehot_3)))
plt.show()

data_input_onehot_4=data_input[['other_similarity', 'characteristic_money']]
X = np.array(data_input_onehot_4)
db_onehot_4=DBSCAN(eps = 0.09, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_4.labels_, dtype=bool)
core_samples_mask[db_onehot_4.core_sample_indices_] = True
labels_onehot_4 = db_onehot_4.labels_
data['labels_onehot_4 : other_similarity, characteristic_money'] = labels_onehot_4

n_clusters_onehot_4 = len(set(labels_onehot_4)) - (1 if -1 in labels_onehot_4 else 0)
n_noise_onehot_4 = list(labels_onehot_4).count(-1)

print('Variable : other_similarity, characteristic_money')
print('Estimated number of clusters: %d' % n_clusters_onehot_4)
print('Estimated number of noise points: %d' % n_noise_onehot_4)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_4))
print()

unique_labels = set(labels_onehot_4)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_4 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.suptitle('other_similarity vs. characteristic_money')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_4, metrics.silhouette_score(X, labels_onehot_4)))
plt.show()

data_input_onehot_5=data_input[['other_similarity', 'characteristic']]
X = np.array(data_input_onehot_5)
db_onehot_5=DBSCAN(eps = 0.09, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_5.labels_, dtype=bool)
core_samples_mask[db_onehot_5.core_sample_indices_] = True
labels_onehot_5 = db_onehot_5.labels_
data['labels_onehot_5 : other_similarity, characteristic'] = labels_onehot_5

n_clusters_onehot_5 = len(set(labels_onehot_5)) - (1 if -1 in labels_onehot_5 else 0)
n_noise_onehot_5 = list(labels_onehot_5).count(-1)

print('Variable : other_similarity, characteristic')
print('Estimated number of clusters: %d' % n_clusters_onehot_5)
print('Estimated number of noise points: %d' % n_noise_onehot_5)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_5))
print()

unique_labels = set(labels_onehot_5)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_5 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.suptitle('other_similarity vs. characteristic')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_5, metrics.silhouette_score(X, labels_onehot_5)))
plt.show()

data_input_onehot_6=data_input[['other_similarity', 'feature_integer']]
X = np.array(data_input_onehot_6)
db_onehot_6=DBSCAN(eps = 0.09, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_6.labels_, dtype=bool)
core_samples_mask[db_onehot_6.core_sample_indices_] = True
labels_onehot_6 = db_onehot_6.labels_
data['labels_onehot_6 : other_similarity, feature_integer'] = labels_onehot_6

n_clusters_onehot_6 = len(set(labels_onehot_6)) - (1 if -1 in labels_onehot_6 else 0)
n_noise_onehot_6 = list(labels_onehot_6).count(-1)

print('Variable : other_similarity, feature_integer')
print('Estimated number of clusters: %d' % n_clusters_onehot_6)
print('Estimated number of noise points: %d' % n_noise_onehot_6)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_6))
print()

unique_labels = set(labels_onehot_6)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_6 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.suptitle('other_similarity vs. feature_integer')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_6, metrics.silhouette_score(X, labels_onehot_6)))
plt.show()

data_input_onehot_7=data_input[['other_similarity', 'characteristic_money', 'grade_min']]
X = np.array(data_input_onehot_7)
db_onehot_7=DBSCAN(eps = 0.09, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_7.labels_, dtype=bool)
core_samples_mask[db_onehot_7.core_sample_indices_] = True
labels_onehot_7 = db_onehot_7.labels_
data['labels_onehot_7 : other_similarity, characteristic_money, grade_min'] = labels_onehot_7

n_clusters_onehot_7 = len(set(labels_onehot_7)) - (1 if -1 in labels_onehot_7 else 0)
n_noise_onehot_7 = list(labels_onehot_7).count(-1)

print('Variable : other_similarity, characteristic_money, grade_min')
print('Estimated number of clusters: %d' % n_clusters_onehot_7)
print('Estimated number of noise points: %d' % n_noise_onehot_7)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_7))
print()

unique_labels = set(labels_onehot_7)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_7 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('other_similarity vs. characteristic_money vs. grade_min')
plt.suptitle('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_7, metrics.silhouette_score(X, labels_onehot_7)))
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')

x = X[:,2]
y = X[:,1]
z = X[:,0]
ax.scatter(x,y,z,c=labels_onehot_7, s = 20, alpha = 0.5, cmap='rainbow')
plt.xlabel('grade_min')
plt.ylabel('characteristic_money')
ax.set_zlabel('other_similarity')
plt.suptitle('grade_min(X) vs. characteristic_money(Y) vs. other_similarity(Z)')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' %(n_clusters_onehot_7, metrics.silhouette_score(X, labels_onehot_7)))
plt.show()

data_input_onehot_8=data_input[['other_similarity', 'characteristic_money', 'characteristic']]
X = np.array(data_input_onehot_8)
db_onehot_8=DBSCAN(eps = 0.12, min_samples = 20).fit(X)
core_samples_mask = np.zeros_like(db_onehot_8.labels_, dtype=bool)
core_samples_mask[db_onehot_8.core_sample_indices_] = True
labels_onehot_8 = db_onehot_8.labels_
data['labels_onehot_8 : other_similarity, characteristic_money, characteristic'] = labels_onehot_8

n_clusters_onehot_8 = len(set(labels_onehot_8)) - (1 if -1 in labels_onehot_8 else 0)
n_noise_onehot_8 = list(labels_onehot_8).count(-1)

print('Variable : other_similarity, characteristic_money, characteristic')
print('Estimated number of clusters: %d' % n_clusters_onehot_8)
print('Estimated number of noise points: %d' % n_noise_onehot_8)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels_onehot_8))
print()

unique_labels = set(labels_onehot_8)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0,1,len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0,0,0,1]

    class_member_mask = (labels_onehot_8 == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:,0],xy[:,1],'o',markerfacecolor = tuple(col),
             markeredgecolor = 'k', markersize = 14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.suptitle('other_similarity vs. characteristic_money vs. characteristic')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' % (n_clusters_onehot_8, metrics.silhouette_score(X, labels_onehot_8)))
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection = '3d')

x = data_input_onehot_8['characteristic']
y = data_input_onehot_8['characteristic_money']
z = data_input_onehot_8['other_similarity']
ax.scatter(x,y,z,c=labels_onehot_8, s = 20, alpha = 0.5, cmap='rainbow')
plt.xlabel('characteristic')
plt.ylabel('characteristic_money')
ax.set_zlabel('other_similarity')
plt.suptitle('characteristic(X) vs. characteristic_money(Y) vs. other_similarity(Z)')
plt.title('Estimated number of clusters : %d, Silhouette Coefficient : %f' %(n_clusters_onehot_8, metrics.silhouette_score(X, labels_onehot_8)))
plt.show()

data.to_excel("data2.xlsx", encoding="utf-8")

```
DBSCAN.py 는 장학금 데이터에 대해 비지도학습 클러스터링 방법론 중 하나인 DBSCAN을 사이킷-런의 Library 로 구현한 코드다. 장학금 데이터에서 사용한 세부 변수는 코드에 정의해 두었고, 사용한 hyperparameter 들 또한 정의하였다. 최종 출력은 각 클러스터링 결과물에 대한 excel file 이다.
 
