import sqlite3
import numpy as np
import pandas as pd
import os


def student_info(line):
    conn = sqlite3.connect(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/database/peachtree.db")
    cur = conn.cursor()
    sql = "INSERT INTO students VALUES('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')"
    cur.execute(sql.format(line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9]))
    conn.commit()
    conn.close()


def scholarship_info(data):
    id_scholarship = data[0]
    scholarship_name = data[1]
    year = data[2]
    in_school = data[3]
    if in_school == 0:
        in_school = '교외'
    else:
        in_school = '교내'
    activity = data[4]
    if activity == 0:
        activity = '활동성'
    else:
        activity = '수혜성'
    characteristic = data[5]
    if characteristic == 0:
        characteristic = '소득연계'
    elif characteristic == 1:
        characteristic = "성적연계"
    elif characteristic == 2:
        characteristic = "소득연계 & 성적연계"
    elif characteristic == 3:
        characteristic = "해당없음"
    major = data[6]
    sem_min = data[7]
    sem_max = data[8]
    sex = data[9]
    if sex == 0:
        sex = "여성"
    elif sex == 1:
        sex = "남성"
    elif sex == 2:
        sex = "상관없음"
    age_min = data[10]
    age_max = data[11]
    grade_min = float(data[12])
    grade_max = float(data[13])
    last_grade_min = float(data[14])
    last_grade_max = float(data[15])
    pause = data[16]
    if pause == 0:
        pause = "휴학"
    elif pause == 1:
        pause = "재학"
    elif pause == 2:
        pause = "상관없음"

    income_min = data[17]
    income_max = data[18]
    characteristic_money = data[19]
    if characteristic_money == 0:
        characteristic_money = "등록금"
    elif characteristic_money == 1:
        characteristic_money = "지원금"
    elif characteristic_money == 2:
        characteristic_money = "등록금 & 지원금"
    recommendation = data[20]
    if recommendation == 0:
        recommendation = "필요하지 않습니다."
    elif recommendation == 1:
        recommendation = "필요합니다"
    region = data[21]
    link = data[22]
    date_start = data[23]
    date_end = data[24]
    scholarship_price = float(data[25]) * 10000
    paybyhour = data[26]
    if paybyhour == 0:
        paybyhour = "아님"
    else:
        paybyhour = "맞음"
    feature_integer = data[27]
    feature = data[28]
    feature_specified = data[29]
    other = data[30]

    return {
        'name': scholarship_name,
        'year': year,
        'in_school': in_school,
        'activity': activity,
        'characteristic': characteristic,
        'major': major,
        'sem_min': sem_min,
        'sem_max': sem_max,
        'sex': sex,
        'age_min': age_min,
        'age_max': age_max,
        'grade_min': grade_min,
        'grade_max': grade_max,
        'last_grade_min': last_grade_min,
        'last_grade_max': last_grade_max,
        'pause': pause,
        'income_min': income_min,
        'income_max': income_max,
        'characteristic_money': characteristic_money,
        'recommendation': recommendation,
        'region': region,
        'link': link,
        'date_start': date_start,
        'date_end': date_end,
        'scholarship_price': scholarship_price,
        'paybyhour': paybyhour,
        'feature': feature,
        'feature_specified': feature_specified,
        'other': other
    }


def cluster_info(scholarship_id_list):
    conn = sqlite3.connect(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/database/peachtree.db")
    cur = conn.cursor()
    result_dict = {}

    for i in range(len(scholarship_id_list)):
        id = scholarship_id_list[i]
        cur.execute(f"SELECT * FROM scholarship where id_scholarship=='{id}'")
        data = list((cur.fetchall())[0])

        result_dict[i] = scholarship_info(data)

    conn.close()

    return result_dict


def big_cluster_info(cluster_list):
    result_dict = {}
    for i in range(len(cluster_list)):
        scholarship_id_list = cluster_list[i]
        result_dict[i] = cluster_info(scholarship_id_list)
    return result_dict


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
                       1,
                       -1, -1]

    # 직전성적은 직전성적의 mincut
    last_grade_min_set = list(set(df['last_grade_min']))
    last_grade_min_set = sorted(last_grade_min_set)
    '''last_grade_min_label=[]
    for _ in last_grade_min_set:
        last_grade_min_label.append(int(df[df['last_grade_min']==_]['label_2'].head(1)))'''
    # 계산 복잡도를 줄이기 위해 결과 바로 입력
    last_grade_min_label = [0, -1, -1, -1, -1, -1, 5, 6, -1, 3, 3, 3, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 7,
                            7]

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


def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


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
