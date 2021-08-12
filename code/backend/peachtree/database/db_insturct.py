import sqlite3
import pandas as pd

# db구성
conn = sqlite3.connect("peachtree.db")
cur = conn.cursor()

# 학생 table 생성
cur.execute(
    "CREATE TABLE students(\
    student_name VARCHAR(20) NOT NULL,\
    age INT NOT NULL,\
    sex VARCHAR(10) NOT NULL,\
    id_students INT NOT NULL,\
    major VARCHAR(100) NOT NULL,\
    avg_score FLOAT NOT NULL,\
    last_score FLOAT NOT NULL,\
    place VARCHAR(100) NULL,\
    income INT(2) NULL,\
    semester INT NULL,\
    PRIMARY KEY (`id_students`))"
)

# scholarship table 생성
cur.execute(
    "CREATE TABLE scholarship(\
    id_scholarship INT NOT NULL,\
    scholarship_name VARCHAR(80) NOT NULL,\
    scholarship_year INT NOT NULL,\
    in_school INT NOT NULL,\
    activity INT NOT NULL,\
    characteristic INT NOT NULL,\
    major VARCHAR(500) NOT NULL,\
    sem_min INT NOT NULL,\
    sem_max INT NOT NULL,\
    sex INT NOT NULL,\
    age_min INT NOT NULL,\
    age_max INT NOT NULL,\
    grade_min DECIMAL NOT NULL,\
    grade_max DECIMAL NOT NULL,\
    last_grade_min DECIMAL NOT NULL,\
    last_grade_max DECIMAL NOT NULL,\
    pause DECIMAL NOT NULL,\
    income_min INT NOT NULL,\
    income_max INT NOT NULL,\
    charcteristic_money INT NOT NULL,\
    recommendation INT NOT NULL,\
    region VARCHAR(500),\
    link VARCHAR(1000),\
    date_start TIMESTAMP NOT NULL,\
    date_end TIMESTAMP NOT NULL,\
    scholarship_price DECIMAL NOT NULL,\
    paybyhour INT NOT NULL,\
    feature_integer INT NOT NULL,\
    feature VARCHAR(100) NULL,\
    feature_specified VARCHAR(100),\
    other VARCHAR(1000) NULL,\
    label_1 INT NOT NULL,\
    label_2 INT NOT NULL,\
    label_3 INT NOT NULL,\
    label_4 INT NOT NULL,\
    label_5 INT NOT NULL,\
    label_6 INT NOT NULL,\
    label_7 INT NOT NULL,\
    label_8 INT NOT NULL,\
    label_9 INT NOT NULL,\
    label_10 INT NOT NULL,\
    label_11 INT NOT NULL,\
    label_12 INT NOT NULL,\
    label_13 INT NOT NULL,\
    label_14 INT NOT NULL,\
    PRIMARY KEY (`id_scholarship`))"
)

# csv파일 db화
df = pd.read_csv("dataDB.csv", encoding="cp949")
sql = "INSERT INTO scholarship VALUES('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}'\
,'{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')"
for index, line in df.iterrows():
    cur.execute(
        sql.format(line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10],
                   line[11], line[12], line[13], line[14], line[15], line[16], line[17], line[18], line[19], line[20],
                   line[21], line[22], line[23], line[24], line[25], line[26], line[27], line[28], line[29], line[30],
                   line[31], line[32], line[33], line[34], line[35], line[36], line[37], line[38], line[39], line[40],
                   line[41], line[42], line[43], line[44]))
conn.commit()
cur.execute('SELECT * FROM scholarship')
cur.fetchall()
