from flask import Flask, request, jsonify
from flask_restx import Api
from flask_cors import CORS
from backend.process_part import student_info, first_filter, student_val, similarity_scholarship, filter_engine
import sqlite3
import os

conn = sqlite3.connect(os.path.abspath("") + "/database/peachtree.db")
cur = conn.cursor()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)
CORS(app)


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
    semester = request.args.get('semester')
    semester = (int(semester[0])-1)*2+int(semester[2])

    line = [name, age, sex, id_students, major, avg_score, last_score, place, income, semester]

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
