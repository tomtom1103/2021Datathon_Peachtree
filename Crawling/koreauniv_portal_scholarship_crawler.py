from selenium.webdriver import Chrome
from urllib.request import urlretrieve
import pandas as pd
import shutil
import glob
import os
import re
import time


#실행코드
#driver = Chrome()
#driver.get("https://portal.korea.ac.kr/front/Intro.kpd")

# 폴더 제작 함수
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# 장학금 공지의 테이블 명 긁어오는 함수
def crawl_table(data):
    body_num = ((1, 2), (1, 4), (2, 2), (2, 4), (3, 0), (4, 2), (4, 4), (4, 6), (5, 0))  # 상단테이블 위치
    content_list = []
    for _ in body_num:
        if _[1] != 0:
            body_path = (
                'body > div > div.page > form > table > tbody > tr:nth-child({0}) > td:nth-child({1})'.format(str(_[0]),str(_[1])))
        else:
            body_path = ('body > div > div.page > form > table > tbody > tr:nth-child({}) > td'.format(str(_[0])))

        content_list.append(driver.find_element_by_css_selector(body_path).text)  # 앞 테이블 내용

    content_list.append(driver.find_element_by_css_selector('body > div > div.page > form > table > tbody > tr:nth-child(6) > td').text)  # 내용

    data = data.append(pd.Series(content_list, index=data.columns), ignore_index=True)
    return data

def imagecrawler():
    n=1
    newfoldername=driver.find_element_by_css_selector('body > div > div.page > form > table > tbody > tr:nth-child(5) > td').text
    newfoldername=re.sub('/', '_', newfoldername)# / 를 _로 바꿔주는 정규식
    newfoldername=re.sub('\"','',newfoldername) #"를 없애주는 정규식
    newpath=('./img/{}/'.format(newfoldername))#저장할 경로
    newfile=('./img/{}/{}_{}.jpg'.format(newfoldername,newfoldername,n))#이미지 명까지 포함된 경로
    createFolder(newpath)
    for _ in driver.find_elements_by_xpath('/html/body/div/div[2]/form/table/tbody//img'):
        urlretrieve(_.get_attribute('src'),newfile)
        n+=1

#장학금 공지에 올라와 있는 첨부파일 저장하는 함수
def filedownloader():
    newfoldername=driver.find_element_by_css_selector('body > div > div.page > form > table > tbody > tr:nth-child(5) > td').text
    newfoldername=re.sub('/', '_', newfoldername)# / 를 _로 바꿔주는 정규식
    newfoldername=re.sub('\"','',newfoldername) #"를 없애주는 정규식
    newfoldername=re.sub('>',' ',newfoldername)
    newfoldername=re.sub('[\(\)]','',newfoldername)
    newpath=('./첨부파일/{}'.format(newfoldername))#저장할 경로
    if not os.path.isdir(newpath): #이미 다운로드 받았으면 다시 하지 않는다.
        createFolder(newpath)
        for _ in driver.find_elements_by_xpath('/html/body/div/div[2]/form/table/tbody/tr[7]/td/p/a'):
            _.click()
            time.sleep(0.4) #0.4초 기다렸는데도 안받아지는 놈은 클릭해도 반응이 없는 놈
        #time.sleep(5) #다운로드 될때까지 기다려주는 놈 첨할땐 주석 처리해도 됨
        source = r'C:\Users\woojo\Downloads' #파일 다운로드된 폴더(초기에 비어있어야 함)
        files = os.listdir(source)

        while True: #다운로드 다 안됐는데 넘어가면 시스템 오류남
            if len(glob.glob1(source,"*.crdownload"))!=0:
                print('wait until download is finish')
            else:
                break
        if len(driver.find_elements_by_xpath('/html/body/div/div[2]/form/table/tbody/tr[7]/td/p/a')) != len(files):
            num=len(driver.find_elements_by_xpath('/html/body/div/div[2]/form/table/tbody/tr[7]/td/p/a'))-len(files)
            open(newpath+'/ 첨부파일 {}개부족' .format(num),'w') #누락된 파일이 있으면 첨부파일부족이 뜸
        for file in files:
            new_path = shutil.move(f"{source}/{file}", newpath)

#메인 크롤러 함수
def surfer(data):
    page_num=1
    while True:
        if page_num==1:
            driver.switch_to.frame("_component")
        tab_num=len(driver.find_elements_by_xpath('//*[@id="Search"]/table/tbody/tr'))#테이블 아이탬 몇개있는지 확인
        print(f"페이지 번호 : {page_num}, 탭 수: {tab_num}")
        for i in range(1,tab_num+1):
            if i!=1:#볼드체 제목 클릭
                driver.switch_to.frame("_component")
            path=('//*[@id="Search"]/table/tbody/tr[%d]/td[3]/b/a' %i)
            try: #일반 제목 클릭
                element=driver.find_element_by_xpath(path)
            except:
                path=('//*[@id="Search"]/table/tbody/tr[%d]/td[3]/a' %i)
                element=driver.find_element_by_xpath(path)

            element.click()#여기가 클릭하는 부분 (이밑으로 크롤러 넣으면 됨)
            data=crawl_table(data)#테이블 긁어오는 함수 (이밑으로 사진 저장, 파일 저장 함수 넣으면 됨)
            if driver.find_elements_by_xpath('/html/body/div/div[2]/form/table/tbody//img'): #이미지가 있으면
                imagecrawler()#이미지 긁어오는 함수
            if driver.find_elements_by_xpath('/html/body/div/div[2]/form/table/tbody/tr[7]/td/p/a'): #첨부파일이 있으면
                filedownloader()#첨부파일 다운로드하는 함수
            print(path) #어디쯤인가 확인.
            time.sleep(0.1)
            driver.back()
        driver.switch_to.frame("_component")
        btn_next=driver.find_element_by_css_selector('#Search > div.paging > div > a.btn.next')
        if btn_next.get_attribute('href')[-4:]=='prev':#맨마지막 페이지 도달 하면 break
            print("끝도달")
            break
        btn_next.click()
        page_num+=1
        time.sleep(0.1)
    return data

def runthistocrawlcurrentscholarship():
    driver = Chrome()
    driver.get("https://portal.korea.ac.kr/front/Intro.kpd")
    with open('idpw.txt', 'r') as f:  # idpw는 따로 저장
        data = f.read()
    id_, pw_ = re.split(r'(\n)', data)[0], re.split(r'(\n)', data)[2]

    # ID 입력
    driver.find_element_by_name('id').clear()
    driver.find_element_by_name('id').send_keys(id_)
    # password 입력
    driver.find_element_by_name('pw').clear()
    driver.find_element_by_name('pw').send_keys(pw_)
    # 로그인
    time.sleep(2)
    driver.find_element_by_name('loginsubmit').click()

    time.sleep(2)
    driver.find_element_by_xpath('//*[@id="header"]/div[2]/div/div/ul/li[2]/a').click()

    sc_df = surfer(sc_df)

def runthistocrawlpastscholarships():
    driver.back()
    driver.find_element_by_xpath('//*[@id="m102"]/a').click()  # 장학 클릭
    driver.find_element_by_xpath('//*[@id="sm110"]/a').click()  # 지난 장학금 공지 클릭
    year = list(range(2010, 2022))  # 2010년 부터 2021년까지 있음
    for k in year:
        driver.switch_to.frame("_component")
        driver.find_element_by_xpath('//*[@id="Search"]/div[2]/div[1]/div/input[1]').clear()  # 년도창비우기
        driver.find_element_by_xpath('//*[@id="Search"]/div[2]/div[1]/div/input[1]').send_keys(k)  # 년도 입력
        driver.find_element_by_xpath('//*[@id="Search"]/div[2]/div[1]/span[2]/input').click()  # 검색 클릭
        driver.find_element_by_css_selector('#Search > div.paging > div > a.btn.first').click()  # 1페이지로 초기화
        driver.find_element_by_xpath('//*[@id="Search"]/table/tbody/tr[1]/td[3]/a').click()  # 아무페이지나 먼저 클릭
        driver.back()
        sc_df = surfer(sc_df)
        driver.back()
        sc_df

def runthisfordataEDA():
    # 제목 ,폴더나 파일명에 사용될 수 없는 기호들 삭제
    for i in range(len(sc_df)):
        temp = sc_df['제목'][i]
        temp = re.sub(r'/', '_', temp)
        temp = re.sub('\"', '', temp)
        temp = re.sub('>', ' ', temp)
        temp = re.sub(':', ' ', temp)
        temp = re.sub('[\(\)]', '', temp)
        sc_df.iloc[i]['제목'] = temp

    sc_df.drop_duplicates(inplace=True)
    sc_df.reset_index(inplace=True)

    sc_df.to_excel("장학금데이터.xlsx")
    final_sc = pd.read_excel("장학금데이터.xlsx")

    # 내용도 TXT 파일로 보면 좋을 것 같아 추출
    for i in range(len(final_sc)):
        newfoldername = (final_sc.iloc[i]['제목'])
        newfoldername = re.sub(':', ' ', newfoldername)
        newpath = ('./txt/{}/'.format(newfoldername))  # 저장할 경로
        if not os.path.isdir(newpath):
            createFolder(newpath)
            with open('{}{}.txt'.format(newpath, newfoldername), 'w', encoding='UTF8')as f:
                f.write(str(final_sc.iloc[i]['내용']))

def runthisforsavinglocal():
    # 넘버링이 되어있는 통합폴더명 제작
    final_sc['제목2'] = ""
    for i in range(len(final_sc)):
        final_sc['제목'][i] = re.sub(':', ' ', final_sc['제목'][i])
        final_sc['제목2'][i] = f'{str(i).zfill(4)}_' + final_sc.iloc[i]['제목']

    # 통합폴더에 다 옮겨서 저장하기
    for i in range(len(final_sc)):
        totalpath = ('./total/{}/'.format(final_sc.iloc[i]['제목2']))
        imagepath = ('./img/{}/'.format(final_sc.iloc[i]['제목']))
        txtpath = ('./txt/{}/'.format(final_sc.iloc[i]['제목']))
        filepath = ('./첨부파일/{}/'.format(final_sc.iloc[i]['제목']))
        createFolder(totalpath)

        if os.path.isdir(filepath):
            files = os.listdir(filepath)
            for file in files:
                shutil.copy(f"{filepath}/{file}", totalpath)
        if os.path.isdir(imagepath):
            files = os.listdir(imagepath)
            for file in files:
                shutil.copy(f"{imagepath}/{file}", totalpath)
        if os.path.isdir(txtpath):
            files = os.listdir(txtpath)
            for file in files:
                shutil.copy(f"{txtpath}/{file}", totalpath)