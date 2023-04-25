# STS : 문장 간 유사도 측정 프로젝트
- Naver boostcamp LV1 Basic Project NLP 03
- 프로젝트 전체 기간 (2주) : 4월 10일 (월) 10:00 ~ 4월 20일 (목) 19:00
<br>

## 1️. 프로젝트 개요
- Semantic Text Similarity (STS) Task    
  - 두 문장을 입력받고 두 문장의 의미적 유사도를 점수로 나타내기
  - 두 문장이 의미가 유사할수록 높은 점수를 주고 그렇지 않을 수록 낮은 점수를 줘야함 (0점 ~ 5점 사이)
  
- 환경
    - 팀 구성 및 컴퓨팅 환경 : 5인 1팀(로또냥냥펀치). 인당 V100 서버를 VS code와 SSH로 연결하여 사용
    - 협업 환경 : Notion, GitHub
    - 의사소통 : Slack, Zoom
    
- 프로젝트 구조 및 사용 데이터셋
    - 프로젝트 구조: <br> Transformer 계열의 모델을 통해 구현한 Cross Encoder를 통해 각 문장의 유사도를 측정하고 그 값을 0에서 5사이의 값으로 스케일링 한 후 출력
    - 사용 데이터셋: <br> 문장 짝과 그 문장 짝의 유사도를 가지고 있는 데이터셋을 활용하였고 각각 Train은 9324개, Valid는 550개, Test는 1100개의 데이터로 이루어져 있음
<br>

## 2️. 수행 결과
|index|Model|Test Pearson|Data handling|비고|
|:-:|----------------------|:-----:|:-------|--------|
|1|snunlp/KR-ELECTRA-discriminator|0.9336|문장 순서 변경|10 epoch|
|2|snunlp/KR-ELECTRA-discriminator|0.9295|문장 순서 변경|30 epoch|
|3|snunlp/KR-ELECTRA-discriminator|0.9304|자음/모음 단독으로 반복 전처리, 사람 토큰 추가, <br> 문장 순서 변경|25 epoch|
|4|snunlp/KR-ELECTRA-discriminator|0.9317|자음/모음 단독으로 반복 전처리, 사람 토큰 추가, <br>label 1 이상인 데이터에 대해서만 문장 순서 변경|25 epoch|
|5|snunlp/KR-ELECTRA-discriminator|0.9325|맞춤법 교정 전처리, <br> label 1 이상인 데이터에 대해서만 문장 순서 변경|25 epoch|

- 최종 제출 모델 : [1],[4],[5] 앙상블 모델 (test pearson : 0.9307)
- 제출 모델 중 가장 성능이 좋은 모델 : [1],[4] soft voting 앙상블 모델 (test pearson :0.9337)
- public leader board와 private leader board 사이의 차이가 큼. public leader board 점수를 기준으로 최종 모델 선정하였으나 이는 잘못된 방식<br>(일반화 성능이 높은 모델을 기준으로 선택해야함)
