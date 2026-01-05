LSTM + PPO 주식 트레이딩 실험
이 리포지토리는 시간대별 주가 및 기술지표 데이터를 활용해

LSTM으로 다음 시점 로그 수익률을 예측하고,
그 예측값을 포함한 상태를 사용해 PPO 강화학습으로 매수/매도 정책을 학습하는 실험 코드입니다.
주요 코드는 다음 두 파일에 있습니다.

lstm.py — LSTM 수익률 예측 모델 학습
ppo.py — LSTM 예측을 활용한 PPO 강화학습 트레이딩

공통 설정
두 파일 모두 공통으로 다음 설정을 사용합니다.

종목 리스트
TICKERS = ['AAPL', 'AMD', 'AMZN', 'GOOGL', 'META', 'NVDA', 'PLTR', 'TSLA']

입력 피처 (실제 CSV에 존재하는 컬럼만 사용)

가격 및 거래량: Open, High, Low, Close, Volume
기술 지표: RSI, MACD, MACD_Signal, MA20, VWAP, ATR
심리/외부 지표: News_Sentiment, Fear_Greed_Index, XLK
공통 하이퍼파라미터

시퀀스 길이: SEQ_LENGTH = 60 (과거 60개의 시점으로 입력 구성)
초기 자본: INITIAL_BALANCE = 10000
수수료: TRANSACTION_FEE = 0.0005 (0.05%)
1. lstm.py — LSTM 수익률 예측 모델
목적
각 종목별 시계열 데이터를 이용해 “다음 타임스텝의 로그 수익률(Log_Return)” 을 예측하는 LSTM 모델을 학습합니다.

입력: 과거 60개 시점의 여러 특징(feature)들
출력: 다음 시점의 로그 수익률 1개 값
주요 구성
경로/하이퍼파라미터 설정

DATA_DIR, MODEL_DIR 설정
SEQ_LENGTH, HIDDEN_SIZE, NUM_LAYERS, EPOCHS, BATCH_SIZE, LEARNING_RATE 등 정의
모델 클래스 StockLSTM

nn.LSTM + nn.Linear 구조
입력: (batch, seq_len, input_size)
마지막 타임스텝의 hidden state를 받아 output_size=1 로 회귀
학습 함수 train_model(ticker)

데이터 로드
우선 data/{TICKER}_hourly_dataset.csv
없으면 data/{TICKER}_hourly_alp_yf_dataset_v2.csv
전처리
결측치 전방 채우기(ffill) 후 dropna
타겟 생성
Log_Return = log(Close_t / Close_{t-1})
입력/타겟 분리
X = FEATURES 중 실제 존재하는 컬럼만
y = Log_Return
학습/검증 분리 (80% / 20%)
스케일링
MinMaxScaler 로 X, y 를 train 구간 기준으로만 fit
시퀀스 생성
슬라이딩 윈도우로 (길이 60 시퀀스, 그 다음 시점의 y) 생성
모델 학습
MSELoss, Adam 사용
validation loss 기준으로 best 모델 저장
결과 저장
models/{TICKER}_lstm.pth
models/{TICKER}_scaler_X.pkl, models/{TICKER}_scaler_y.pkl


2. ppo.py — LSTM + PPO 강화학습 트레이딩
목적
이미 학습된 LSTM 모델이 예측하는 다음 시점 수익률과
여러 기술지표, 포지션 상태 등을 관측값으로 사용해,
PPO 에이전트가 “언제 매수/매도/보유할지”를 학습하도록 합니다.

주요 구성
LSTM 클래스 StockLSTM 재정의

lstm.py 의 구조와 동일
여기서는 학습이 아니라, 저장된 가중치 로드 후 예측 전용으로 사용
강화학습 환경 StockTradingEnv (Gym 환경)

상태(관측값, 길이 10)

LSTM 예측 수익률 (다음 시점 로그 수익률)
현재 수익률 (직전 봉 대비)
VWAP 괴리율
RSI 정규화 (RSI/100)
MACD 값
ATR / Close (변동성 비율)
뉴스 심리지수 (News_Sentiment)
공포/탐욕 지수 (Fear_Greed_Index/100)
주식 비중 (자산 중 주식 가치 비율)
현금 비중 (자산 중 현금 비율)
행동 공간 (spaces.Discrete(3))

0: 전량 매도 (보유 주식 있으면 모두 매도)
1: 보유 (아무 행동 안 함)
2: 가능한 한 최대 수량 매수 (수수료 포함)
보상 함수

한 스텝에서의 자산 변화율(%):
[
reward = \frac{\text{현재자산} - \text{이전자산}}{\text{이전자산}} \times 100
]
즉, 수익이 나면 양의 보상, 손실이면 음의 보상
에피소드 종료 조건

시계열의 마지막 시점까지 도달하면 done = True
학습 함수 train_ppo(ticker)

데이터 로드 및 전처리 (LSTM 때와 동일 CSV 로직)
학습/검증 분리 (80% / 20%)
LSTM 모델 및 스케일러 로드
models/{TICKER}_lstm.pth
models/{TICKER}_scaler_X.pkl, models/{TICKER}_scaler_y.pkl
환경 생성
학습: DummyVecEnv([lambda: StockTradingEnv(train_df, ...)])
검증: StockTradingEnv(val_df, ...)
PPO 모델 학습
PPO 모델 저장
ppo_models/{TICKER}_ppo
검증 루프
검증 구간에서 deterministic 정책으로 평가
최종 자산, 수익, 수익률, 거래 횟수 출력 및 dict로 반환
