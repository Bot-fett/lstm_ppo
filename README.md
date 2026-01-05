LSTM + PPO 주식 트레이딩 코드 설명
이 저장소는 시계열 LSTM 모델로 다음 시점 수익률을 예측하고,
그 예측 결과를 활용해 PPO 강화학습으로 주식 매매 정책을 학습하는 구조입니다.

핵심 코드는 두 파일입니다.

lstm.py
ppo.py
1. lstm.py — LSTM 수익률 예측 모델
역할 요약
시계열 주가 및 각종 기술지표를 입력으로 받아
“다음 타임스텝의 로그 수익률(Log_Return)” 을 예측하는 LSTM 모델을 학습합니다.
티커별로 학습된 모델 가중치(*_lstm.pth)와 스케일러(*_scaler_*.pkl)를 저장하며,
이 결과는 PPO 코드에서 미래 수익률 예측 피처로 사용됩니다.
주요 구성 요소
하이퍼파라미터 및 피처 정의

공통 티커: AAPL, AMD, AMZN, GOOGL, META, NVDA, PLTR, TSLA
시퀀스 길이 SEQ_LENGTH = 60
→ 과거 60개 시점의 데이터를 한 번에 입력으로 사용
입력 피처 예시: Open, High, Low, Close, Volume, RSI, MACD, VWAP, ATR, News_Sentiment, Fear_Greed_Index, XLK 등
→ 실제 CSV에 있는 컬럼만 자동 선택해서 사용
LSTM 모델 클래스 StockLSTM

구조: LSTM(input_size → hidden_size) + Linear(hidden_size → 1)
입력: (배치 크기, 시퀀스 길이=60, 피처 개수)
출력: 스칼라 1개 → “다음 시점의 로그 수익률”
동작:
전체 시퀀스를 LSTM에 통과시킨 뒤
마지막 타임스텝의 hidden state만 뽑아서 선형 레이어에 통과
회귀 문제이므로 MSE 손실로 학습
데이터 처리 및 타깃 생성

CSV 로드 후 결측치는 전방 채우기(ffill) + 남은 NaN 제거
타깃 변수 Log_Return 생성:
Log_Return = log(Close_t / Close_{t-1})
입력 X: 위에서 정의한 FEATURE 컬럼들
타깃 y: Log_Return (shape: (N, 1))
학습/검증 분리 & 스케일링

시계열 순서를 유지한 채 앞 80% = 학습, 뒤 20% = 검증으로 분리
MinMaxScaler로 입력 X와 타깃 y를 각각 스케일링
fit은 학습 구간에서만, 검증 구간은 transform만 적용 → 데이터 누수 방지
시퀀스 생성 로직

슬라이딩 윈도우 방식:
입력 시퀀스: X[i : i+60]
타깃: 시퀀스 바로 다음 시점 y[i+60]
이렇게 해서 (샘플 수, 60, 피처 수) 형태의 3차원 입력 텐서 생성
학습 루프

옵티마이저: Adam, 손실함수: MSELoss
에폭마다:
미니배치 학습으로 train loss 계산
전체 검증 구간으로 val loss 계산
검증 손실이 최소일 때의 가중치를 티커별로 저장:
models/{티커}_lstm.pth
나중에 PPO에서 같은 스케일링을 쓰기 위해 스케일러도 함께 저장:
models/{티커}_scaler_X.pkl, models/{티커}_scaler_y.pkl
2. ppo.py — LSTM 예측을 활용한 PPO 트레이딩
역할 요약
lstm.py에서 학습한 LSTM 모델을 불러와
**“다음 시점 수익률 예측값 + 현재 기술지표 + 포지션 상태”**를 관측값으로 사용하는
주식 트레이딩 환경을 정의하고,
Stable-Baselines3의 PPO 알고리즘으로
매수/매도/보유 정책을 학습하는 코드입니다.
주요 구성 요소
LSTM 모델 재사용

StockLSTM 클래스를 동일 구조로 다시 정의
각 티커별로 저장된:
models/{티커}_lstm.pth
models/{티커}_scaler_X.pkl, models/{티커}_scaler_y.pkl
을 로드해 예측 전용으로 사용
여기서 LSTM의 역할:
과거 60개 구간의 피처 시퀀스를 받아
**“다음 시점 로그 수익률(예상값)”**을 출력
이 값이 PPO 에이전트의 관측값 중 하나로 들어감
강화학습 환경 StockTradingEnv (Gym Env)

관측값(Observation)
한 시점에서 에이전트가 보는 상태는 길이 10짜리 실수 벡터입니다.

LSTM 예측 수익률 (다음 시점 Log_Return 예상값)
현재 수익률 (직전 봉 대비 가격 변화율)
VWAP 괴리율 (현재가 vs VWAP)
RSI 정규화 값 (RSI / 100)
MACD 값
ATR / Close (가격 대비 변동성 비율)
뉴스 심리지수 (News_Sentiment)
공포/탐욕 지수 (Fear_Greed_Index / 100)
자산 중 주식 비중 (shares_value / total_assets)
자산 중 현금 비중 (cash / total_assets)
이렇게 **“미래 수익률 예측 + 현재 시장 상태 + 포트폴리오 상태”**를
한 번에 상태로 제공하여, 에이전트가 보다 풍부한 정보를 바탕으로 의사결정을 하도록 설계되어 있습니다.

행동(Action)
행동 공간은 Discrete(3) 으로 정의되어 있습니다.

0 : 매도 (보유 주식이 있다면 전량 매도)
1 : 보유 (아무 행동도 하지 않음)
2 : 매수 (수수료를 고려해 현재 자금으로 살 수 있는 최대 주식 수량 매수)
즉, 단일 종목에 대해 전량 매수 / 전량 매도 / 홀딩 세 가지 전략을 학습합니다.

보상(Reward)
한 타임스텝에서의 보상은
**자산 변화율(%)**로 정의됩니다.

[
\text{reward} =
\frac{\text{현재 자산} - \text{이전 자산}}
{\text{이전 자산}} \times 100
]

수익이 나면 양의 보상
손실이 나면 음의 보상
따라서 PPO 에이전트는 장기적으로 총 자산을 최대화하도록 정책을 학습하게 됩니다.
PPO 학습 흐름 train_ppo(ticker)

한 티커에 대해 다음 과정을 수행합니다.

시계열 데이터 로드 및 전처리

LSTM 학습에 사용했던 것과 동일한 CSV 파일을 이용
시계열 순서를 유지한 채 앞 80%는 학습용, 뒤 20%는 검증용으로 분리
LSTM 및 스케일러 로드

StockLSTM 인스턴스를 만들고 저장된 가중치 로드
X, y에 사용했던 MinMaxScaler 로드를 통해
LSTM 입력/출력 스케일을 동일하게 복원
환경 생성

학습 환경:
DummyVecEnv([lambda: StockTradingEnv(train_df, ...)])
(Stable-Baselines3에서 요구하는 벡터화 환경 포맷)
검증 환경:
StockTradingEnv(val_df, ...) 단일 환경
PPO 에이전트 학습

정책 네트워크: MlpPolicy (MLP 기반)
주요 하이퍼파라미터 예시:
학습률 3e-4
배치 크기 64
n_steps = 2048
model.learn(total_timesteps=30000)
→ 지정된 timesteps 동안 환경을 탐험하며 정책 업데이트
모델 저장 및 검증

학습 완료 후:
ppo_models/{티커}_ppo 로 PPO 모델 저장
검증 구간에서 deterministic 정책으로 시뮬레이션:
최종 자산, 수익 금액, 수익률(%), 거래 횟수, 총 보상 등을 계산
각 티커의 성과를 요약해서 출력 및 집계
여러 티커에 대한 반복 학습

메인 블록에서 정의된 TICKERS 리스트에 대해
train_ppo 를 순차적으로 호출합니다.
각 종목별 성과를 모아서 콘솔에 요약 출력하고,
별도의 CSV 파일로도 저장할 수 있도록 구성되어 있습니다.
