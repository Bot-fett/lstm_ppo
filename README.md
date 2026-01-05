# 📈 AI 기반 주식 예측 및 자동매매 시스템 (AI Stock Prediction & Auto-Trading System)

## 📖 프로젝트 개요 (Overview)
단순히 과거 데이터를 학습하는 것을 넘어, 예측된 정보를 바탕으로 "언제 사고파는 것이 수익을 극대화하는가?"를 판단하는 에이전트를 구축.

---

## 🏗 시스템 파이프라인 (Pipeline Architecture)

전체 시스템은 데이터 분석 -> 미래 예측(LSTM) -> 실전 매매(PPO)의 3단계로 구성됩니다.

### 1️⃣ Phase 1: Data Preprocessing
* **Input:** 8개 기술 우량주의 Hourly 데이터 (OHLCV) + 기술적 지표 + 거시경제 지표
* **Process:** * 결측치 처리 및 정렬
    * **MinMax Scaling:** 모델 학습을 위해 모든 데이터를 0~1 사이로 정규화
    * **Sliding Window:** 시계열 학습을 위해 60시간(Sequence Length) 단위로 데이터셋 구성

### 2️⃣ Phase 2: Market Prediction (LSTM)
* **Model:** **LSTM (Long Short-Term Memory)**
* **Role:** **"시장 예언가"**
* **Logic:** 과거 60시간의 주가 흐름과 보조지표 패턴을 분석하여, **다음 시점의 종가(Close Price)**를 예측합니다.
* **Objective:** 예측 오차(MSE Loss) 최소화

### 3️⃣ Phase 3: Strategy Optimization (PPO)
* **Model:** **PPO (Proximal Policy Optimization)**
* **Role:** **"트레이딩 승부사"**
* **Logic:** * 현재 시장 상황(State)과 계좌 잔고를 관찰합니다.
    * 매수(Buy), 매도(Sell), 관망(Hold) 중 가장 기대 수익이 높은 행동(Action)을 확률적으로 선택합니다.
    * 수익이 나면 보상(+), 손실이 나면 벌점(-)을 받아 점차 똑똑한 트레이더로 진화합니다.

---

## 📊 데이터셋 상세 (Dataset Details)

미국 나스닥 주요 기술주 8개 종목의 1시간 봉(Hourly) 데이터를 사용합니다.

| Feature | Description |
| :--- | :--- |
| **기간** | 2016.01 ~ 2025.12  |
| **대상 종목** | `AAPL`, `AMD`, `AMZN`, `GOOGL`, `META`, `NVDA`, `PLTR`, `TSLA` |
| **기본 시세** | `Open`, `High`, `Low`, `Close`, `Volume` |
| **기술적 지표** | `MA20` (이동평균), `RSI` (상대강도), `MACD`, `ATR` (변동성), `VWAP` |
| **거시 지표** | `VIX` (공포지수), `TNX` (국채금리), `DXY` (달러인덱스), `QQQ/XLK` (시장지수) |

> **Note:** 종목별 상장일 및 데이터 수집 현황에 따라 학습 시작 시점(Start Date)은 상이할 수 있습니다. (예: PLTR은 2020년부터 시작)

## 🧠 모델 로직 및 구조 (Model Specs)

### 1. LSTM (주가 예측 모델)
* **입력 차원 (Input Dimension):** (Batch Size, 60, Feature Size)
    * 60시간(Sequence)의 데이터를 하나의 묶음으로 입력받습니다.
* **은닉층 크기 (Hidden Dimension):** 64 (데이터의 특징을 추출하는 노드의 개수)
* **계층 구조 (Layers):** 2-Layer Stacked LSTM (2개의 층을 쌓아 복잡한 시계열 패턴 학습)
* **최적화 (Optimization):** Adam Optimizer (학습률: 0.001) + MSE(평균제곱오차) 손실 함수 사용
* **조기 종료 (Early Stopping):** 검증 손실(Val Loss)이 더 이상 줄어들지 않으면 학습을 자동으로 멈춰 과적합(Overfitting) 방지

### 2. PPO (트레이딩 에이전트)
* **환경 (Environment):** `Gymnasium` 인터페이스를 따른 자체 제작 주식 거래 환경
* **상태 공간 (State Space):** 에이전트가 판단을 위해 관찰하는 데이터
    * 정규화된(Normalized) 현재 시장 데이터 (가격, 거래량, 보조지표 등)
    * 현재 계좌의 현금 잔고 비율
    * 보유 중인 주식의 현재 가치
* **행동 공간 (Action Space):** 3가지 이산형(Discrete) 행동 중 하나 선택
    * `0`: **관망 (Hold)** - 현재 포지션 유지
    * `1`: **매수 (Buy)** - 주식 구매
    * `2`: **매도 (Sell)** - 주식 판매
* **보상 함수 (Reward Function):** 에이전트의 행동을 평가하는 점수
    * 포트폴리오 총자산의 변동률 (수익률 기반 보상)
    * 잦은 매매를 방지하기 위해 거래 수수료(0.05%)를 비용으로 차감
