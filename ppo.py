import pandas as pd
import numpy as np
import torch
import joblib
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ==========================================
# [1] 설정 영역
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PPO_MODEL_DIR = os.path.join(BASE_DIR, "ppo_models")

if not os.path.exists(PPO_MODEL_DIR):
    os.makedirs(PPO_MODEL_DIR)

TICKERS = ['AAPL', 'AMD', 'AMZN', 'GOOGL', 'META', 'NVDA', 'PLTR', 'TSLA']
SEQ_LENGTH = 60
INITIAL_BALANCE = 10000 
TRANSACTION_FEE = 0.0005  # 수수료

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature 리스트 (LSTM과 동일해야 함)
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
    'RSI', 'MACD', 'MACD_Signal', 'MA20', 
    'VWAP', 'ATR', 'News_Sentiment', 'Fear_Greed_Index', 'XLK'
]

# ==========================================
# [2] LSTM 클래스 (구조 동일)
# ==========================================
class StockLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# [3] 주식 거래 환경 (Gym)
# ==========================================
class StockTradingEnv(gym.Env):
    def __init__(self, df, lstm_model, scaler_X, scaler_y, initial_balance=10000, transaction_fee=0.0005):
        super(StockTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.lstm_model = lstm_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Action: 0=매도, 1=보유, 2=매수
        self.action_space = spaces.Discrete(3)
        
        # Observation Space 정의
        # 1. LSTM예측수익률, 2. 현재수익률(전봉대비), 3. VWAP괴리율, 4. RSI/100, 
        # 5. MACD, 6. ATR/Close(변동성비율), 7. 심리지수, 8. 공포탐욕/100, 9. 보유비율, 10. 현금비율
        self.obs_dim = 10 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Feature 컬럼 인덱싱 준비
        self.feature_cols = [f for f in FEATURES if f in self.df.columns]
        
        self.current_step = SEQ_LENGTH
        self.balance = initial_balance
        self.shares_held = 0
        self.total_assets = initial_balance
        self.max_assets = initial_balance
        self.trades = []

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = SEQ_LENGTH
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_assets = self.initial_balance
        self.max_assets = self.initial_balance
        self.trades = []
        return self._get_observation(), {}

    def _get_lstm_prediction(self):
        """LSTM을 이용해 '다음 타임스텝의 예상 수익률' 예측"""
        if self.current_step < SEQ_LENGTH:
            return 0.0
        
        # LSTM 입력 데이터 추출 (SEQ_LENGTH 만큼)
        sequence = self.df[self.feature_cols].iloc[self.current_step - SEQ_LENGTH:self.current_step].values
        # 스케일링
        sequence_scaled = self.scaler_X.transform(sequence)
        
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
            # 예측된 스케일된 수익률
            pred_scaled = self.lstm_model(seq_tensor).cpu().numpy()[0, 0]
        
        # 원래 수익률 스케일로 복원
        pred_return = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
        return pred_return

    def _get_observation(self):
        # 현재 데이터 가져오기
        row = self.df.iloc[self.current_step]
        prev_close = self.df.iloc[self.current_step - 1]['Close']
        
        # 1. LSTM 예측 (예상 수익률)
        predicted_return = self._get_lstm_prediction()
        
        # 2. 현재 변동률 (전봉 대비)
        current_return = (row['Close'] - prev_close) / prev_close
        
        # 3. VWAP 괴리율 (현재가가 VWAP보다 얼마나 높냐/낮냐)
        vwap_diff = (row['Close'] - row['VWAP']) / row['VWAP'] if 'VWAP' in row else 0
        
        # 4. 기타 지표 정규화
        rsi_norm = row['RSI'] / 100.0 if 'RSI' in row else 0.5
        macd_val = row['MACD'] if 'MACD' in row else 0
        atr_ratio = (row['ATR'] / row['Close']) if 'ATR' in row else 0 # 가격 대비 변동성
        sentiment = row['News_Sentiment'] if 'News_Sentiment' in row else 0
        fear_greed = row['Fear_Greed_Index'] / 100.0 if 'Fear_Greed_Index' in row else 0.5
        
        # 5. 포트폴리오 상태 (정규화)
        total_val = self.balance + self.shares_held * row['Close']
        shares_ratio = (self.shares_held * row['Close']) / total_val # 자산 중 주식 비중 (0~1)
        cash_ratio = self.balance / total_val # 자산 중 현금 비중 (0~1)

        obs = np.array([
            predicted_return,
            current_return,
            vwap_diff,
            rsi_norm,
            macd_val,
            atr_ratio,
            sentiment,
            fear_greed,
            shares_ratio,
            cash_ratio
        ], dtype=np.float32)
        
        # NaN 방지
        return np.nan_to_num(obs)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        prev_assets = self.total_assets
        
        # 행동 수행
        if action == 0:  # 매도
            if self.shares_held > 0:
                sell_amount = self.shares_held * current_price * (1 - self.transaction_fee)
                self.balance += sell_amount
                self.trades.append({'action': 'SELL', 'price': current_price, 'step': self.current_step})
                self.shares_held = 0
                
        elif action == 2:  # 매수
            if self.balance > current_price:
                max_shares = int(self.balance / (current_price * (1 + self.transaction_fee)))
                if max_shares > 0:
                    cost = max_shares * current_price * (1 + self.transaction_fee)
                    self.balance -= cost
                    self.shares_held += max_shares
                    self.trades.append({'action': 'BUY', 'price': current_price, 'step': self.current_step})

        # 자산 갱신
        self.total_assets = self.balance + self.shares_held * current_price
        
        # 보상 계산: (현재 자산 - 이전 자산) / 이전 자산 * 100 (퍼센트 단위 보상)
        reward = ((self.total_assets - prev_assets) / prev_assets) * 100
        
        # 페널티: 너무 매매를 안하면(Hold만 하면) 약간의 페널티를 주어 매매 유도
        # if action == 1:
        #     reward -= 0.001 

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        return self._get_observation(), reward, done, truncated, {'total_assets': self.total_assets}

# ==========================================
# [4] PPO 학습 함수
# ==========================================
def train_ppo(ticker):
    print(f"\n🤖 [{ticker}] PPO 강화학습 시작...")
    
    # 데이터 로드
    file_path = os.path.join(DATA_DIR, f"{ticker}_hourly_dataset.csv")
    if not os.path.exists(file_path):
        file_path = os.path.join(DATA_DIR, f"{ticker}_hourly_alp_yf_dataset_v2.csv") # 파일명 주의
        
    df = pd.read_csv(file_path)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    # LSTM 모델 로드
    feature_cols = [f for f in FEATURES if f in df.columns]
    lstm_model = StockLSTM(input_size=len(feature_cols))
    lstm_model.load_state_dict(torch.load(f"{MODEL_DIR}/{ticker}_lstm.pth", map_location=device))
    lstm_model.to(device)
    lstm_model.eval()
    
    scaler_X = joblib.load(f"{MODEL_DIR}/{ticker}_scaler_X.pkl")
    scaler_y = joblib.load(f"{MODEL_DIR}/{ticker}_scaler_y.pkl")

    # 학습/검증 분리
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    # 학습/검증 데이터 크기 출력
    print(f"📊 학습 데이터: {len(train_df)}행, 검증 데이터: {len(val_df)}행")
    
    # 환경 생성
    train_env = DummyVecEnv([lambda: StockTradingEnv(train_df, lstm_model, scaler_X, scaler_y, INITIAL_BALANCE, TRANSACTION_FEE)])
    val_env = StockTradingEnv(val_df, lstm_model, scaler_X, scaler_y, INITIAL_BALANCE, TRANSACTION_FEE)
    
    # 모델 정의 및 학습
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=3e-4, batch_size=64, n_steps=2048)
    model.learn(total_timesteps=30000) # 학습 횟수 조절 가능

    # 모델 저장
    model.save(f"{PPO_MODEL_DIR}/{ticker}_ppo")
    print(f"✅ PPO 모델 저장 완료: {ticker}")

    # 검증
    print(f"\n🔍 [{ticker}] 검증 시작...")
    obs, _ = val_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = val_env.step(action)
        total_reward += reward
        if truncated:
            break

    final_assets = info['total_assets']
    profit = final_assets - INITIAL_BALANCE
    profit_rate = (profit / INITIAL_BALANCE) * 100

    print(f"\n{'='*50}")
    print(f"[{ticker}] 검증 결과")
    print(f"{'='*50}")
    print(f"초기 자본: ${INITIAL_BALANCE:,.2f}")
    print(f"최종 자산: ${final_assets:,.2f}")
    print(f"수익: ${profit:,.2f} ({profit_rate:.2f}%)")
    print(f"총 보상: {total_reward:.4f}")
    print(f"거래 횟수: {len(val_env.trades)}")
    print(f"{'='*50}\n")

    return {
        'ticker': ticker,
        'initial': INITIAL_BALANCE,
        'final': final_assets,
        'profit': profit,
        'profit_rate': profit_rate,
        'total_reward': total_reward,
        'trades': len(val_env.trades)
    }

if __name__ == "__main__":
    print("="*60)
    print("PPO 강화학습 기반 주식 트레이딩")
    print("="*60)

    results = []

    for ticker in TICKERS:
        try:
            result = train_ppo(ticker)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # 전체 결과 요약
    if results:
        print("\n" + "="*60)
        print("전체 결과 요약")
        print("="*60)
        for r in results:
            print(f"{r['ticker']}: ${r['profit']:,.2f} ({r['profit_rate']:.2f}%), 거래: {r['trades']}회")

        avg_profit_rate = np.mean([r['profit_rate'] for r in results])
        print(f"\n평균 수익률: {avg_profit_rate:.2f}%")
        print("="*60)

        # 검증 결과 CSV 저장
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(PPO_MODEL_DIR, "ppo_train_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n💾 검증 결과 CSV 저장 완료: {summary_path}")