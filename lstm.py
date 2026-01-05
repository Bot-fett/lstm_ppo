import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ==========================================
# [1] 설정 영역
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# 사용할 종목 리스트
TICKERS = ['AAPL', 'AMD', 'AMZN', 'GOOGL', 'META', 'NVDA', 'PLTR', 'TSLA']

# 모델 하이퍼파라미터
SEQ_LENGTH = 60       
HIDDEN_SIZE = 64      
NUM_LAYERS = 2        
EPOCHS = 50           
BATCH_SIZE = 32       
LEARNING_RATE = 0.001 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 학습 장치: {device}")

# 학습에 사용할 변수들
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
    'RSI', 'MACD', 'MACD_Signal', 'MA20', 
    'VWAP', 'ATR', 'News_Sentiment', 'Fear_Greed_Index', 'XLK'
]

# ==========================================
# [2] LSTM 모델 정의
# ==========================================
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # 마지막 타임스텝의 결과
        return out

# ==========================================
# [3] 학습 함수
# ==========================================
def train_model(ticker):
    print(f"\n📡 [{ticker}] 모델 학습 시작...")
    
    # 1. 데이터 로드 (V2 데이터셋 권장)
    # 파일명이 상황에 따라 다를 수 있으니 확인 필요
    file_path = os.path.join(DATA_DIR, f"{ticker}_hourly_dataset.csv") 
    
    # 만약 V2 파일명이 다르다면 아래와 같이 수정하세요
    if not os.path.exists(file_path):
        file_path = os.path.join(DATA_DIR, f"{ticker}_hourly_alp_yf_dataset_v2.csv")

    if not os.path.exists(file_path):
        print(f"❌ 파일 없음: {file_path}")
        return

    df = pd.read_csv(file_path)
    df.fillna(method='ffill', inplace=True) # 결측치 보간
    df.dropna(inplace=True)

    # 2. 타겟 변수 생성 (로그 수익률)
    # 절대 가격(Close)을 맞추는 건 어렵기때문에 변화율(Return)을 맞춤
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True) # 수익률 계산으로 생긴 첫 행 NaN 제거

    # Feature 선택 (존재하는 컬럼만 필터링)
    available_features = [f for f in FEATURES if f in df.columns]
    print(f"✅ 사용된 Features: {available_features}")
    
    # X: 여러 기술적 지표들, y: 다음 시점의 로그 수익률
    X_data = df[available_features].values
    y_data = df[['Log_Return']].values

    # ======================================================
    # Data Leakage 방지: Split 후 Scaling
    # ======================================================
    split_idx = int(len(X_data) * 0.8)
    
    X_train_raw = X_data[:split_idx]
    X_val_raw = X_data[split_idx:]
    y_train_raw = y_data[:split_idx]
    y_val_raw = y_data[split_idx:]

    # 스케일러 정의
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler() # y값(수익률)도 스케일링 권장

    # Train 데이터로만 Fit
    X_train = scaler_X.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw)

    # Val 데이터는 Train 기준으로 Transform
    X_val = scaler_X.transform(X_val_raw)
    y_val = scaler_y.transform(y_val_raw)

    # 3. 시계열 데이터셋 생성 (Sliding Window)
    def create_sequences(X, y, seq_length):
        xs, ys = [], []
        for i in range(len(X) - seq_length):
            xs.append(X[i : i+seq_length])
            ys.append(y[i+seq_length]) # 다음 시점의 수익률 예측
        return np.array(xs), np.array(ys)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, SEQ_LENGTH)

    # 텐서 변환
    train_dataset = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32).to(device),
        torch.tensor(y_train_seq, dtype=torch.float32).to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32).to(device)

    # 4. 모델 초기화
    model = StockLSTM(input_size=len(available_features), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 학습 루프
    best_loss = float('inf')
    patience = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        avg_train_loss = train_loss / len(train_loader)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}")

        # Early Stopping check (선택사항)
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience = 0
            # 최고 성능 모델 저장
            torch.save(model.state_dict(), f"{MODEL_DIR}/{ticker}_lstm.pth")
        else:
            patience += 1
            
    # 스케일러 저장 (예측 시 필수)
    joblib.dump(scaler_X, f"{MODEL_DIR}/{ticker}_scaler_X.pkl")
    joblib.dump(scaler_y, f"{MODEL_DIR}/{ticker}_scaler_y.pkl") # y 스케일러도 저장
    
    print(f"✅ {ticker} 학습 완료. (Best Val Loss: {best_loss:.6f})")

if __name__ == "__main__":
    for ticker in TICKERS:
        try:
            train_model(ticker)
        except Exception as e:
            print(f"⚠️ {ticker} 에러 발생: {e}")
            import traceback
            traceback.print_exc()