import cv2
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym
from gymnasium import spaces

# CSV 파일 로드 (파일 경로는 상황에 맞게 조정)
df_age = pd.read_csv("../data/category_weight_age.csv")
df_sex = pd.read_csv("../data/category_weight_sex.csv")
df_time = pd.read_csv("../data/category_weight_time.csv")
df_season = pd.read_csv("../data/category_weight_season.csv")

# 광고 카테고리 목록
AD_CATEGORIES = ['교양_오락_문화', '교육', '교통', '내구재', '외식', '의류']

# 상태 인코딩을 위한 사전 정의
AGE_CATEGORIES = ["20세미만", "20-30세", "31-40세", "41-50세", "51-60세"]
GENDER_CATEGORIES = ["남성", "여성"]
# 감정은 예시로 happy, neutral, sad로 가정한다.
EMOTION_CATEGORIES = ["happy", "neutral", "sad"]
TIME_CATEGORIES = ["오전", "오후"]
WEATHER_CATEGORIES = ["봄", "여름", "가을", "겨울"]

def one_hot_encode(item, category_list):
    vec = np.zeros(len(category_list))
    if item in category_list:
        idx = category_list.index(item)
        vec[idx] = 1
    return vec

def encode_state(state):
    """
    state는 딕셔너리 형태로,
    {"age": "31-40세", "gender": "여성", "emotion": "happy", "time": "오후", "weather": "봄"}
    형식이다.
    각 항목을 one-hot encoding하여 state 벡터를 생성한다.
    """
    age_vec = one_hot_encode(state["age"], AGE_CATEGORIES)
    gender_vec = one_hot_encode(state["gender"], GENDER_CATEGORIES)
    emotion_vec = one_hot_encode(state["emotion"], EMOTION_CATEGORIES)
    time_vec = one_hot_encode(state["time"], TIME_CATEGORIES)
    weather_vec = one_hot_encode(state["weather"], WEATHER_CATEGORIES)
    state_vector = np.concatenate([age_vec, gender_vec, emotion_vec, time_vec, weather_vec])
    return torch.FloatTensor(state_vector)

def get_initial_bias(state):
    """
    CSV 파일들을 활용하여, 상태에 해당하는 행의 가중치들을 평균내어 초기 편향 벡터를 생성한다.
    state 딕셔너리의 각 항목(연령, 성별, 시간, 계절)을 이용한다.
    반환값은 numpy array로, shape = (6,) 광고 카테고리별 편향 값이다.
    """
    bias_vectors = []
    # 연령 편향
    row = df_age[df_age["연령"] == state["age"]]
    if not row.empty:
        vec = row.iloc[0][AD_CATEGORIES].values.astype(float)
        bias_vectors.append(vec)
    # 성별 편향
    row = df_sex[df_sex["성별"] == state["gender"]]
    if not row.empty:
        vec = row.iloc[0][AD_CATEGORIES].values.astype(float)
        bias_vectors.append(vec)
    # 시간 편향
    row = df_time[df_time["시간대"] == state["time"]]
    if not row.empty:
        vec = row.iloc[0][AD_CATEGORIES].values.astype(float)
        bias_vectors.append(vec)
    # 계절(날씨) 편향
    row = df_season[df_season["계절"] == state["weather"]]
    if not row.empty:
        vec = row.iloc[0][AD_CATEGORIES].values.astype(float)
        bias_vectors.append(vec)
    if bias_vectors:
        combined_bias = np.mean(bias_vectors, axis=0)
    else:
        combined_bias = np.zeros(len(AD_CATEGORIES))
    return combined_bias

# DQN 신경망 정의
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay Buffer 정의
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# DQN Agent 정의
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, initial_bias, training=True):
        """
        상태와 CSV로부터 산출된 초기 편향을 결합하여 행동(광고 카테고리)를 결정한다.
        epsilon-greedy 방식을 사용한다.
        """
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if training and random.random() < epsilon:
            action = random.randrange(self.action_dim)
            return action
        else:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state)
                # 초기 편향을 더한다. (각 상태별 CSV 기반 가중치)
                bias = torch.FloatTensor(initial_bias).to(self.device)
                q_values = q_values + bias
                action = q_values.argmax().item()
                return action

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        q_values = self.policy_net(state_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = reward_batch + self.gamma * next_q_value * (1 - done_batch)

        loss = nn.MSELoss()(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 환경 클래스 정의 (gym.Env를 상속하여, 실시간 광고 추천 환경을 모사한다.)
class AdEnv(gym.Env):
    def __init__(self):
        super(AdEnv, self).__init__()
        # 상태 벡터 차원은 one-hot 인코딩 결과로 5 + 2 + 3 + 2 + 4 = 16
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(AD_CATEGORIES))

        self.current_state = None

    def reset(self, seed=None, options=None):
        # reset 메서드 추가: gymnasium.Env의 표준에 맞춰 구현
        # seed와 options 파라미터를 추가하여 gymnasium의 reset 인터페이스와 호환되게 함
        super().reset(seed=seed)
        
        # options 파라미터를 통해 초기 상태를 받을 수 있도록 함
        if options is not None and 'state_vector' in options:
            self.current_state = options['state_vector']
        else:
            # 기본 상태 벡터 생성 (모두 0)
            self.current_state = np.zeros(16, dtype=np.float32)
        
        # gymnasium.Env의 reset은 상태와 추가 정보를 튜플로 반환
        return self.current_state, {}

    def step(self, action, reward):
        # 한 에피소드는 한 사용자에 대한 추천으로 구성한다.
        # action은 광고 카테고리 index, reward는 실제 응시시간(초)
        # 여기서는 에피소드가 종료되었다고 가정한다.
        done = True
        info = {"recommended_ad": AD_CATEGORIES[action]}
        return self.current_state, reward, done, False, info  # gymnasium 0.26.0부터 truncated가 추가됨

# 실시간 광고 추천 시스템 통합 코드
def main():
    # 상태 벡터 차원은 16, 행동 수는 6이다.
    state_dim = 16
    action_dim = len(AD_CATEGORIES)
    agent = DQNAgent(state_dim, action_dim)
    env = AdEnv()

    batch_size = 32
    target_update_interval = 50  # 일정 스텝마다 target network 업데이트
    step_count = 0

    # 카메라와 객체 탐지, 시선 추적 코드는 기존 코드를 그대로 사용한다.
    # 여기서는 시뮬레이션용 더미 데이터를 생성하여 에이전트 동작을 확인한다.
    # 실제 통합 시, 아래 dummy_user_data 대신 실시간으로 추출한 데이터를 넣으면 된다.

    dummy_users = [
        {"age": "31-40세", "gender": "여성", "emotion": "happy", "time": "오후", "weather": "봄"},
        {"age": "20-30세", "gender": "남성", "emotion": "neutral", "time": "오전", "weather": "여름"},
        {"age": "41-50세", "gender": "여성", "emotion": "sad", "time": "오후", "weather": "가을"},
    ]

    # 무한 반복하여 사용자 등장 시 광고 추천 및 학습을 진행한다.
    while True:
        # 실제 시스템에서는 새로운 사용자가 탐지되면, 해당 사용자의 state 정보를 추출한다.
        # 여기서는 dummy_users에서 순차적으로 가져온다.
        for user in dummy_users:
            # 상태 인코딩
            state_vector = encode_state(user)
            # CSV 파일 기반 초기 편향 계산
            initial_bias = get_initial_bias(user)
            
            # 환경 리셋: options 딕셔너리를 통해 상태 벡터 전달
            state, _ = env.reset(options={'state_vector': state_vector.numpy()})
            
            # 에이전트가 행동(광고 카테고리)를 선택한다.
            action = agent.select_action(state_vector, initial_bias, training=True)
            recommended_ad = AD_CATEGORIES[action]
            print(f"추천 광고: {recommended_ad} / 사용자 정보: {user}")

            # 여기서 실시간으로 광고를 출력하고, 사용자 응시시간(gaze_time)을 측정한다.
            # 실제 시스템에서는 카메라 모듈과 연결하여 측정하며, 아래는 시뮬레이션 값이다.
            simulated_gaze_time = random.uniform(0.5, 5.0)  # 0.5초 ~ 5초 사이 응시했다고 가정
            reward = simulated_gaze_time  # 보상은 응시시간(초)
            print(f"측정된 응시시간: {reward:.2f} sec")

            # 환경 step 수행 (에피소드 종료)
            next_state, r, done, truncated, info = env.step(action, reward)  # truncated 파라미터 추가
            
            # 다음 상태는 새 사용자가 탐지될 때마다 reset하므로, 여기서는 next_state = state 그대로 사용
            agent.replay_buffer.push(state, action, reward, state, done)

            # 에이전트 학습
            agent.update(batch_size)
            step_count += 1
            if step_count % target_update_interval == 0:
                agent.update_target()
                print("타깃 네트워크 업데이트")

            # 추천 광고를 1초 안에 출력해야 하므로, 여기서는 sleep(1) 후 다음 사용자로 넘어간다고 가정한다.
            time.sleep(1)

        # 실제 시스템에서는 새로운 사용자가 등장할 때까지 기다린다.
        # 시뮬레이션에서는 dummy_users를 반복한다.

if __name__ == "__main__":
    main()