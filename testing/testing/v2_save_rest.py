"""
# 기본 실행 (새로운 모델로 시작)
python main.py

# 이전 모델 이어서 학습 + 시각화 활성화
python main.py --continue_training --visualization

# 이전 모델 이어서 학습 + 시각화 + 저장 주기 조절(20스텝마다)
python main.py --continue_training --visualization --save_interval 20
"""

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
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # Setting GUI backend

# CSV 파일 로드 (파일 경로는 상황에 맞게 조정)
df_age = pd.read_csv("../data/category_weight_age.csv")
df_sex = pd.read_csv("../data/category_weight_sex.csv")
df_time = pd.read_csv("../data/category_weight_time.csv")
df_season = pd.read_csv("../data/category_weight_season.csv")

# 광고 카테고리 목록
AD_CATEGORIES = ['Culture_Entertainment', 'Education', 'Transportation', 'Durables', 'Dining', 'Clothing']

# 상태 인코딩을 위한 사전 정의
AGE_CATEGORIES = ["under20", "20-30", "31-40", "41-50", "51-60"]
GENDER_CATEGORIES = ["male", "female"]
# 감정은 예시로 happy, neutral, sad로 가정한다.
EMOTION_CATEGORIES = ["happy", "neutral", "sad"]
TIME_CATEGORIES = ["morning", "afternoon"]
WEATHER_CATEGORIES = ["spring", "summer", "fall", "winter"]

# 모델 저장/로드 경로 설정
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

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
    # Map Korean categories to English categories for visualization
    age_mapping = {"20세미만": "under20", "20-30세": "20-30", "31-40세": "31-40", 
                   "41-50세": "41-50", "51-60세": "51-60"}
    gender_mapping = {"남성": "male", "여성": "female"}
    time_mapping = {"오전": "morning", "오후": "afternoon"}
    weather_mapping = {"봄": "spring", "여름": "summer", "가을": "fall", "겨울": "winter"}
    
    # Map input state to English categories
    age_eng = age_mapping.get(state["age"], state["age"])
    gender_eng = gender_mapping.get(state["gender"], state["gender"])
    time_eng = time_mapping.get(state["time"], state["time"])
    weather_eng = weather_mapping.get(state["weather"], state["weather"])
    
    age_vec = one_hot_encode(age_eng, AGE_CATEGORIES)
    gender_vec = one_hot_encode(gender_eng, GENDER_CATEGORIES)
    emotion_vec = one_hot_encode(state["emotion"], EMOTION_CATEGORIES)
    time_vec = one_hot_encode(time_eng, TIME_CATEGORIES)
    weather_vec = one_hot_encode(weather_eng, WEATHER_CATEGORIES)
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
        # Map the Korean column names to English AD_CATEGORIES for visualization
        vec = np.zeros(len(AD_CATEGORIES))
        original_categories = ['교양_오락_문화', '교육', '교통', '내구재', '외식', '의류']
        for i, cat in enumerate(original_categories):
            if cat in row.columns:
                vec[i] = row.iloc[0][cat]
        bias_vectors.append(vec)
        
    # 성별 편향
    row = df_sex[df_sex["성별"] == state["gender"]]
    if not row.empty:
        vec = np.zeros(len(AD_CATEGORIES))
        original_categories = ['교양_오락_문화', '교육', '교통', '내구재', '외식', '의류']
        for i, cat in enumerate(original_categories):
            if cat in row.columns:
                vec[i] = row.iloc[0][cat]
        bias_vectors.append(vec)
        
    # 시간 편향
    row = df_time[df_time["시간대"] == state["time"]]
    if not row.empty:
        vec = np.zeros(len(AD_CATEGORIES))
        original_categories = ['교양_오락_문화', '교육', '교통', '내구재', '외식', '의류']
        for i, cat in enumerate(original_categories):
            if cat in row.columns:
                vec[i] = row.iloc[0][cat]
        bias_vectors.append(vec)
        
    # 계절(날씨) 편향
    row = df_season[df_season["계절"] == state["weather"]]
    if not row.empty:
        vec = np.zeros(len(AD_CATEGORIES))
        original_categories = ['교양_오락_문화', '교육', '교통', '내구재', '외식', '의류']
        for i, cat in enumerate(original_categories):
            if cat in row.columns:
                vec[i] = row.iloc[0][cat]
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
    
    def save(self, path):
        # 리플레이 버퍼를 파일로 저장
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"Replay buffer saved to {path}")
    
    def load(self, path):
        # 파일에서 리플레이 버퍼 로드
        if os.path.exists(path):
            with open(path, 'rb') as f:
                loaded_buffer = pickle.load(f)
                # 기존 버퍼 비우고 로드된 데이터로 채우기
                self.buffer.clear()
                for experience in loaded_buffer:
                    self.buffer.append(experience)
            print(f"Loaded {len(self.buffer)} experiences from {path}")
        else:
            print(f"Warning: {path} does not exist. Starting with a new buffer.")

# Visualization 클래스 추가
class Visualizer:
    def __init__(self, agent, dummy_users):
        self.agent = agent
        self.dummy_users = dummy_users
        self.fig, self.axs = plt.subplots(2, 1, figsize=(12, 10))
        self.reward_history = []
        self.q_values_history = {user_idx: {ad_idx: [] for ad_idx in range(len(AD_CATEGORIES))} 
                                for user_idx in range(len(dummy_users))}
        self.fig.suptitle('Ad Recommendation System Learning Progress')
        
        # First subplot: Reward graph
        self.axs[0].set_title('Reward History')
        self.axs[0].set_xlabel('Step')
        self.axs[0].set_ylabel('Reward (Gaze Time)')
        self.reward_line, = self.axs[0].plot([], [], 'b-')
        
        # Second subplot: Q-value change graph
        self.axs[1].set_title('Q-Values by User Type')
        self.axs[1].set_xlabel('Step')
        self.axs[1].set_ylabel('Q-Value')
        
        # Map Korean user info to English for display
        self.user_display_info = []
        for user in dummy_users:
            age_map = {"20세미만": "under20", "20-30세": "20-30", "31-40세": "31-40", 
                       "41-50세": "41-50", "51-60세": "51-60"}
            gender_map = {"남성": "male", "여성": "female"}
            time_map = {"오전": "morning", "오후": "afternoon"}
            weather_map = {"봄": "spring", "여름": "summer", "가을": "fall", "겨울": "winter"}
            
            display_info = {
                "age": age_map.get(user["age"], user["age"]),
                "gender": gender_map.get(user["gender"], user["gender"]),
                "emotion": user["emotion"],
                "time": time_map.get(user["time"], user["time"]),
                "weather": weather_map.get(user["weather"], user["weather"])
            }
            self.user_display_info.append(display_info)
        
        self.q_lines = {}
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for user_idx in range(len(dummy_users)):
            self.q_lines[user_idx] = {}
            for ad_idx in range(len(AD_CATEGORIES)):
                user_info = f"{self.user_display_info[user_idx]['age']}/{self.user_display_info[user_idx]['gender']}"
                label = f"User {user_idx+1} ({user_info}): {AD_CATEGORIES[ad_idx]}"
                line, = self.axs[1].plot([], [], colors[ad_idx % len(colors)], 
                                       label=label)
                self.q_lines[user_idx][ad_idx] = line
        
        self.axs[1].legend(loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.ion()  # Interactive mode on
        plt.show(block=False)
    
    def update_data(self, user_idx, action, reward, q_values):
        # Update reward history
        self.reward_history.append(reward)
        
        # Update Q-value history for this user
        for ad_idx in range(len(AD_CATEGORIES)):
            self.q_values_history[user_idx][ad_idx].append(q_values[ad_idx].item())
    
    def update_plot(self):
        # Update reward graph
        x = range(len(self.reward_history))
        self.reward_line.set_data(x, self.reward_history)
        self.axs[0].relim()
        self.axs[0].autoscale_view()
        
        # Update Q-value graph
        for user_idx in range(len(self.dummy_users)):
            for ad_idx in range(len(AD_CATEGORIES)):
                history = self.q_values_history[user_idx][ad_idx]
                if history:  # Only update if there's data
                    x = range(len(history))
                    self.q_lines[user_idx][ad_idx].set_data(x, history)
        
        self.axs[1].relim()
        self.axs[1].autoscale_view()
        
        # Refresh graphs
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

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
        
        # 학습 관련 지표를 저장할 리스트
        self.loss_history = []
        self.epsilon_history = []

    def select_action(self, state, initial_bias, training=True):
        """
        상태와 CSV로부터 산출된 초기 편향을 결합하여 행동(광고 카테고리)를 결정한다.
        epsilon-greedy 방식을 사용한다.
        """
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        
        # epsilon 값 히스토리에 추가
        self.epsilon_history.append(epsilon)

        if training and random.random() < epsilon:
            action = random.randrange(self.action_dim)
            return action, None  # Return None for Q-values when taking random action
        else:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state)
                # Add initial bias (CSV-based weights for each state)
                bias = torch.FloatTensor(initial_bias).to(self.device)
                q_values_with_bias = q_values + bias
                action = q_values_with_bias.argmax().item()
                return action, q_values_with_bias

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return 0  # Return 0 when loss is not calculated
            
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
        loss_value = loss.item()
        self.loss_history.append(loss_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss_value  # Return the calculated loss value

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_model(self, path):
        # Save model state
        model_state = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'loss_history': self.loss_history,
            'epsilon_history': self.epsilon_history
        }
        torch.save(model_state, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        # Load model state
        if os.path.exists(path):
            model_state = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(model_state['policy_net'])
            self.target_net.load_state_dict(model_state['target_net'])
            self.optimizer.load_state_dict(model_state['optimizer'])
            self.steps_done = model_state['steps_done']
            self.loss_history = model_state['loss_history']
            self.epsilon_history = model_state['epsilon_history']
            print(f"Model loaded from {path} (Steps: {self.steps_done})")
            return True
        else:
            print(f"Warning: {path} does not exist. Starting with a new model.")
            return False

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
    # 명령행 인자 처리
    parser = argparse.ArgumentParser(description="Ad Recommendation System")
    parser.add_argument('--continue_training', action='store_true', 
                        help='Load previously saved model and continue training')
    parser.add_argument('--save_interval', type=int, default=100, 
                        help='Steps interval to save the model (default: 100)')
    parser.add_argument('--visualization', action='store_true',
                        help='Enable visualization of the learning process')
    args = parser.parse_args()
    
    # 상태 벡터 차원은 16, 행동 수는 6이다.
    state_dim = 16
    action_dim = len(AD_CATEGORIES)
    agent = DQNAgent(state_dim, action_dim)
    env = AdEnv()

    # 모델 경로 설정
    model_path = os.path.join(MODEL_DIR, "dqn_model.pth")
    buffer_path = os.path.join(MODEL_DIR, "replay_buffer.pkl")
    
    # 이어서 학습할 경우 모델과 리플레이 버퍼 로드
    if args.continue_training:
        agent.load_model(model_path)
        agent.replay_buffer.load(buffer_path)
    
    batch_size = 32
    target_update_interval = 50  # 일정 스텝마다 target network 업데이트
    step_count = 0

    # 더미 사용자 데이터
    dummy_users = [
        {"age": "31-40세", "gender": "여성", "emotion": "happy", "time": "오후", "weather": "봄"},
        {"age": "20-30세", "gender": "남성", "emotion": "neutral", "time": "오전", "weather": "여름"},
        {"age": "41-50세", "gender": "여성", "emotion": "sad", "time": "오후", "weather": "가을"},
    ]
    
    # Original to English mapping for display
    original_to_eng_ad = {
        '교양_오락_문화': 'Culture_Entertainment', 
        '교육': 'Education', 
        '교통': 'Transportation', 
        '내구재': 'Durables', 
        '외식': 'Dining', 
        '의류': 'Clothing'
    }
    
    # 시각화 객체 초기화 (시각화 옵션이 활성화된 경우)
    visualizer = None
    if args.visualization:
        visualizer = Visualizer(agent, dummy_users)
    
    try:
        # 무한 반복하여 사용자 등장 시 광고 추천 및 학습을 진행한다.
        while True:
            # 실제 시스템에서는 새로운 사용자가 탐지되면, 해당 사용자의 state 정보를 추출한다.
            # 여기서는 dummy_users에서 순차적으로 가져온다.
            for user_idx, user in enumerate(dummy_users):
                # 상태 인코딩
                state_vector = encode_state(user)
                # CSV 파일 기반 초기 편향 계산
                initial_bias = get_initial_bias(user)
                
                # 환경 리셋: options 딕셔너리를 통해 상태 벡터 전달
                state, _ = env.reset(options={'state_vector': state_vector.numpy()})
                
                # 에이전트가 행동(광고 카테고리)를 선택한다.
                action, q_values = agent.select_action(state_vector, initial_bias, training=True)
                
                # Get the English name for display
                original_ad = next((k for k, v in original_to_eng_ad.items() if v == AD_CATEGORIES[action]), AD_CATEGORIES[action])
                recommended_ad = AD_CATEGORIES[action]
                
                print(f"Recommended Ad: {recommended_ad} / User Info: {user}")

                # 여기서 실시간으로 광고를 출력하고, 사용자 응시시간(gaze_time)을 측정한다.
                # 실제 시스템에서는 카메라 모듈과 연결하여 측정하며, 아래는 시뮬레이션 값이다.
                simulated_gaze_time = random.uniform(0.5, 5.0)  # 0.5초 ~ 5초 사이 응시했다고 가정
                reward = simulated_gaze_time  # 보상은 응시시간(초)
                print(f"Measured Gaze Time: {reward:.2f} sec")

                # 환경 step 수행 (에피소드 종료)
                next_state, r, done, truncated, info = env.step(action, reward)
                
                # 다음 상태는 새 사용자가 탐지될 때마다 reset하므로, 여기서는 next_state = state 그대로 사용
                agent.replay_buffer.push(state, action, reward, state, done)

                # 시각화 데이터 업데이트
                if visualizer and q_values is not None:
                    visualizer.update_data(user_idx, action, reward, q_values)
                
                # 에이전트 학습
                loss = agent.update(batch_size)
                step_count += 1
                
                # 일정 스텝마다 타깃 네트워크 업데이트
                if step_count % target_update_interval == 0:
                    agent.update_target()
                    print("Target network updated")
                
                # 일정 스텝마다 모델 저장
                if step_count % args.save_interval == 0:
                    agent.save_model(model_path)
                    agent.replay_buffer.save(buffer_path)
                
                # 시각화 업데이트
                if visualizer and step_count % 5 == 0:  # 5스텝마다 시각화 업데이트 (성능 고려)
                    visualizer.update_plot()

                # 추천 광고를 1초 안에 출력해야 하므로, 여기서는 sleep(1) 후 다음 사용자로 넘어간다고 가정한다.
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        agent.save_model(model_path)
        agent.replay_buffer.save(buffer_path)
        print("Program terminated.")
        
        # 종료 전 시각화 창 유지 (사용자가 닫을 때까지)
        if visualizer:
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    main()