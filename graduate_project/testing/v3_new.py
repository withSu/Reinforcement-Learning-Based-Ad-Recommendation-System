#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN 기반 광고 추천 시스템 (AdVise 프로젝트)

이 모듈은 사용자 특성(나이, 성별, 감정 등)을 기반으로 최적의 광고 카테고리를 추천하는
강화학습(DQN) 기반 시스템입니다. 사용자의 응시 시간(gaze time)을 보상으로 활용하여
지속적으로 학습하고 추천 정확도를 향상시킵니다.

Authors: [Your Names]
Version: 1.0.0
"""

import os
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
import pickle
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import json
import logging
from datetime import datetime

matplotlib.use('TkAgg')  # GUI 백엔드 설정

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ad_recommendation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdRecommendation")

# 디렉토리 설정
PROJECT_ROOT = "/home/a/A_2025/AdVise-ML/graduate_project"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
USER_DATA_PATH = os.path.join(DATA_DIR, "user_features.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# CSV 파일 로드 (파일 경로는 상황에 맞게 조정)
try:
    df_age = pd.read_csv(os.path.join(DATA_DIR, "category_weight_age.csv"))
    df_sex = pd.read_csv(os.path.join(DATA_DIR, "category_weight_sex.csv"))
    df_time = pd.read_csv(os.path.join(DATA_DIR, "category_weight_time.csv"))
    df_season = pd.read_csv(os.path.join(DATA_DIR, "category_weight_season.csv"))
    logger.info("CSV 파일 로드 성공")
except Exception as e:
    logger.error(f"CSV 파일 로드 중 오류: {e}")
    raise

# 광고 카테고리 목록 (영어와 한글 매핑)
AD_CATEGORIES_EN = ['Culture_Entertainment', 'Education', 'Transportation', 'Durables', 'Dining', 'Clothing']
AD_CATEGORIES_KO = ['교양_오락_문화', '교육', '교통', '내구재', '외식', '의류']
AD_CATEGORIES_MAP = dict(zip(AD_CATEGORIES_KO, AD_CATEGORIES_EN))
AD_CATEGORIES = AD_CATEGORIES_EN  # 시각화용 영어 카테고리

# 상태 인코딩을 위한 카테고리 정의
AGE_CATEGORIES = ["under20", "20-30", "31-40", "41-50", "51-60"]
AGE_MAP = {"20세미만": "under20", "20-30세": "20-30", "31-40세": "31-40", "41-50세": "41-50", "51-60세": "51-60"}

GENDER_CATEGORIES = ["male", "female"]
GENDER_MAP = {"남성": "male", "여성": "female"}

EMOTION_CATEGORIES = ["happy", "neutral", "sad"]

TIME_CATEGORIES = ["morning", "afternoon"]
TIME_MAP = {"오전": "morning", "오후": "afternoon"}

WEATHER_CATEGORIES = ["spring", "summer", "fall", "winter"]
WEATHER_MAP = {"봄": "spring", "여름": "summer", "가을": "fall", "겨울": "winter"}

def one_hot_encode(item, category_list):
    """
    카테고리 항목을 원-핫 인코딩 벡터로 변환
    
    Args:
        item: 카테고리 항목
        category_list: 카테고리 목록
        
    Returns:
        numpy.ndarray: 원-핫 인코딩 벡터
    """
    vec = np.zeros(len(category_list))
    if item in category_list:
        idx = category_list.index(item)
        vec[idx] = 1
    return vec

def encode_state(state, use_english=True):
    """
    상태 딕셔너리를 원-핫 인코딩된 벡터로 변환
    
    Args:
        state (dict): 사용자 특성 딕셔너리 {'age', 'gender', 'emotion', 'time', 'weather'}
        use_english (bool): 영어 카테고리 사용 여부
        
    Returns:
        torch.FloatTensor: 인코딩된 상태 벡터
    """
    # 한글 -> 영어 매핑 (영어 카테고리 사용 시)
    if use_english:
        age = AGE_MAP.get(state["age"], state["age"])
        gender = GENDER_MAP.get(state["gender"], state["gender"])
        emotion = state["emotion"]  # 이미 영어
        time = TIME_MAP.get(state["time"], state["time"])
        weather = WEATHER_MAP.get(state["weather"], state["weather"])
    else:
        age = state["age"]
        gender = state["gender"]
        emotion = state["emotion"]
        time = state["time"]
        weather = state["weather"]
    
    # 각 카테고리 원-핫 인코딩
    age_vec = one_hot_encode(age, AGE_CATEGORIES)
    gender_vec = one_hot_encode(gender, GENDER_CATEGORIES)
    emotion_vec = one_hot_encode(emotion, EMOTION_CATEGORIES)
    time_vec = one_hot_encode(time, TIME_CATEGORIES)
    weather_vec = one_hot_encode(weather, WEATHER_CATEGORIES)
    
    # 전체 상태 벡터 생성
    state_vector = np.concatenate([age_vec, gender_vec, emotion_vec, time_vec, weather_vec])
    return torch.FloatTensor(state_vector)

def get_initial_bias(state):
    """
    CSV 파일을 기반으로 사용자 특성에 맞는 초기 편향 벡터 생성
    
    사용자의 연령, 성별, 시간, 계절 특성을 바탕으로 각 CSV 파일에서 적절한 초기 편향 가중치를 추출하여
    가중 평균을 계산합니다. 이는 강화학습 모델에 사전 지식을 주입하는 역할을 합니다.
    
    Args:
        state (dict): 사용자 특성 딕셔너리
        
    Returns:
        numpy.ndarray: 광고 카테고리별 편향 벡터 (shape=(6,))
    """
    # 각 특성별 중요도 가중치 (연령과 성별이 시간이나 계절보다 중요)
    feature_weights = {"age": 0.4, "gender": 0.3, "time": 0.15, "weather": 0.15}
    weighted_vectors = []
    factors = []
    
    # 연령 편향
    row = df_age[df_age["연령"] == state["age"]]
    if not row.empty:
        vec = np.zeros(len(AD_CATEGORIES))
        for i, cat in enumerate(AD_CATEGORIES_KO):
            if cat in row.columns:
                vec[i] = row.iloc[0][cat]
        weighted_vectors.append(vec * feature_weights["age"])
        factors.append(feature_weights["age"])
    
    # 성별 편향
    row = df_sex[df_sex["성별"] == state["gender"]]
    if not row.empty:
        vec = np.zeros(len(AD_CATEGORIES))
        for i, cat in enumerate(AD_CATEGORIES_KO):
            if cat in row.columns:
                vec[i] = row.iloc[0][cat]
        weighted_vectors.append(vec * feature_weights["gender"])
        factors.append(feature_weights["gender"])
    
    # 시간 편향
    row = df_time[df_time["시간대"] == state["time"]]
    if not row.empty:
        vec = np.zeros(len(AD_CATEGORIES))
        for i, cat in enumerate(AD_CATEGORIES_KO):
            if cat in row.columns:
                vec[i] = row.iloc[0][cat]
        weighted_vectors.append(vec * feature_weights["time"])
        factors.append(feature_weights["time"])
    
    # 계절(날씨) 편향
    row = df_season[df_season["계절"] == state["weather"]]
    if not row.empty:
        vec = np.zeros(len(AD_CATEGORIES))
        for i, cat in enumerate(AD_CATEGORIES_KO):
            if cat in row.columns:
                vec[i] = row.iloc[0][cat]
        weighted_vectors.append(vec * feature_weights["weather"])
        factors.append(feature_weights["weather"])
    
    # 감정 상태에 따른 추가 편향 (CSV 없이 휴리스틱 규칙 적용)
    emotion = state.get("emotion", "neutral")
    emotion_bias = np.zeros(len(AD_CATEGORIES))
    
    if emotion == "happy":
        # 행복한 감정일 때는 문화/오락, 외식, 의류에 더 높은 가중치
        emotion_bias[0] = 0.15  # 교양_오락_문화
        emotion_bias[4] = 0.15  # 외식
        emotion_bias[5] = 0.1   # 의류
    elif emotion == "sad":
        # 슬픈 감정일 때는 교육, 내구재에 더 높은 가중치
        emotion_bias[1] = 0.15  # 교육
        emotion_bias[3] = 0.1   # 내구재
    
    # 평균 편향 계산 (가중 평균)
    if weighted_vectors:
        combined_bias = sum(weighted_vectors) / sum(factors)
        # 감정 편향 추가
        combined_bias += emotion_bias
        # 정규화 (0~1 범위로)
        combined_bias = combined_bias / max(1.0, np.max(combined_bias))
    else:
        combined_bias = np.zeros(len(AD_CATEGORIES))
    
    return combined_bias

# DQN 신경망 정의
class DQN(nn.Module):
    """
    Deep Q-Network 신경망 모델
    
    사용자 특성 상태를 입력으로 받아 각 광고 카테고리에 대한 Q-값을 출력하는 신경망입니다.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        """
        DQN 신경망 초기화
        
        Args:
            input_dim (int): 입력 차원 (상태 벡터 차원)
            output_dim (int): 출력 차원 (행동 수)
            hidden_dim (int): 은닉층 뉴런 수
        """
        super(DQN, self).__init__()
        # 신경망 레이어 구성
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # 가중치 초기화 (Kaiming/He 초기화)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        """
        순전파
        
        Args:
            x (torch.Tensor): 입력 텐서 (상태 벡터)
            
        Returns:
            torch.Tensor: 출력 텐서 (Q-값)
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay Buffer 정의
class ReplayBuffer:
    """
    경험 재생 버퍼
    
    강화학습에서 경험을 저장하고 샘플링하는 버퍼입니다.
    시간적 상관관계를 줄이고 학습 안정성을 높이는 역할을 합니다.
    """
    def __init__(self, capacity=10000):
        """
        ReplayBuffer 초기화
        
        Args:
            capacity (int): 버퍼 최대 용량
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        경험 추가
        
        Args:
            state: 현재 상태
            action: 선택한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        배치 샘플링
        
        Args:
            batch_size (int): 샘플 크기
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        """버퍼 길이 반환"""
        return len(self.buffer)
    
    def save(self, path):
        """
        리플레이 버퍼를 파일로 저장
        
        Args:
            path (str): 저장 경로
        """
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        logger.info(f"Replay buffer saved to {path}")
    
    def load(self, path):
        """
        파일에서 리플레이 버퍼 로드
        
        Args:
            path (str): 로드할 파일 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    loaded_buffer = pickle.load(f)
                    self.buffer.clear()
                    for experience in loaded_buffer:
                        self.buffer.append(experience)
                logger.info(f"Loaded {len(self.buffer)} experiences from {path}")
                return True
            except Exception as e:
                logger.error(f"Error loading replay buffer: {e}")
                return False
        else:
            logger.warning(f"Warning: {path} does not exist. Starting with a new buffer.")
            return False

# DQN Agent 정의
class DQNAgent:
    """
    DQN 에이전트
    
    Deep Q-Network 알고리즘에 기반한 강화학습 에이전트입니다.
    Double DQN, 입실론-탐욕 정책 등을 구현하여 안정적인 학습을 제공합니다.
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500):
        """
        DQNAgent 초기화
        
        Args:
            state_dim (int): 상태 공간 차원
            action_dim (int): 행동 공간 차원
            lr (float): 학습률
            gamma (float): 할인율
            epsilon_start (float): 초기 입실론 값
            epsilon_final (float): 최종 입실론 값
            epsilon_decay (float): 입실론 감소율
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # 입실론 관련
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 신경망 초기화
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 옵티마이저 설정
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
        # 학습 지표 추적
        self.loss_history = []
        self.epsilon_history = []
        self.reward_history = []
        self.q_values_per_state = {}  # 상태별 Q-값 추적

    def select_action(self, state, initial_bias, training=True):
        """
        입실론-탐욕 정책에 따라 행동 선택
        
        Args:
            state (torch.Tensor): 상태 벡터
            initial_bias (numpy.ndarray): 초기 편향 벡터
            training (bool): 학습 모드 여부
            
        Returns:
            tuple: (선택된 행동 인덱스, Q-값 벡터)
        """
        # 입실론 값 계산 (지수적 감소)
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        self.epsilon_history.append(epsilon)
        
        # 탐색(exploration) - 랜덤 행동
        if training and random.random() < epsilon:
            action = random.randrange(self.action_dim)
            return action, None
        # 활용(exploitation) - 최적 행동
        else:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state)
                
                # 초기 편향 적용
                bias = torch.FloatTensor(initial_bias).to(self.device)
                q_values_with_bias = q_values + bias
                
                # 최대 Q-값을 가진 행동 선택
                action = q_values_with_bias.argmax().item()
                
                # 상태 키 생성 (추적용)
                state_key = tuple(state.cpu().numpy().round(3))
                if state_key not in self.q_values_per_state:
                    self.q_values_per_state[state_key] = []
                self.q_values_per_state[state_key].append(q_values.cpu().numpy())
                
                return action, q_values_with_bias

    def update(self, batch_size):
        """
        배치 데이터로 신경망 업데이트 (Double DQN 알고리즘)
        
        Args:
            batch_size (int): 배치 크기
            
        Returns:
            float: 손실값
        """
        if len(self.replay_buffer) < batch_size:
            return 0
        
        # 리플레이 버퍼에서 배치 샘플 추출
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)
        
        # 텐서 변환 및 디바이스 이동
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        
        # 현재 Q-값 계산
        q_values = self.policy_net(state_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # 타겟 Q-값 계산 (Double DQN 방식)
        with torch.no_grad():
            # 정책 네트워크로 다음 행동 선택
            next_action = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
            # 타겟 네트워크로 다음 상태의 Q-값 평가
            next_q_values = self.target_net(next_state_batch)
            next_q_value = next_q_values.gather(1, next_action).squeeze(1)
            
            # 타겟 계산: 보상 + 감가율 * 다음 상태 최대 Q-값
            expected_q_value = reward_batch + self.gamma * next_q_value * (1 - done_batch)
        
        # 손실 계산 및 역전파
        loss = nn.MSELoss()(q_value, expected_q_value)
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # 그래디언트 클리핑 (학습 안정화)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        return loss_value

    def update_target(self):
        """타겟 네트워크를 정책 네트워크로 업데이트"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        logger.debug("타겟 네트워크 업데이트")
    
    def add_reward(self, reward):
        """보상 히스토리에 추가"""
        self.reward_history.append(reward)
    
    def save_model(self, path):
        """
        모델 상태 저장
        
        Args:
            path (str): 저장 경로
        """
        model_state = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'loss_history': self.loss_history,
            'epsilon_history': self.epsilon_history,
            'reward_history': self.reward_history,
            'q_values_per_state': self.q_values_per_state
        }
        torch.save(model_state, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        모델 상태 로드
        
        Args:
            path (str): 로드할 파일 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        if os.path.exists(path):
            try:
                model_state = torch.load(path, map_location=self.device)
                self.policy_net.load_state_dict(model_state['policy_net'])
                self.target_net.load_state_dict(model_state['target_net'])
                self.optimizer.load_state_dict(model_state['optimizer'])
                self.steps_done = model_state['steps_done']
                self.loss_history = model_state['loss_history']
                self.epsilon_history = model_state['epsilon_history']
                if 'reward_history' in model_state:
                    self.reward_history = model_state['reward_history']
                if 'q_values_per_state' in model_state:
                    self.q_values_per_state = model_state['q_values_per_state']
                logger.info(f"Model loaded from {path} (Steps: {self.steps_done})")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
        else:
            logger.warning(f"Warning: {path} does not exist. Starting with a new model.")
            return False

# 환경 클래스 정의
class AdEnv(gym.Env):
    """
    광고 추천 환경
    
    이 환경은 사용자에게 광고를 추천하고 응시 시간에 따른 보상을 제공합니다.
    OpenAI Gym 인터페이스를 따라 구현되었습니다.
    """
    def __init__(self):
        """AdEnv 초기화"""
        super(AdEnv, self).__init__()
        # 상태 공간: 16차원 (5 + 2 + 3 + 2 + 4 = 16)
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32)
        # 행동 공간: 6개 광고 카테고리
        self.action_space = spaces.Discrete(len(AD_CATEGORIES))
        self.current_state = None
        self.episode_count = 0

    def reset(self, seed=None, options=None):
        """
        환경 초기화
        
        Args:
            seed: 랜덤 시드
            options: 추가 옵션 (state_vector를 포함할 수 있음)
            
        Returns:
            tuple: (상태, 추가 정보)
        """
        super().reset(seed=seed)
        
        if options is not None and 'state_vector' in options:
            self.current_state = options['state_vector']
        else:
            self.current_state = np.zeros(16, dtype=np.float32)
        
        self.episode_count += 1
        return self.current_state, {"episode": self.episode_count}

    def step(self, action, reward):
        """
        환경에서 한 스텝 진행
        
        Args:
            action (int): 선택된 광고 카테고리 인덱스
            reward (float): 응시 시간으로 측정된 보상
            
        Returns:
            tuple: (상태, 보상, 종료 여부, 잘림 여부, 정보)
        """
        done = True  # 한 사용자당 한 에피소드로 취급
        info = {
            "recommended_ad": AD_CATEGORIES[action],
            "recommended_ad_ko": AD_CATEGORIES_KO[action],
            "episode": self.episode_count
        }
        return self.current_state, reward, done, False, info

# 시각화 클래스 정의
class Visualizer:
    """
    학습 과정 시각화
    
    DQN 학습 과정과 결과를 실시간으로 시각화하는 클래스입니다.
    """
    def __init__(self, agent, user_attributes_list):
        """
        Visualizer 초기화
        
        Args:
            agent (DQNAgent): DQN 에이전트
            user_attributes_list (list): 사용자 특성 목록
        """
        self.agent = agent
        self.user_attributes_list = user_attributes_list
        
        # 사용자 영어 표시 정보 변환
        self.user_display_info = []
        for user in user_attributes_list:
            display_info = {
                "age": AGE_MAP.get(user["age"], user["age"]),
                "gender": GENDER_MAP.get(user["gender"], user["gender"]),
                "emotion": user["emotion"],
                "time": TIME_MAP.get(user["time"], user["time"]),
                "weather": WEATHER_MAP.get(user["weather"], user["weather"])
            }
            self.user_display_info.append(display_info)
        
        # Matplotlib 설정
        plt.style.use('ggplot')  # 보기 좋은 스타일 적용
        self.fig = plt.figure(figsize=(14, 18))
        
        # 서브플롯 설정
        self.setup_subplots()
        
        # 데이터 추적
        self.reward_history = []
        self.step_count = 0
        self.q_values_history = {user_idx: {ad_idx: [] for ad_idx in range(len(AD_CATEGORIES))} 
                                for user_idx in range(len(user_attributes_list))}
        self.attribute_rewards = {}  # 속성별 보상 평균
        
        # 그래프 선 초기화
        self.initialize_plot_lines()
        
        # 플롯 설정
        plt.tight_layout()
        plt.ion()  # 대화형 모드 활성화
        plt.show(block=False)
    
    
    def update_plot(self):
        """그래프 업데이트"""
        # 보상 그래프 업데이트
        x = range(len(self.reward_history))
        self.reward_line.set_data(x, self.reward_history)
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # 이동 평균 추가
        if len(self.reward_history) > 10:
            window_size = min(10, len(self.reward_history) - 1)
            moving_avg = np.convolve(self.reward_history, np.ones(window_size)/window_size, mode='valid')
            if hasattr(self, 'avg_line'):
                self.avg_line.remove()
            self.avg_line, = self.ax1.plot(range(window_size-1, len(self.reward_history)), 
                                        moving_avg, 'r-', linewidth=2, label='Moving Avg (10)')
            self.ax1.legend(loc='upper right')
        
        # Q-값 그래프 업데이트
        for user_idx in range(len(self.user_attributes_list)):
            for ad_idx in range(len(AD_CATEGORIES)):
                history = self.q_values_history[user_idx][ad_idx]
                if history:
                    x = range(len(history))
                    self.q_lines[user_idx][ad_idx].set_data(x, history)
        
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # 속성별 보상 그래프 업데이트
        if self.attribute_rewards:
            # 기존 막대 제거
            if self.bars is not None:
                for bar in self.bars:
                    bar.remove()
            
            # 속성별 평균 보상 계산
            attr_names = []
            attr_avgs = []
            
            for attr, rewards in sorted(self.attribute_rewards.items()):
                if rewards:  # 보상이 있는 경우만
                    attr_names.append(attr)
                    attr_avgs.append(np.mean(rewards))
            
            # 막대 그래프 그리기
            colors = []
            for attr in attr_names:
                if attr.startswith('Age'):
                    colors.append('#E41A1C')  # 빨강
                elif attr.startswith('Gender'):
                    colors.append('#377EB8')  # 파랑
                elif attr.startswith('Emotion'):
                    colors.append('#4DAF4A')  # 초록
                elif attr.startswith('Time'):
                    colors.append('#984EA3')  # 보라
                elif attr.startswith('Weather'):
                    colors.append('#FF7F00')  # 주황
            
            self.bars = self.ax3.bar(attr_names, attr_avgs, color=colors)
            
            # 가독성을 위한 설정
            self.ax3.set_xticklabels(attr_names, rotation=45, ha='right')
            self.ax3.set_ylim([0, max(attr_avgs) * 1.1])  # 최댓값의 1.1배로 y축 설정
            
            # 값 레이블 추가
            for bar, avg in zip(self.bars, attr_avgs):
                height = bar.get_height()
                self.ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{avg:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 손실 및 입실론 그래프 업데이트
        if hasattr(self.agent, 'loss_history') and self.agent.loss_history:
            x = range(len(self.agent.loss_history))
            self.loss_line.set_data(x, self.agent.loss_history)
        
        if hasattr(self.agent, 'epsilon_history') and self.agent.epsilon_history:
            x = range(len(self.agent.epsilon_history))
            self.epsilon_line.set_data(x, self.agent.epsilon_history)
        
        self.ax4.relim()
        self.ax4.autoscale_view()
        
        # 그래프 갱신
        plt.draw()
        plt.pause(0.01)  # 짧은 일시 정지로 UI 갱신 보장
    
    
    def setup_subplots(self):
        """서브플롯 영역 설정"""
        # 메인 제목
        self.fig.suptitle('AdVise: Real-time Ad Recommendation System Analytics', fontsize=16)
        
        # 서브플롯 1: 보상 히스토리
        self.ax1 = self.fig.add_subplot(4, 1, 1)
        self.ax1.set_title('Reward History (User Gaze Time)')
        self.ax1.set_xlabel('Step (count)')
        self.ax1.set_ylabel('Gaze Time (seconds)')
        
        # 서브플롯 2: Q-값 변화
        self.ax2 = self.fig.add_subplot(4, 1, 2)
        self.ax2.set_title('Q-Values by User Type')
        self.ax2.set_xlabel('Step (count)')
        self.ax2.set_ylabel('Q-Value (expected reward)')
        
        # 서브플롯 3: 속성별 평균 응시 시간
        self.ax3 = self.fig.add_subplot(4, 1, 3)
        self.ax3.set_title('Average Gaze Time by User Attributes')
        self.ax3.set_xlabel('User Attributes')
        self.ax3.set_ylabel('Average Gaze Time (seconds)')
        
        # 서브플롯 4: 손실 및 입실론 변화
        self.ax4 = self.fig.add_subplot(4, 1, 4)
        self.ax4.set_title('Training Metrics')
        self.ax4.set_xlabel('Step (count)')
        self.ax4.set_ylabel('Loss / Epsilon')
    
    def initialize_plot_lines(self):
        """플롯 선 초기화"""
        # 보상 히스토리 선
        self.reward_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Gaze Time')
        
        # Q-값 선
        self.q_lines = {}
        colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33']
        for user_idx in range(len(self.user_attributes_list)):
            self.q_lines[user_idx] = {}
            for ad_idx in range(len(AD_CATEGORIES)):
                user_info = f"{self.user_display_info[user_idx]['age']}/{self.user_display_info[user_idx]['gender']}"
                label = f"User {user_idx+1} ({user_info}): {AD_CATEGORIES[ad_idx]}"
                line, = self.ax2.plot([], [], color=colors[ad_idx % len(colors)], 
                                      label=label, linewidth=1.5)
                self.q_lines[user_idx][ad_idx] = line
        
        # 범례 설정
        self.ax2.legend(loc='upper left', fontsize='x-small', ncol=2)
        
        # 속성별 보상 바 그래프 (처음에는 비어있음)
        self.bars = None
        
        # 손실 및 입실론 선
        self.loss_line, = self.ax4.plot([], [], 'r-', linewidth=2, label='Loss')
        self.epsilon_line, = self.ax4.plot([], [], 'g-', linewidth=2, label='Epsilon')
        self.ax4.legend(loc='upper right')
        
        # 오른쪽 y축 (입실론용)
        self.ax4_twin = self.ax4.twinx()
        self.ax4_twin.set_ylabel('Epsilon')
        self.ax4_twin.set_ylim([0, 1])
    
    def update_data(self, user_idx, action, reward, q_values=None):
        """
        새 데이터 업데이트
        
        Args:
            user_idx (int): 사용자 인덱스
            action (int): 선택된 행동 (광고 카테고리)
            reward (float): 보상 (응시 시간)
            q_values (torch.Tensor): Q-값 벡터
        """
        self.step_count += 1
        
        # 보상 추적
        self.reward_history.append(reward)
        
        # Q-값 추적 (q_values가 제공된 경우)
        if q_values is not None:
            for ad_idx in range(len(AD_CATEGORIES)):
                self.q_values_history[user_idx][ad_idx].append(q_values[ad_idx].item())
        
        # 속성별 보상 추적
        user = self.user_attributes_list[user_idx]
        
        # 연령 보상 추적
        age_key = f"Age: {user['age']}"
        if age_key not in self.attribute_rewards:
            self.attribute_rewards[age_key] = []
        self.attribute_rewards[age_key].append(reward)
        
        # 성별 보상 추적
        gender_key = f"Gender: {user['gender']}"
        if gender_key not in self.attribute_rewards:
            self.attribute_rewards[gender_key] = []
        self.attribute_rewards[gender_key].append(reward)
        
        # 감정 보상 추적
        emotion_key = f"Emotion: {user['emotion']}"
        if emotion_key not in self.attribute_rewards:
            self.attribute_rewards[emotion_key] = []
        self.attribute_rewards[emotion_key].append(reward)
        
        # 시간 보상 추적
        time_key = f"Time: {user['time']}"
        if time_key not in self.attribute_rewards:
            self.attribute_rewards[time_key] = []
        self.attribute_rewards[time_key].append(reward)
        
        # 날씨 보상 추적
        weather_key = f"Weather: {user['weather']}"
        if weather_key not in self.attribute_rewards:
            self.attribute_rewards[weather_key] = []
        self.attribute_rewards[weather_key].append(reward)

    


def load_user_features():
    """
    카메라 추적 시스템에서 저장한 사용자 특성 데이터 로드
    
    Returns:
        list: 사용자 특성 데이터 목록
    """
    if os.path.exists(USER_DATA_PATH):
        try:
            with open(USER_DATA_PATH, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            logger.info(f"{len(user_data)}개의 사용자 특성 데이터를 로드했습니다.")
            return user_data
        except Exception as e:
            logger.error(f"사용자 특성 데이터 로드 중 오류: {e}")
            return []
    else:
        logger.warning(f"경고: {USER_DATA_PATH}가 존재하지 않습니다. 시뮬레이션 모드로 실행됩니다.")
        return []

def main():
    """광고 추천 시스템 메인 함수"""
    # 명령행 인자 처리
    parser = argparse.ArgumentParser(description="DQN 기반 광고 추천 시스템")
    parser.add_argument('--continue_training', action='store_true', 
                        help='이전에 저장된 모델을 로드하여 학습을 이어서 진행')
    parser.add_argument('--save_interval', type=int, default=20, 
                        help='모델을 저장할 스텝 간격 (기본값: 20)')
    parser.add_argument('--visualization', action='store_true',
                        help='학습 과정의 시각화 활성화')
    parser.add_argument('--simulation', action='store_true',
                        help='시뮬레이션 모드 (camera_tracker에서 데이터를 사용하지 않음)')
    parser.add_argument('--interval', type=float, default=5.0,
                        help='사용자 데이터 확인 간격 (초, 기본값: 5)')
    parser.add_argument('--debug', action='store_true',
                        help='디버그 모드 활성화')
    args = parser.parse_args()
    
    # 디버그 모드일 경우 로깅 레벨 변경
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("디버그 모드 활성화")
    
    # 시뮬레이션 모드를 위한 더미 사용자 데이터
    dummy_users = [
        {"age": "31-40세", "gender": "여성", "emotion": "happy", "time": "오후", "weather": "봄"},
        {"age": "20-30세", "gender": "남성", "emotion": "neutral", "time": "오전", "weather": "여름"},
        {"age": "41-50세", "gender": "여성", "emotion": "sad", "time": "오후", "weather": "가을"},
    ]
    
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
    
    # 학습 파라미터
    batch_size = 32
    target_update_interval = 50  # 타겟 네트워크 업데이트 간격
    step_count = 0
    
    # 시각화 객체 초기화 (시각화 옵션이 활성화된 경우)
    visualizer = None
    if args.visualization:
        visualizer = Visualizer(agent, dummy_users)
    
    # 마지막 데이터 로드 시간
    last_data_load_time = 0
    # 처리된 데이터의 타임스탬프 추적
    processed_timestamps = set()
    
    # 모델 평가 지표
    total_rewards = 0
    episode_count = 0
    
    try:
        logger.info("광고 추천 시스템이 시작되었습니다.")
        
        while True:
            current_time = time.time()
            user_list = []
            
            # 시뮬레이션 모드 또는 일정 간격으로 데이터 로드
            if args.simulation:
                # 시뮬레이션 모드: 더미 사용자 데이터 사용
                user_list = dummy_users
            elif current_time - last_data_load_time >= args.interval:
                # 카메라 추적 시스템에서 저장한 사용자 특성 데이터 로드
                user_data = load_user_features()
                
                # 새로운 데이터만 필터링
                new_users = []
                for user in user_data:
                    if 'timestamp' in user and user['timestamp'] not in processed_timestamps:
                        processed_timestamps.add(user['timestamp'])
                        # 'id'와 'timestamp' 키는 학습에 불필요
                        user_copy = user.copy()
                        if 'id' in user_copy:
                            del user_copy['id']
                        if 'timestamp' in user_copy:
                            del user_copy['timestamp']
                        new_users.append(user_copy)
                
                user_list = new_users
                last_data_load_time = current_time
                
                if new_users:
                    logger.info(f"{len(new_users)}개의 새로운 사용자 데이터를 처리합니다.")
                
            # 각 사용자에 대해 광고 추천 수행
            for user_idx, user in enumerate(user_list):
                # 응시 시간이 없으면 기본값 사용
                if 'gaze_time' not in user:
                    user['gaze_time'] = random.uniform(0.5, 5.0)
                
                # 더미 사용자 매핑 (시각화용)
                dummy_idx = 0
                if user["age"] == "20-30세" and user["gender"] == "남성":
                    dummy_idx = 1
                elif user["age"] == "41-50세" and user["gender"] == "여성":
                    dummy_idx = 2
                
                # 상태 인코딩
                state_vector = encode_state(user)
                # CSV 파일 기반 초기 편향 계산
                initial_bias = get_initial_bias(user)
                
                # 환경 리셋
                state, _ = env.reset(options={'state_vector': state_vector.numpy()})
                
                # 에이전트가 행동(광고 카테고리) 선택
                action, q_values = agent.select_action(state_vector, initial_bias, training=True)
                recommended_ad = AD_CATEGORIES[action]
                original_ad = AD_CATEGORIES_KO[action]
                
                # 추천된 광고와 사용자 정보 출력
                logger.info(f"추천 광고: {recommended_ad} ({original_ad}) / 사용자 정보: {user}")
                
                # 응시 시간을 보상으로 사용
                reward = user['gaze_time']
                agent.add_reward(reward)
                logger.info(f"응시 시간: {reward:.2f} sec")
                
                # 통계 업데이트
                total_rewards += reward
                episode_count += 1
                
                # 환경 스텝 수행
                next_state, r, done, truncated, info = env.step(action, reward)
                
                # 리플레이 버퍼에 경험 저장
                agent.replay_buffer.push(state, action, reward, state, done)
                
                # 시각화 업데이트 - 이 부분을 수정
                if visualizer:  # 매 프레임마다 업데이트 (매 사용자 처리 후)
                    # 시각화 창이 닫혔으면 다시 생성
                    if not plt.fignum_exists(visualizer.fig.number):
                        visualizer = Visualizer(agent, dummy_users)
                    visualizer.update_plot()
                
                # 에이전트 학습
                loss = agent.update(batch_size)
                step_count += 1
                
                # 타겟 네트워크 업데이트
                if step_count % target_update_interval == 0:
                    agent.update_target()
                    logger.info(f"타겟 네트워크 업데이트 (스텝: {step_count})")
                
                # 모델 저장
                if step_count % args.save_interval == 0:
                    agent.save_model(model_path)
                    agent.replay_buffer.save(buffer_path)
                    
                    # 학습 진행 상황 출력
                    avg_reward = total_rewards / max(1, episode_count)
                    logger.info(f"스텝: {step_count}, 에피소드: {episode_count}, 평균 보상: {avg_reward:.2f}")
                    # 통계 초기화
                    total_rewards = 0
                    episode_count = 0
            
            # 시각화 업데이트
            if visualizer and step_count % 5 == 0 and user_list:
                visualizer.update_plot()
            
            # 시뮬레이션 모드에서는 지연 추가
            if args.simulation:
                time.sleep(1)
            else:
                # 약간의 지연만 추가 (CPU 부하 감소)
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("\n학습이 중단되었습니다. 모델을 저장합니다...")
        agent.save_model(model_path)
        agent.replay_buffer.save(buffer_path)
        logger.info("프로그램을 종료합니다.")
    
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
    
    finally:
        # 종료 전 시각화 창 유지 (사용자가 닫을 때까지)
        if visualizer:
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    main()