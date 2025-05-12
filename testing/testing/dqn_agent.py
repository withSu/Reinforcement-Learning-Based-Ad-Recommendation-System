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
import json
import logging
from datetime import datetime
import argparse
from torch import serialization

# Add numpy.core.multiarray.scalar to safe globals for PyTorch 2.6+
try:
    import numpy.core.multiarray
    serialization.add_safe_globals(['scalar'], globs=numpy.core.multiarray.__dict__)
except Exception as e:
    pass  # If it fails, we'll handle this during load/save

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
    # 시뮬레이션을 위한 더미 데이터 생성
    logger.warning(f"CSV 파일 로드 실패, 더미 데이터 생성: {e}")
    # 연령 더미 데이터
    age_categories = ["20세미만", "20-30세", "31-40세", "41-50세", "51-60세"]
    ad_categories = ["교양_오락_문화", "교육", "교통", "내구재", "외식", "의류"]
    
    df_age = pd.DataFrame(columns=["연령"] + ad_categories)
    for age in age_categories:
        row = [age] + [random.uniform(0.3, 1.0) for _ in range(len(ad_categories))]
        df_age.loc[len(df_age)] = row
    
    # 성별 더미 데이터
    df_sex = pd.DataFrame(columns=["성별"] + ad_categories)
    for sex in ["남성", "여성"]:
        row = [sex] + [random.uniform(0.3, 1.0) for _ in range(len(ad_categories))]
        df_sex.loc[len(df_sex)] = row
    
    # 시간 더미 데이터
    df_time = pd.DataFrame(columns=["시간대"] + ad_categories)
    for time in ["오전", "오후"]:
        row = [time] + [random.uniform(0.3, 1.0) for _ in range(len(ad_categories))]
        df_time.loc[len(df_time)] = row
    
    # 계절 더미 데이터
    df_season = pd.DataFrame(columns=["계절"] + ad_categories)
    for season in ["봄", "여름", "가을", "겨울"]:
        row = [season] + [random.uniform(0.3, 1.0) for _ in range(len(ad_categories))]
        df_season.loc[len(df_season)] = row

# 광고 카테고리 목록 (영어와 한글 매핑)
AD_CATEGORIES_EN = ['Culture_Entertainment', 'Education', 'Transportation', 'Durables', 'Dining', 'Clothing']
AD_CATEGORIES_KO = ['교양_오락_문화', '교육', '교통', '내구재', '외식', '의류']
AD_CATEGORIES_MAP = dict(zip(AD_CATEGORIES_KO, AD_CATEGORIES_EN))
AD_CATEGORIES = AD_CATEGORIES_EN  # 시각화용 영어 카테고리

# 상태 인코딩을 위한 카테고리 정의
# 상태 인코딩을 위한 카테고리 정의 부분 수정
AGE_CATEGORIES = ["20세미만", "20-30세", "31-40세", "41-50세", "51-60세", "61-70세", "70세이상"]
AGE_MAP = {
    "20세미만": "under20", "20-30세": "20-30", "31-40세": "31-40", 
    "41-50세": "41-50", "51-60세": "51-60", "61-70세": "61-70", 
    "70세이상": "over70"
}

TIME_CATEGORIES = ["06-09시", "09-11시", "11-15시", "15-19시", "19-21시", "21-06시"]
TIME_MAP = {
    "06-09시": "morning_early", "09-11시": "morning_late", 
    "11-15시": "afternoon_early", "15-19시": "afternoon_late", 
    "19-21시": "evening", "21-06시": "night"
}
# 시뮬레이션용 더미 사용자 데이터
DUMMY_USERS = [
    {"age": "31-40세", "gender": "여성", "emotion": "happy", "time": "11-15시", "weather": "봄"},
    {"age": "20-30세", "gender": "남성", "emotion": "neutral", "time": "06-09시", "weather": "여름"},
    {"age": "41-50세", "gender": "여성", "emotion": "sad", "time": "15-19시", "weather": "가을"},
    {"age": "20세미만", "gender": "남성", "emotion": "happy", "time": "19-21시", "weather": "겨울"},
    {"age": "51-60세", "gender": "남성", "emotion": "neutral", "time": "21-06시", "weather": "봄"},
    {"age": "61-70세", "gender": "여성", "emotion": "sad", "time": "09-11시", "weather": "여름"},
    {"age": "70세이상", "gender": "남성", "emotion": "happy", "time": "11-15시", "weather": "가을"},
]
# CSV 파일 로드 부분 수정
try:
    df_age = pd.read_csv(os.path.join(DATA_DIR, "category_weight_age.csv"), 
                         header=0, names=["index", "연령"] + AD_CATEGORIES_KO)
    df_sex = pd.read_csv(os.path.join(DATA_DIR, "category_weight_sex.csv"), 
                         header=0, names=["index", "성별"] + AD_CATEGORIES_KO)
    df_time = pd.read_csv(os.path.join(DATA_DIR, "category_weight_time.csv"), 
                          header=0, names=["index", "시간대"] + AD_CATEGORIES_KO)
    df_season = pd.read_csv(os.path.join(DATA_DIR, "category_weight_season.csv"), 
                            header=0, names=["index", "계절"] + AD_CATEGORIES_KO)
    logger.info("CSV 파일 로드 성공")
except Exception as e:
    # 기존 시뮬레이션 코드 유지...
    logger.error(f"CSV 파일 로드 중 오류: {e}")
    # 더미 데이터 생성 코드...

# get_initial_bias 함수 수정
def get_initial_bias(state):
    """
    CSV 파일을 기반으로 사용자 특성에 맞는 초기 편향 벡터 생성
    
    Args:
        state (dict): 사용자 특성 딕셔너리 {'age', 'gender', 'emotion', 'time', 'weather'}
        
    Returns:
        numpy.ndarray: 광고 카테고리별 편향 벡터 (shape=(6,))
    """
    # 각 특성별 중요도 가중치 (연령과 성별이 시간이나 계절보다 중요)
    feature_weights = {"age": 0.35, "gender": 0.25, "time": 0.20, "weather": 0.20}
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

# camera_tracker.py에 있는 시간대 결정 로직 업데이트 (참고)
def get_time_period(current_hour):
    """현재 시간을 6개 시간대로 분류"""
    if 6 <= current_hour < 9:
        return "06-09시"
    elif 9 <= current_hour < 11:
        return "09-11시"
    elif 11 <= current_hour < 15:
        return "11-15시"
    elif 15 <= current_hour < 19:
        return "15-19시"
    elif 19 <= current_hour < 21:
        return "19-21시"
    else:  # 21-06시
        return "21-06시"
    
    

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
        self.q_values_history = {}    # 카테고리별 Q-값 히스토리 (시각화용)

        # 사용자 특성별 보상 추적 (시각화용)
        self.attribute_rewards = {
            'age': {}, 'gender': {}, 'emotion': {}, 'time': {}, 'weather': {}
        }

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
    
    def update_attribute_rewards(self, user_attributes, reward):
        """사용자 특성별 보상 추적 (시각화용)"""
        # 각 특성별 보상 추적
        age = user_attributes.get('age')
        if age:
            if age not in self.attribute_rewards['age']:
                self.attribute_rewards['age'][age] = []
            self.attribute_rewards['age'][age].append(reward)
        
        gender = user_attributes.get('gender')
        if gender:
            if gender not in self.attribute_rewards['gender']:
                self.attribute_rewards['gender'][gender] = []
            self.attribute_rewards['gender'][gender].append(reward)
            
        emotion = user_attributes.get('emotion')
        if emotion:
            if emotion not in self.attribute_rewards['emotion']:
                self.attribute_rewards['emotion'][emotion] = []
            self.attribute_rewards['emotion'][emotion].append(reward)
            
        time = user_attributes.get('time')
        if time:
            if time not in self.attribute_rewards['time']:
                self.attribute_rewards['time'][time] = []
            self.attribute_rewards['time'][time].append(reward)
            
        weather = user_attributes.get('weather')
        if weather:
            if weather not in self.attribute_rewards['weather']:
                self.attribute_rewards['weather'][weather] = []
            self.attribute_rewards['weather'][weather].append(reward)
    
    def update_q_values_history(self, user_type, q_values):
        """사용자 타입별 Q-값 히스토리 추적 (시각화용)"""
        if q_values is None:
            return
            
        # 사용자 타입이 키로 존재하지 않으면 초기화
        if user_type not in self.q_values_history:
            self.q_values_history[user_type] = {ad_idx: [] for ad_idx in range(self.action_dim)}
        
        # 각 광고 카테고리별 Q-값 추적
        q_values_np = q_values.cpu().numpy()
        for ad_idx in range(self.action_dim):
            self.q_values_history[user_type][ad_idx].append(q_values_np[ad_idx])
    
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
            'q_values_per_state': self.q_values_per_state,
            'q_values_history': self.q_values_history,
            'attribute_rewards': self.attribute_rewards
        }
        
        try:
            # PyTorch 2.6+ 대응: weights_only=False로 저장
            torch.save(model_state, path, weights_only=False)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            # 실패한 경우 예외 처리 방법 지정
            logger.error(f"Error saving model with weights_only=False: {e}")
            try:
                # 기본 저장 방식으로 시도
                torch.save(model_state, path)
                logger.info(f"Model saved to {path} with default settings")
            except Exception as fallback_e:
                logger.error(f"Error saving model with default settings: {fallback_e}")
    
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
                # PyTorch 2.6+ 대응: weights_only=False로 로드
                model_state = torch.load(path, map_location=self.device, weights_only=False)
                
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
                if 'q_values_history' in model_state:
                    self.q_values_history = model_state['q_values_history']
                if 'attribute_rewards' in model_state:
                    self.attribute_rewards = model_state['attribute_rewards']
                logger.info(f"Model loaded from {path} (Steps: {self.steps_done})")
                return True
            except Exception as e:
                logger.error(f"Error loading model with weights_only=False: {e}")
                try:
                    # 기본 로드 방식으로 시도
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
                    if 'q_values_history' in model_state:
                        self.q_values_history = model_state['q_values_history']
                    if 'attribute_rewards' in model_state:
                        self.attribute_rewards = model_state['attribute_rewards']
                    logger.info(f"Model loaded from {path} with default settings (Steps: {self.steps_done})")
                    return True
                except Exception as fallback_e:
                    logger.error(f"Error loading model with default settings: {fallback_e}")
                    return False
        else:
            logger.warning(f"Warning: {path} does not exist. Starting with a new model.")
            return False