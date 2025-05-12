#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 DQN 기반 광고 추천 시스템 (AdVise 프로젝트)

이 모듈은 사용자 특성(나이, 성별, 감정 등)을 기반으로 최적의 광고 카테고리를 추천하는
강화학습(DQN) 기반 시스템입니다. 사용자의 응시 시간(gaze time)을 보상으로 활용하여
지속적으로 학습하고 추천 정확도를 향상시킵니다.

개선된 부분:
1. 시각화 기능 강화 - 실시간 학습 상태 및 추천 결과 시각화
2. 사용자 특성별 보상 추적 개선
3. 초기 가중치 설정 로직 최적화
4. 사용자 인터페이스 추가
"""

import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, Counter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading
import json
import logging
from datetime import datetime

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
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리
DATA_DIR = os.path.join(PROJECT_ROOT, "data")  # 이건 그대로 둠
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 파일 경로 설정
USER_DATA_PATH = os.path.join(DATA_DIR, "user_features.json")
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_model.pth")
BUFFER_PATH = os.path.join(MODEL_DIR, "replay_buffer.pkl")

# 광고 카테고리 목록 (영어와 한글 매핑)
AD_CATEGORIES_EN = ['Culture_Entertainment', 'Education', 'Transportation', 'Durables', 'Dining', 'Clothing']
AD_CATEGORIES_KO = ['교양_오락_문화', '교육', '교통', '내구재', '외식', '의류']
AD_CATEGORIES_MAP = dict(zip(AD_CATEGORIES_KO, AD_CATEGORIES_EN))
AD_CATEGORIES = AD_CATEGORIES_EN  # 시각화용 영어 카테고리

# 상태 인코딩을 위한 카테고리 정의
AGE_CATEGORIES = ["20세미만", "20-30세", "31-40세", "41-50세", "51-60세", "61-70세", "70세이상"]
GENDER_CATEGORIES = ["남성", "여성"]
EMOTION_CATEGORIES = ["happy", "neutral", "sad", "angry", "surprise", "fear", "disgust"]
TIME_CATEGORIES = ["오전", "오후"]
WEATHER_CATEGORIES = ["봄", "여름", "가을", "겨울"]

# 광고 이미지 및 내용 하드코딩 (실제 애플리케이션에서는 데이터베이스나 파일에서 로드)
SAMPLE_ADS = {
    'Culture_Entertainment': {
        'title': '신규 영화 개봉 안내',
        'content': '화제의 영화 지금 상영중!',
        'target': '20-40대 남녀',
        'color': '#FF9999'  # 연한 빨강
    },
    'Education': {
        'title': '온라인 강의 할인 이벤트',
        'content': '자기계발의 기회, 30% 할인',
        'target': '20-30대 취준생',
        'color': '#99CCFF'  # 연한 파랑
    },
    'Transportation': {
        'title': '신규 전기차 출시',
        'content': '친환경 미래형 자동차',
        'target': '30-50대 남성',
        'color': '#99FF99'  # 연한 초록
    },
    'Durables': {
        'title': '가전제품 연말 세일',
        'content': '최대 50% 할인행사 진행중',
        'target': '30-50대 여성',
        'color': '#FFCC99'  # 연한 주황
    },
    'Dining': {
        'title': '건강한 식단 배달 서비스',
        'content': '든든한 한 끼, 첫 주문 무료',
        'target': '20-40대 직장인',
        'color': '#CC99FF'  # 연한 보라
    },
    'Clothing': {
        'title': '시즌 아웃렛 대개방',
        'content': '유명 브랜드 최대 70% 할인',
        'target': '전 연령층',
        'color': '#FFFF99'  # 연한 노랑
    }
}

# CSV 파일 로드 함수
def load_csv_data():
    """CSV 파일을 로드하고 초기화합니다"""
    try:
        df_age = pd.read_csv(os.path.join(DATA_DIR, "category_weight_age.csv"))
        df_sex = pd.read_csv(os.path.join(DATA_DIR, "category_weight_sex.csv"))
        df_time = pd.read_csv(os.path.join(DATA_DIR, "category_weight_time.csv"))
        df_season = pd.read_csv(os.path.join(DATA_DIR, "category_weight_season.csv"))
        logger.info("CSV 파일 로드 성공")
        return df_age, df_sex, df_time, df_season
    except Exception as e:
        logger.warning(f"CSV 파일 로드 실패, 더미 데이터 생성: {e}")
        # 더미 데이터 생성
        age_categories = AGE_CATEGORIES
        ad_categories = AD_CATEGORIES_KO
        
        df_age = pd.DataFrame(columns=["연령"] + ad_categories)
        for age in age_categories:
            row = [age] + [random.uniform(0.3, 1.0) for _ in range(len(ad_categories))]
            df_age.loc[len(df_age)] = row
        
        df_sex = pd.DataFrame(columns=["성별"] + ad_categories)
        for sex in GENDER_CATEGORIES:
            row = [sex] + [random.uniform(0.3, 1.0) for _ in range(len(ad_categories))]
            df_sex.loc[len(df_sex)] = row
        
        df_time = pd.DataFrame(columns=["시간대"] + ad_categories)
        for time in TIME_CATEGORIES:
            row = [time] + [random.uniform(0.3, 1.0) for _ in range(len(ad_categories))]
            df_time.loc[len(df_time)] = row
        
        df_season = pd.DataFrame(columns=["계절"] + ad_categories)
        for season in WEATHER_CATEGORIES:
            row = [season] + [random.uniform(0.3, 1.0) for _ in range(len(ad_categories))]
            df_season.loc[len(df_season)] = row
        
        # CSV 파일 저장 (나중에 사용하기 위해)
        df_age.to_csv(os.path.join(DATA_DIR, "category_weight_age.csv"), index=False)
        df_sex.to_csv(os.path.join(DATA_DIR, "category_weight_sex.csv"), index=False)
        df_time.to_csv(os.path.join(DATA_DIR, "category_weight_time.csv"), index=False)
        df_season.to_csv(os.path.join(DATA_DIR, "category_weight_season.csv"), index=False)
        
        logger.info("더미 CSV 데이터 생성 및 저장 완료")
        return df_age, df_sex, df_time, df_season

# CSV 데이터 로드
df_age, df_sex, df_time, df_season = load_csv_data()

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

def encode_state(state):
    """
    상태 딕셔너리를 원-핫 인코딩된 벡터로 변환
    
    Args:
        state (dict): 사용자 특성 딕셔너리 {'age', 'gender', 'emotion', 'time', 'weather'}
        
    Returns:
        torch.FloatTensor: 인코딩된 상태 벡터
    """
    # 각 카테고리 원-핫 인코딩
    age_vec = one_hot_encode(state["age"], AGE_CATEGORIES)
    gender_vec = one_hot_encode(state["gender"], GENDER_CATEGORIES)
    emotion_vec = one_hot_encode(state["emotion"], EMOTION_CATEGORIES)
    time_vec = one_hot_encode(state["time"], TIME_CATEGORIES)
    weather_vec = one_hot_encode(state["weather"], WEATHER_CATEGORIES)
    
    # 전체 상태 벡터 생성
    state_vector = np.concatenate([age_vec, gender_vec, emotion_vec, time_vec, weather_vec])
    return torch.FloatTensor(state_vector)

def get_initial_bias(state):
    """
    CSV 파일을 기반으로 사용자 특성에 맞는 초기 편향 벡터 생성
    
    Args:
        state (dict): 사용자 특성 딕셔너리
        
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
    elif emotion == "angry":
        # 화난 감정일 때는 교통, 내구재에 더 높은 가중치
        emotion_bias[2] = 0.15  # 교통
        emotion_bias[3] = 0.1   # 내구재
    elif emotion == "surprise":
        # 놀란 감정일 때는 문화/오락, 교육에 더 높은 가중치
        emotion_bias[0] = 0.15  # 교양_오락_문화
        emotion_bias[1] = 0.1   # 교육
    
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
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        """버퍼 길이 반환"""
        return len(self.buffer)
    
    def save(self, path):
        """
        리플레이 버퍼를 파일로 저장
        
        Args:
            path (str): 저장 경로
        """
        import pickle
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
        import pickle
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
        self.q_values_history = {}
        
        # 사용자 특성별 보상 추적
        self.attribute_rewards = {
            'age': {}, 'gender': {}, 'emotion': {}, 'time': {}, 'weather': {}
        }
        
        # 추천 결과 추적
        self.recommendations = []

    def select_action(self, state, initial_bias=None, training=True):
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
                
                # 초기 편향이 제공된 경우 적용
                if initial_bias is not None:
                    bias = torch.FloatTensor(initial_bias).to(self.device)
                    q_values_with_bias = q_values + bias
                else:
                    q_values_with_bias = q_values
                
                # 최대 Q-값을 가진 행동 선택
                action = q_values_with_bias.argmax().item()
                
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
    
    def update_attribute_rewards(self, user_attributes, reward, action):
        """사용자 특성별 보상 추적"""
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
        
        # 추천 결과 저장
        self.recommendations.append({
            'user': user_attributes,
            'action': action,
            'category': AD_CATEGORIES[action],
            'reward': reward,
            'timestamp': time.time()
        })
    
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
            'attribute_rewards': self.attribute_rewards,
            'recommendations': self.recommendations
        }
        
        try:
            torch.save(model_state, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
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
                
                if 'loss_history' in model_state:
                    self.loss_history = model_state['loss_history']
                if 'epsilon_history' in model_state:
                    self.epsilon_history = model_state['epsilon_history']
                if 'reward_history' in model_state:
                    self.reward_history = model_state['reward_history']
                if 'attribute_rewards' in model_state:
                    self.attribute_rewards = model_state['attribute_rewards']
                if 'recommendations' in model_state:
                    self.recommendations = model_state['recommendations']
                
                logger.info(f"Model loaded from {path} (Steps: {self.steps_done})")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
        else:
            logger.warning(f"Warning: {path} does not exist. Starting with a new model.")
            return False

# 시뮬레이션용 더미 사용자 생성
def generate_dummy_users(count=10):
    """시뮬레이션용 더미 사용자 생성"""
    dummy_users = []
    for _ in range(count):
        user = {
            "age": random.choice(AGE_CATEGORIES),
            "gender": random.choice(GENDER_CATEGORIES),
            "emotion": random.choice(EMOTION_CATEGORIES),
            "time": random.choice(TIME_CATEGORIES),
            "weather": random.choice(WEATHER_CATEGORIES),
            "gaze_time": random.uniform(0.5, 5.0)  # 0.5 ~ 5.0초 응시
        }
        dummy_users.append(user)
    return dummy_users

# 시각화용 클래스 정의
class AdVisualization:
    """광고 추천 결과 시각화 클래스"""
    
    def __init__(self, agent, root=None):
        """
        시각화 초기화
        
        Args:
            agent (DQNAgent): DQN 에이전트
            root (tk.Tk): Tkinter 루트 윈도우 (없으면 생성)
        """
        self.agent = agent
        
        # GUI 설정
        if root is None:
            self.root = tk.Tk()
            self.root.title("AdVise: 광고 추천 시스템")
            self.root.geometry("1200x800")
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        else:
            self.root = root
        
        # 메인 프레임
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 왼쪽 패널 (시각화)
        self.viz_frame = ttk.LabelFrame(self.main_frame, text="시각화 & 분석")
        self.viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 오른쪽 패널 (컨트롤)
        self.control_frame = ttk.LabelFrame(self.main_frame, text="제어 & 상태")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5, ipadx=5, ipady=5)
        
        # 시각화 탭
        self.notebook = ttk.Notebook(self.viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 탭 1: 학습 그래프
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="학습 그래프")
        
        # 탭 2: 사용자-광고 매칭
        self.recommendation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.recommendation_frame, text="추천 결과")
        
        # 탭 3: 속성별 보상
        self.rewards_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.rewards_frame, text="속성별 보상")
        
        # 탭 4: 사용자 데이터
        self.user_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.user_frame, text="사용자 데이터")
        
        # 현재 사용자 정보와 추천 광고
        self.current_ad_frame = ttk.LabelFrame(self.control_frame, text="현재 추천 광고")
        self.current_ad_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.ad_title_label = ttk.Label(self.current_ad_frame, text="광고 제목", font=("Arial", 12, "bold"))
        self.ad_title_label.pack(pady=(10, 0))
        
        self.ad_content_label = ttk.Label(self.current_ad_frame, text="광고 내용", wraplength=200)
        self.ad_content_label.pack(pady=(5, 0))
        
        self.ad_target_label = ttk.Label(self.current_ad_frame, text="타겟층", font=("Arial", 8, "italic"))
        self.ad_target_label.pack(pady=(5, 10))
        
        # 현재 사용자 정보
        self.user_info_frame = ttk.LabelFrame(self.control_frame, text="현재 사용자 정보")
        self.user_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 사용자 정보 표시
        self.user_age_label = ttk.Label(self.user_info_frame, text="나이: -")
        self.user_age_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.user_gender_label = ttk.Label(self.user_info_frame, text="성별: -")
        self.user_gender_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.user_emotion_label = ttk.Label(self.user_info_frame, text="감정: -")
        self.user_emotion_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.user_time_label = ttk.Label(self.user_info_frame, text="시간대: -")
        self.user_time_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.user_weather_label = ttk.Label(self.user_info_frame, text="계절: -")
        self.user_weather_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.user_gaze_label = ttk.Label(self.user_info_frame, text="응시 시간: - 초")
        self.user_gaze_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # 컨트롤 프레임
        self.sim_control_frame = ttk.LabelFrame(self.control_frame, text="시뮬레이션 컨트롤")
        self.sim_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 시뮬레이션 버튼
        self.run_sim_button = ttk.Button(self.sim_control_frame, text="시뮬레이션 실행", command=self.run_simulation)
        self.run_sim_button.pack(fill=tk.X, padx=5, pady=5)
        
        # 시뮬레이션 사용자 수 선택
        self.user_count_frame = ttk.Frame(self.sim_control_frame)
        self.user_count_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.user_count_frame, text="사용자 수:").pack(side=tk.LEFT, padx=5)
        
        self.user_count_var = tk.StringVar(value="10")
        self.user_count_combobox = ttk.Combobox(
            self.user_count_frame, 
            textvariable=self.user_count_var,
            values=["5", "10", "20", "50", "100"],
            state="readonly",
            width=5
        )
        self.user_count_combobox.pack(side=tk.LEFT, padx=5)
        
        # 속도 조절
        self.speed_frame = ttk.Frame(self.sim_control_frame)
        self.speed_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.speed_frame, text="시뮬레이션 속도:").pack(side=tk.LEFT, padx=5)
        
        self.speed_var = tk.StringVar(value="보통")
        self.speed_combobox = ttk.Combobox(
            self.speed_frame, 
            textvariable=self.speed_var,
            values=["느리게", "보통", "빠르게", "매우 빠르게"],
            state="readonly",
            width=10
        )
        self.speed_combobox.pack(side=tk.LEFT, padx=5)
        
        # 시스템 상태 정보
        self.status_frame = ttk.LabelFrame(self.control_frame, text="시스템 상태")
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.steps_label = ttk.Label(self.status_frame, text=f"학습 스텝: {self.agent.steps_done}")
        self.steps_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.epsilon_label = ttk.Label(self.status_frame, text=f"현재 입실론: {self._get_current_epsilon():.4f}")
        self.epsilon_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.loss_label = ttk.Label(self.status_frame, text=f"현재 손실: {self._get_current_loss():.4f}")
        self.loss_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.avg_reward_label = ttk.Label(self.status_frame, text=f"평균 보상: {self._get_avg_reward():.2f}")
        self.avg_reward_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # 파일 저장/로드 버튼
        self.file_frame = ttk.Frame(self.control_frame)
        self.file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.save_button = ttk.Button(self.file_frame, text="모델 저장", command=self.save_model)
        self.save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=5)
        
        self.load_button = ttk.Button(self.file_frame, text="모델 로드", command=self.load_model)
        self.load_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=5)
        
        # 시뮬레이션 플래그
        self.simulation_running = False
        self.current_user = None
        self.current_action = None
        
        # 초기 그래프 생성
        self.init_graphs()
        
    def init_graphs(self):
        """초기 그래프 설정"""
        # 학습 그래프 설정
        self.training_fig = Figure(figsize=(6, 8), dpi=100)
        
        # 서브플롯 1: 보상 그래프
        self.reward_plot = self.training_fig.add_subplot(311)
        self.reward_plot.set_title('보상 히스토리')
        self.reward_plot.set_xlabel('스텝')
        self.reward_plot.set_ylabel('보상 (응시시간)')
        
        # 서브플롯 2: 손실 그래프
        self.loss_plot = self.training_fig.add_subplot(312)
        self.loss_plot.set_title('손실 히스토리')
        self.loss_plot.set_xlabel('스텝')
        self.loss_plot.set_ylabel('손실값')
        
        # 서브플롯 3: 입실론 그래프
        self.epsilon_plot = self.training_fig.add_subplot(313)
        self.epsilon_plot.set_title('입실론 감소')
        self.epsilon_plot.set_xlabel('스텝')
        self.epsilon_plot.set_ylabel('입실론')
        self.epsilon_plot.set_ylim(0, 1)
        
        # 학습 그래프 캔버스 추가
        self.training_canvas = FigureCanvasTkAgg(self.training_fig, master=self.training_frame)
        self.training_canvas.draw()
        self.training_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 추천 결과 그래프 설정
        self.recommendation_fig = Figure(figsize=(6, 8), dpi=100)
        
        # 광고 카테고리 추천 비율 (파이 차트)
        self.recommendation_plot = self.recommendation_fig.add_subplot(211)
        self.recommendation_plot.set_title('광고 추천 비율')
        
        # Q-값 히스토그램
        self.q_value_plot = self.recommendation_fig.add_subplot(212)
        self.q_value_plot.set_title('카테고리별 Q-값 분포')
        self.q_value_plot.set_xlabel('광고 카테고리')
        self.q_value_plot.set_ylabel('평균 Q-값')
        
        # 추천 결과 캔버스 추가
        self.recommendation_canvas = FigureCanvasTkAgg(self.recommendation_fig, master=self.recommendation_frame)
        self.recommendation_canvas.draw()
        self.recommendation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 속성별 보상 그래프 설정
        self.rewards_fig = Figure(figsize=(6, 8), dpi=100)
        self.rewards_plot = self.rewards_fig.add_subplot(111)
        self.rewards_plot.set_title('사용자 속성별 평균 보상')
        self.rewards_plot.set_ylabel('평균 보상 (응시시간)')
        
        # 속성별 보상 캔버스 추가
        self.rewards_canvas = FigureCanvasTkAgg(self.rewards_fig, master=self.rewards_frame)
        self.rewards_canvas.draw()
        self.rewards_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 사용자 데이터 표시 설정
        self.user_tree = ttk.Treeview(
            self.user_frame, 
            columns=("age", "gender", "emotion", "time", "weather", "action", "reward"), 
            show="headings",
            height=20
        )
        
        # 각 열 설정
        self.user_tree.heading("age", text="나이")
        self.user_tree.heading("gender", text="성별")
        self.user_tree.heading("emotion", text="감정")
        self.user_tree.heading("time", text="시간대")
        self.user_tree.heading("weather", text="계절")
        self.user_tree.heading("action", text="추천 광고")
        self.user_tree.heading("reward", text="응시 시간")
        
        self.user_tree.column("age", width=80, anchor=tk.CENTER)
        self.user_tree.column("gender", width=60, anchor=tk.CENTER)
        self.user_tree.column("emotion", width=80, anchor=tk.CENTER)
        self.user_tree.column("time", width=80, anchor=tk.CENTER)
        self.user_tree.column("weather", width=60, anchor=tk.CENTER)
        self.user_tree.column("action", width=150, anchor=tk.CENTER)
        self.user_tree.column("reward", width=80, anchor=tk.CENTER)
        
        # 스크롤바 추가
        scrollbar = ttk.Scrollbar(self.user_frame, orient=tk.VERTICAL, command=self.user_tree.yview)
        self.user_tree.configure(yscrollcommand=scrollbar.set)
        
        # Treeview 및 스크롤바 배치
        self.user_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # 초기 데이터 로드
        self.update_plots()
        self.update_status()
        self.update_user_tree()
        
    def update_plots(self):
        """모든 그래프 업데이트"""
        # 1. 학습 그래프 업데이트
        self._update_training_plots()
        
        # 2. 추천 결과 그래프 업데이트
        self._update_recommendation_plots()
        
        # 3. 속성별 보상 그래프 업데이트
        self._update_rewards_plot()
        
    def _update_training_plots(self):
        """학습 지표 관련 그래프 업데이트"""
        # 그래프 초기화
        self.reward_plot.clear()
        self.loss_plot.clear()
        self.epsilon_plot.clear()
        
        # 보상 그래프
        if self.agent.reward_history:
            x = list(range(len(self.agent.reward_history)))
            self.reward_plot.plot(x, self.agent.reward_history, 'b-')
            
            # 이동 평균 추가
            if len(self.agent.reward_history) > 10:
                window_size = min(10, len(self.agent.reward_history))
                moving_avg = np.convolve(self.agent.reward_history, np.ones(window_size)/window_size, mode='valid')
                self.reward_plot.plot(range(window_size-1, len(self.agent.reward_history)), moving_avg, 'r-', label='이동평균(10)')
                self.reward_plot.legend()
        
        # 손실 그래프
        if self.agent.loss_history:
            x = list(range(len(self.agent.loss_history)))
            self.loss_plot.plot(x, self.agent.loss_history, 'g-')
        
        # 입실론 그래프
        if self.agent.epsilon_history:
            x = list(range(len(self.agent.epsilon_history)))
            self.epsilon_plot.plot(x, self.agent.epsilon_history, 'r-')
        
        # 그래프 제목 및 라벨 설정
        self.reward_plot.set_title('보상 히스토리')
        self.reward_plot.set_xlabel('스텝')
        self.reward_plot.set_ylabel('보상 (응시시간)')
        
        self.loss_plot.set_title('손실 히스토리')
        self.loss_plot.set_xlabel('스텝')
        self.loss_plot.set_ylabel('손실값')
        
        self.epsilon_plot.set_title('입실론 감소')
        self.epsilon_plot.set_xlabel('스텝')
        self.epsilon_plot.set_ylabel('입실론')
        self.epsilon_plot.set_ylim(0, 1)
        
        # 캔버스 업데이트
        self.training_fig.tight_layout()
        self.training_canvas.draw()
    
    def _update_recommendation_plots(self):
        """추천 결과 관련 그래프 업데이트"""
        # 그래프 초기화
        self.recommendation_plot.clear()
        self.q_value_plot.clear()
        
        # 추천 비율 파이 차트 (최근 100개)
        if self.agent.recommendations:
            # 최근 100개의 추천만 사용
            recent_recs = self.agent.recommendations[-100:]
            category_counts = Counter([rec['category'] for rec in recent_recs])
            
            labels = []
            sizes = []
            colors = []
            
            for i, category in enumerate(AD_CATEGORIES):
                count = category_counts.get(category, 0)
                if count > 0:
                    labels.append(f"{category} ({count})")
                    sizes.append(count)
                    colors.append(plt.cm.tab10(i))
            
            if sizes:
                self.recommendation_plot.pie(
                    sizes, 
                    labels=labels, 
                    autopct='%1.1f%%',
                    startangle=90, 
                    colors=colors
                )
                self.recommendation
    def run_simulation(self):
        """시뮬레이션 실행"""
        if self.simulation_running:
            self.simulation_running = False
            self.run_sim_button.config(text="시뮬레이션 실행")
            return
        
        # 시뮬레이션 시작
        self.simulation_running = True
        self.run_sim_button.config(text="시뮬레이션 중지")
        
        # 사용자 수 설정
        try:
            user_count = int(self.user_count_var.get())
        except ValueError:
            user_count = 10
        
        # 시뮬레이션 속도 설정
        speed = self.speed_var.get()
        if speed == "느리게":
            delay = 1.0
        elif speed == "보통":
            delay = 0.5
        elif speed == "빠르게":
            delay = 0.2
        else:  # "매우 빠르게"
            delay = 0.05
        
        # 시뮬레이션 스레드 시작
        simulation_thread = threading.Thread(
            target=self._simulation_loop,
            args=(user_count, delay),
            daemon=True
        )
        simulation_thread.start()
        
    def _simulation_loop(self, user_count, delay):
        """시뮬레이션 반복 실행"""
        # 더미 사용자 생성
        dummy_users = generate_dummy_users(user_count)
        
        # 각 사용자에 대해 반복
        for i, user in enumerate(dummy_users):
            # 시뮬레이션 중지 체크
            if not self.simulation_running:
                break
            
            # 사용자 설정
            self.current_user = user
            
            # UI에 현재 사용자 정보 표시
            self.root.after(0, lambda u=user: self.update_user_info(u))
            
            # 상태 인코딩
            state_vector = encode_state(user)
            
            # 초기 편향 계산
            initial_bias = get_initial_bias(user)
            
            # 행동 선택
            action, q_values = self.agent.select_action(state_vector, initial_bias)
            self.current_action = action
            
            # UI에 현재 광고 표시
            self.root.after(0, lambda a=AD_CATEGORIES[action]: self.update_ad_info(a))
            
            # 에이전트에 보상 추가
            reward = user.get('gaze_time', 1.0)
            self.agent.add_reward(reward)
            
            # 에이전트 특성별 보상 업데이트
            self.agent.update_attribute_rewards(user, reward, action)
            
            # 경험 저장
            self.agent.replay_buffer.push(
                state_vector.numpy(), 
                action, 
                reward, 
                state_vector.numpy(),  # 다음 상태는 같음
                True  # 항상 에피소드 종료
            )
            
            # 에이전트 업데이트
            batch_size = 32
            if len(self.agent.replay_buffer) >= batch_size:
                loss = self.agent.update(batch_size)
            
            # 타겟 네트워크 업데이트
            if i % 10 == 0:
                self.agent.update_target()
            
            # UI 업데이트
            if i % 5 == 0 or i == len(dummy_users) - 1:
                self.root.after(0, self.update_plots)
                self.root.after(0, self.update_status)
                self.root.after(0, self.update_user_tree)
            
            # 지연
            time.sleep(delay)
        
        # 시뮬레이션 완료
        self.root.after(0, lambda: self.run_sim_button.config(text="시뮬레이션 실행"))
        self.simulation_running = False
                
class AdSimulator:
    """광고 추천 시뮬레이터"""
    
    def __init__(self):
        """시뮬레이터 초기화"""
        # 상태 및 행동 공간 차원 계산
        state_dim = (
            len(AGE_CATEGORIES) + 
            len(GENDER_CATEGORIES) + 
            len(EMOTION_CATEGORIES) + 
            len(TIME_CATEGORIES) + 
            len(WEATHER_CATEGORIES)
        )
        action_dim = len(AD_CATEGORIES)
        
        # DQN 에이전트 생성
        self.agent = DQNAgent(state_dim, action_dim)
        
        # 저장된 모델이 있으면 로드
        self.agent.load_model(MODEL_PATH)
        self.agent.replay_buffer.load(BUFFER_PATH)
        
        # GUI 생성
        self.root = tk.Tk()
        self.visualization = AdVisualization(self.agent, self.root)
    
    def run(self):
        """시뮬레이터 실행"""
        self.root.mainloop()          
# 메인 실행 부분
if __name__ == "__main__":
    # 상태 및 행동 공간 차원 계산
    state_dim = (
        len(AGE_CATEGORIES) + 
        len(GENDER_CATEGORIES) + 
        len(EMOTION_CATEGORIES) + 
        len(TIME_CATEGORIES) + 
        len(WEATHER_CATEGORIES)
    )
    action_dim = len(AD_CATEGORIES)
    
    # GUI 모드로 시뮬레이터 실행
    try:
        simulator = AdSimulator()
        simulator.run()
    except Exception as e:
        logger.error(f"시뮬레이터 실행 중 오류 발생: {e}", exc_info=True)