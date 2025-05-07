#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN 기반 광고 추천 시스템 (AdVise 프로젝트) - 모델 클래스

강화학습 모델 관련 클래스들을 정의합니다:
- DQN 신경망
- 리플레이 버퍼
- DQN 에이전트

Authors: [Your Names]
Version: 2.0.0
"""

import os
import time as time_module  # time 모듈 이름 변경
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
import logging
from datetime import datetime

# 설정 로드
from config import *

# 로깅 설정
logger = logging.getLogger("AdRecommendation")

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
        try:
            with open(path, 'wb') as f:
                pickle.dump(list(self.buffer), f)
            logger.info(f"Replay buffer saved to {path}")
        except Exception as e:
            logger.error(f"Error saving replay buffer: {e}")
    
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
                # 파일이 손상되었거나 호환되지 않는 경우, 파일을 새로 시작하도록 처리
                logger.warning("Removing incompatible replay buffer file and starting with a new one")
                try:
                    # 기존 파일 백업
                    backup_path = path + ".bak"
                    os.rename(path, backup_path)
                    logger.info(f"Original replay buffer file backed up to {backup_path}")
                except Exception as be:
                    logger.error(f"Failed to backup replay buffer file: {be}")
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
        
        # 손실값 로깅 추가 - 디버깅용
        logger.debug(f"Batch update - Loss: {loss_value:.4f}, Epsilon: {self.epsilon_history[-1]:.4f}")
        
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
            
        user_time = user_attributes.get('time')  # 'time' 변수명 변경
        if user_time:
            if user_time not in self.attribute_rewards['time']:
                self.attribute_rewards['time'][user_time] = []
            self.attribute_rewards['time'][user_time].append(reward)
            
        weather = user_attributes.get('weather')
        if weather:
            if weather not in self.attribute_rewards['weather']:
                self.attribute_rewards['weather'][weather] = []
            self.attribute_rewards['weather'][weather].append(reward)
        
        # 추천 결과 저장 - time 모듈 이름 충돌 수정
        self.recommendations.append({
            'user': user_attributes,
            'action': action,
            'category': AD_CATEGORIES[action],
            'reward': reward,
            'timestamp': time_module.time()  # time_module 사용
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
            # 디렉토리 확인 및 생성
            os.makedirs(os.path.dirname(path), exist_ok=True)
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
                # PyTorch 2.6 호환성을 위해 weights_only=False로 명시적 설정
                try:
                    # 먼저 weights_only=False로 시도
                    model_state = torch.load(path, map_location=self.device, weights_only=False)
                except (TypeError, ValueError):
                    # 이전 버전 PyTorch의 경우 weights_only 인자가 없을 수 있음
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
                # 파일이 손상되었거나 호환되지 않는 경우, 파일을 새로 시작하도록 처리
                logger.warning("Removing incompatible model file and starting with a new one")
                try:
                    # 기존 모델 파일 백업
                    backup_path = path + ".bak"
                    os.rename(path, backup_path)
                    logger.info(f"Original model file backed up to {backup_path}")
                except Exception as be:
                    logger.error(f"Failed to backup model file: {be}")
                return False
        else:
            logger.warning(f"Warning: {path} does not exist. Starting with a new model.")
            return False