#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN 기반 광고 추천 시스템 (AdVise 프로젝트) - 메인 실행 파일

이 모듈은 전체 시스템의 메인 실행 파일입니다.
- 강화학습 에이전트 생성 및 학습
- 시각화 인터페이스 실행
- 시뮬레이션 모드 및 평가 모드 실행

Authors: [Your Names]
Version: 2.0.0
"""

import os
import time
import argparse
import logging
import tkinter as tk
import numpy as np
import torch

# 설정 및 유틸리티 함수 로드
from config import *
from utils import *
from models import DQNAgent
from visualizer import AdVisualization

# 로깅 설정
logger = logging.getLogger("AdRecommendation")

class AdSimulator:
    """광고 추천 시뮬레이터"""
    
    def __init__(self):
        """시뮬레이터 초기화"""
        # 상태 및 행동 공간 차원 계산
        state_dim = calculate_state_dimension()
        action_dim = len(AD_CATEGORIES)
        
        # DQN 에이전트 생성
        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=LEARNING_RATE,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_final=EPSILON_FINAL,
            epsilon_decay=EPSILON_DECAY
        )
        
        # 저장된 모델이 있으면 로드
        self.agent.load_model(MODEL_PATH)
        self.agent.replay_buffer.load(BUFFER_PATH)
        
        # GUI 생성
        self.root = tk.Tk()
        self.visualization = AdVisualization(self.agent, self.root)
    
    def run(self):
        """시뮬레이터 실행"""
        try:
            self.root.mainloop()
        except Exception as e:
            logger.error(f"GUI 실행 중 오류 발생: {e}", exc_info=True)
        finally:
            # 종료 시 모델 저장
            if self.agent.steps_done > 0:
                self.agent.save_model(MODEL_PATH)
                self.agent.replay_buffer.save(BUFFER_PATH)

def train_console_mode(user_count=100, verbose=True):
    """콘솔 모드에서 학습 실행"""
    # 상태 및 행동 공간 차원 계산
    state_dim = calculate_state_dimension()
    action_dim = len(AD_CATEGORIES)
    
    # DQN 에이전트 생성
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_final=EPSILON_FINAL,
        epsilon_decay=EPSILON_DECAY
    )
    
    # 저장된 모델이 있으면 로드
    agent.load_model(MODEL_PATH)
    agent.replay_buffer.load(BUFFER_PATH)
    
    # 더미 사용자 생성
    dummy_users = generate_dummy_users(user_count)
    
    # 학습 시작
    logger.info(f"콘솔 모드에서 {user_count}명의 사용자로 학습 시작")
    total_rewards = 0
    start_time = time.time()
    
    for i, user in enumerate(dummy_users):
        # 상태 인코딩
        state_vector = encode_state(user)
        
        # 초기 편향 계산
        initial_bias = get_initial_bias(user)
        
        # 행동 선택
        action, q_values = agent.select_action(state_vector, initial_bias)
        
        # 보상 계산 (응시 시간)
        reward = user.get('gaze_time', 1.0)
        total_rewards += reward
        
        # 에이전트 업데이트
        agent.add_reward(reward)
        agent.update_attribute_rewards(user, reward, action)
        
        # 경험 저장
        agent.replay_buffer.push(
            state_vector.numpy(), 
            action, 
            reward, 
            state_vector.numpy(),  # 다음 상태는 같음
            True  # 항상 에피소드 종료
        )
        
        # 신경망 업데이트
        batch_size = BATCH_SIZE
        if len(agent.replay_buffer) >= batch_size:
            loss = agent.update(batch_size)
        
        # 타겟 네트워크 업데이트
        if i % TARGET_UPDATE == 0:
            agent.update_target()
        
        # 진행 상황 출력
        if verbose and i % 10 == 0:
            logger.info(f"사용자 {i+1}/{user_count}, "
                       f"행동: {AD_CATEGORIES[action]}, "
                       f"보상: {reward:.2f}, "
                       f"입실론: {agent.epsilon_history[-1]:.4f}")
    
    # 학습 결과 출력
    elapsed_time = time.time() - start_time
    avg_reward = total_rewards / user_count
    
    logger.info(f"학습 완료! 소요 시간: {elapsed_time:.2f}초")
    logger.info(f"평균 보상: {avg_reward:.4f}, 총 스텝: {agent.steps_done}")
    
    # 모델 저장
    agent.save_model(MODEL_PATH)
    agent.replay_buffer.save(BUFFER_PATH)
    logger.info("모델과 리플레이 버퍼 저장 완료")
    
    return agent

def evaluate_console_mode(user_count=10, verbose=True):
    """콘솔 모드에서 평가 실행"""
    # 상태 및 행동 공간 차원 계산
    state_dim = calculate_state_dimension()
    action_dim = len(AD_CATEGORIES)
    
    # DQN 에이전트 생성
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_final=EPSILON_FINAL,
        epsilon_decay=EPSILON_DECAY
    )
    
    # 저장된 모델 로드
    success = agent.load_model(MODEL_PATH)
    if not success:
        logger.error("모델 로드 실패! 평가를 진행할 수 없습니다.")
        return
    
    # 테스트 사용자 생성
    test_users = generate_dummy_users(user_count)
    
    # 평가 시작
    logger.info(f"콘솔 모드에서 {user_count}명의 사용자로 평가 시작")
    total_rewards = 0
    category_counts = {category: 0 for category in AD_CATEGORIES}
    
    for i, user in enumerate(test_users):
        # 상태 인코딩
        state_vector = encode_state(user)
        
        # 초기 편향 계산
        initial_bias = get_initial_bias(user)
        
        # 행동 선택 (평가 모드에서는 탐색 없음)
        action, q_values = agent.select_action(state_vector, initial_bias, training=False)
        
        # 선택된 광고 카테고리 카운트 증가
        category = AD_CATEGORIES[action]
        category_counts[category] += 1
        
        # 보상 (실제 광고 노출 환경에서는 측정 가능)
        reward = user.get('gaze_time', 1.0)
        total_rewards += reward
        
        # 결과 출력
        if verbose:
            logger.info(f"사용자 {i+1}:")
            logger.info(f"  특성: 나이={user['age']}, 성별={user['gender']}, "
                       f"감정={user['emotion']}, 시간={user['time']}, 계절={user['weather']}")
            logger.info(f"  추천 광고: {category}")
            logger.info(f"  가중치: {initial_bias.round(2)}")
            if q_values is not None:
                q_values_np = q_values.cpu().numpy()
                logger.info(f"  Q-값: {q_values_np.round(2)}")
            logger.info("")
    
    # 평가 결과 요약
    avg_reward = total_rewards / user_count
    logger.info(f"평가 완료! 평균 보상: {avg_reward:.4f}")
    logger.info("카테고리별 추천 비율:")
    for category, count in category_counts.items():
        percentage = (count / user_count) * 100
        logger.info(f"  {category}: {percentage:.1f}% ({count}명)")
    
    return avg_reward, category_counts

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="DQN 기반 광고 추천 시스템")
    parser.add_argument('--mode', type=str, default='gui', choices=['gui', 'train', 'eval'], 
                      help='실행 모드 (gui: GUI 시각화, train: 학습, eval: 평가)')
    parser.add_argument('--users', type=int, default=100, 
                      help='학습/평가 시 사용자 수')
    parser.add_argument('--verbose', action='store_true', 
                      help='상세 로깅 활성화')
    parser.add_argument('--debug', action='store_true', 
                      help='디버그 모드 활성화')
    
    args = parser.parse_args()
    
    # 디버그 모드 설정
    if args.debug:
        logging.getLogger("AdRecommendation").setLevel(logging.DEBUG)
        logger.debug("디버그 모드 활성화")
    
    # 디렉토리 정보 출력
    logger.info(f"프로젝트 루트: {PROJECT_ROOT}")
    logger.info(f"데이터 디렉토리: {DATA_DIR}")
    logger.info(f"모델 디렉토리: {MODEL_DIR}")
    
    try:
        # 모드에 따른 실행
        if args.mode == 'gui':
            logger.info("GUI 모드로 실행")
            simulator = AdSimulator()
            simulator.run()
        elif args.mode == 'train':
            logger.info(f"학습 모드로 실행 (사용자 수: {args.users})")
            train_console_mode(args.users, args.verbose)
        elif args.mode == 'eval':
            logger.info(f"평가 모드로 실행 (사용자 수: {args.users})")
            evaluate_console_mode(args.users, args.verbose)
    except KeyboardInterrupt:
        logger.info("사용자에 의해 프로그램 중단")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}", exc_info=True)
    finally:
        logger.info("프로그램 종료")

if __name__ == "__main__":
    main()