#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN 기반 광고 추천 시스템 (AdVise 프로젝트) - 시각화

추천 결과 및 학습 과정을 시각화하는 클래스를 정의합니다.
Tkinter와 Matplotlib을 활용한 실시간 시각화 인터페이스를 제공합니다.

Authors: [Your Names]
Version: 2.0.0
"""

import os
import time
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading
import logging
from datetime import datetime

# CJK 중에서 Sans 계열을 한글 시각화용으로 지정
try:
    plt.rc('font', family='Noto Sans CJK JP')
    plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지
except Exception as e:
    logger = logging.getLogger("AdRecommendation")
    logger.warning(f"폰트 설정 오류: {e}, 기본 폰트를 사용합니다.")

# 설정 및 유틸리티 함수 로드
from config import *
from utils import encode_state, get_initial_bias, generate_dummy_users

# 로깅 설정
logger = logging.getLogger("AdRecommendation")

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
                self.recommendation_plot.axis('equal')
                self.recommendation_plot.set_title('광고 추천 비율')
        
        # Q-값 히스토그램
        avg_q_values = {}
        total_rewards = {}
        
        # 각 카테고리별 평균 Q-값 계산
        for rec in self.agent.recommendations:
            category = rec['category']
            reward = rec['reward']
            
            if category not in avg_q_values:
                avg_q_values[category] = []
                total_rewards[category] = []
                
            total_rewards[category].append(reward)
        
        # 평균 보상 계산
        category_rewards = []
        category_labels = []
        
        for category in AD_CATEGORIES:
            if category in total_rewards and total_rewards[category]:
                avg_reward = sum(total_rewards[category]) / len(total_rewards[category])
                category_rewards.append(avg_reward)
                category_labels.append(category)
            else:
                category_rewards.append(0)
                category_labels.append(category)
        
        # 막대 그래프 그리기
        bars = self.q_value_plot.bar(
            range(len(category_labels)), 
            category_rewards,
            color=plt.cm.tab10.colors[:len(category_labels)]
        )
        
        # 막대 위에 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                self.q_value_plot.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + 0.1,
                    f'{height:.2f}',
                    ha='center', 
                    va='bottom', 
                    fontsize=8
                )
        
        # x축 레이블 설정
        self.q_value_plot.set_xticks(range(len(category_labels)))
        self.q_value_plot.set_xticklabels(category_labels, rotation=45, ha='right')
        
        # y축 레이블 설정
        self.q_value_plot.set_ylabel('평균 보상')
        self.q_value_plot.set_title('카테고리별 평균 보상')
        
        # 캔버스 업데이트
        self.recommendation_fig.tight_layout()
        self.recommendation_canvas.draw()
    
    def _update_rewards_plot(self):
        """속성별 보상 그래프 업데이트"""
        # 그래프 초기화
        self.rewards_plot.clear()
        
        # 속성별 평균 보상 계산
        all_attr_rewards = []
        
        # 각 속성 타입(나이, 성별 등)별로 루프
        for attr_type, attr_data in self.agent.attribute_rewards.items():
            # 각 속성 값(20대, 30대 등)별로 루프
            for attr_value, rewards in attr_data.items():
                if rewards:  # 보상이 있는 경우만
                    avg_reward = sum(rewards) / len(rewards)
                    all_attr_rewards.append({
                        'type': attr_type,
                        'value': attr_value,
                        'avg_reward': avg_reward,
                        'count': len(rewards)
                    })
        
        # 정렬: 속성 타입 > 평균 보상(내림차순)
        all_attr_rewards.sort(key=lambda x: (x['type'], -x['avg_reward']))
        
        if all_attr_rewards:
            # 데이터 준비
            type_labels = []
            x_positions = []
            avg_rewards = []
            colors = []
            
            current_type = None
            x_pos = 0
            type_colors = {
                'age': 'royalblue',
                'gender': 'orangered',
                'emotion': 'forestgreen',
                'time': 'purple',
                'weather': 'goldenrod'
            }
            
            for i, data in enumerate(all_attr_rewards):
                # 속성 타입이 바뀌면 공간 추가
                if current_type != data['type']:
                    if current_type is not None:
                        x_pos += 1  # 타입 간 공간
                    current_type = data['type']
                
                type_labels.append(f"{data['value']}\n({data['count']}명)")
                x_positions.append(x_pos)
                avg_rewards.append(data['avg_reward'])
                colors.append(type_colors.get(data['type'], 'gray'))
                
                x_pos += 1
            
            # 막대 그래프 그리기
            bars = self.rewards_plot.bar(x_positions, avg_rewards, color=colors)
            
            # 막대 위에 값 표시
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    self.rewards_plot.text(
                        bar.get_x() + bar.get_width()/2., 
                        height + 0.1,
                        f'{height:.2f}',
                        ha='center', 
                        va='bottom', 
                        fontsize=8
                    )
            
            # x축 레이블 설정
            self.rewards_plot.set_xticks(x_positions)
            self.rewards_plot.set_xticklabels(type_labels, rotation=45, ha='right')
            
            # y축 레이블 설정
            self.rewards_plot.set_ylabel('평균 보상 (응시 시간)')
            self.rewards_plot.set_title('사용자 속성별 평균 보상')
            
            # 범례 추가
            legend_elements = [
                plt.Line2D([0], [0], color=color, lw=4, label=attr_type.capitalize())
                for attr_type, color in type_colors.items()
                if any(data['type'] == attr_type for data in all_attr_rewards)
            ]
            self.rewards_plot.legend(handles=legend_elements, loc='upper right')
        
        # 캔버스 업데이트
        self.rewards_fig.tight_layout()
        self.rewards_canvas.draw()
    
    def update_user_tree(self):
        """사용자 데이터 트리뷰 업데이트"""
        # 기존 항목 모두 제거
        for i in self.user_tree.get_children():
            self.user_tree.delete(i)
        
        # 최근 100개 추천 결과 표시
        recent_recs = self.agent.recommendations[-100:] if self.agent.recommendations else []
        
        for i, rec in enumerate(recent_recs):
            user = rec['user']
            action = rec['category']
            reward = rec['reward']
            
            self.user_tree.insert(
                "", 
                tk.END, 
                values=(
                    user.get('age', '-'),
                    user.get('gender', '-'),
                    user.get('emotion', '-'),
                    user.get('time', '-'),
                    user.get('weather', '-'),
                    action,
                    f"{reward:.2f}초"
                )
            )
    
    def update_status(self):
        """시스템 상태 정보 업데이트"""
        # 학습 스텝
        self.steps_label.config(text=f"학습 스텝: {self.agent.steps_done}")
        
        # 현재 입실론
        self.epsilon_label.config(text=f"현재 입실론: {self._get_current_epsilon():.4f}")
        
        # 현재 손실
        self.loss_label.config(text=f"현재 손실: {self._get_current_loss():.4f}")
        
        # 평균 보상
        self.avg_reward_label.config(text=f"평균 보상: {self._get_avg_reward():.2f}초")
    
    def _get_current_epsilon(self):
        """현재 입실론 값 반환"""
        if self.agent.epsilon_history:
            return self.agent.epsilon_history[-1]
        return 1.0
    
    def _get_current_loss(self):
        """현재 손실값 반환"""
        if self.agent.loss_history:
            return self.agent.loss_history[-1]
        return 0.0
    
    def _get_avg_reward(self):
        """최근 10개 보상의 평균 반환"""
        if self.agent.reward_history:
            recent = self.agent.reward_history[-10:]
            return sum(recent) / len(recent)
        return 0.0
    
    def update_user_info(self, user):
        """현재 사용자 정보 업데이트"""
        self.user_age_label.config(text=f"나이: {user.get('age', '-')}")
        self.user_gender_label.config(text=f"성별: {user.get('gender', '-')}")
        self.user_emotion_label.config(text=f"감정: {user.get('emotion', '-')}")
        self.user_time_label.config(text=f"시간대: {user.get('time', '-')}")
        self.user_weather_label.config(text=f"계절: {user.get('weather', '-')}")
        self.user_gaze_label.config(text=f"응시 시간: {user.get('gaze_time', 0):.2f}초")
    
    def update_ad_info(self, ad_category):
        """현재 광고 정보 업데이트"""
        ad_info = SAMPLE_ADS.get(ad_category, {
            'title': '광고 정보 없음',
            'content': '선택된 광고 카테고리에 대한 정보가 없습니다.',
            'target': '-',
            'color': '#CCCCCC'
        })
        
        self.ad_title_label.config(text=ad_info['title'])
        self.ad_content_label.config(text=ad_info['content'])
        self.ad_target_label.config(text=f"타겟층: {ad_info['target']}")
        
        # 광고 배경색 변경 시도
        try:
            style = ttk.Style()
            style.configure('Ad.TLabelframe', background=ad_info['color'])
            self.current_ad_frame.configure(style='Ad.TLabelframe')
        except Exception as e:
            logger.warning(f"광고 배경색 변경 실패: {e}")
    
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
        # 시작 로그
        logger.info(f"시뮬레이션 시작: 사용자 수={user_count}, 지연={delay}초")
        
        try:
            # 더미 사용자 생성
            dummy_users = generate_dummy_users(user_count)
            
            # 각 사용자에 대해 반복
            for i, user in enumerate(dummy_users):
                # 시뮬레이션 중지 체크
                if not self.simulation_running:
                    logger.info("시뮬레이션 중지됨")
                    break
                
                # 진행상황 로깅
                if i % 10 == 0 or i == len(dummy_users) - 1:
                    logger.debug(f"시뮬레이션 진행중: {i+1}/{user_count} 사용자")
                
                # 사용자 설정
                self.current_user = user
                
                # UI에 현재 사용자 정보 표시 (명시적 람다 함수 사용)
                self.root.after(0, lambda u=user: self.update_user_info(u))
                
                # 상태 인코딩
                state_vector = encode_state(user)
                
                # 초기 편향 계산
                initial_bias = get_initial_bias(user)
                
                # 행동 선택
                action, q_values = self.agent.select_action(state_vector, initial_bias)
                self.current_action = action
                
                # UI에 현재 광고 표시 (명시적 람다 함수 사용)
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
                    logger.debug(f"배치 업데이트 - 손실: {loss:.4f}")
                
                # 타겟 네트워크 업데이트
                if i % 10 == 0:
                    self.agent.update_target()
                
                # UI 업데이트 - 매 5명의 사용자마다 또는 마지막 사용자일 때
                if i % 5 == 0 or i == len(dummy_users) - 1:
                    # 명시적 람다 함수 사용
                    self.root.after(0, lambda: self.update_plots())
                    self.root.after(0, lambda: self.update_status())
                    self.root.after(0, lambda: self.update_user_tree())
                
                # 지연
                time.sleep(delay)
            
            # 시뮬레이션 완료
            logger.info(f"시뮬레이션 완료: {user_count}명의 사용자 처리됨")
        except Exception as e:
            logger.error(f"시뮬레이션 오류: {e}", exc_info=True)
        finally:
            # 버튼 상태 복원
            self.root.after(0, lambda: self.run_sim_button.config(text="시뮬레이션 실행"))
            self.simulation_running = False
    
    def save_model(self):
        """모델 저장"""
        try:
            self.agent.save_model(MODEL_PATH)
            self.agent.replay_buffer.save(BUFFER_PATH)
            logger.info("모델과 리플레이 버퍼가 저장되었습니다.")
        except Exception as e:
            logger.error(f"모델 저장 오류: {e}", exc_info=True)
    
    def load_model(self):
        """모델 로드"""
        try:
            success = self.agent.load_model(MODEL_PATH)
            if success:
                self.agent.replay_buffer.load(BUFFER_PATH)
                self.update_plots()
                self.update_status()
                self.update_user_tree()
                logger.info("모델과 리플레이 버퍼를 로드했습니다.")
            else:
                logger.warning("모델 로드에 실패했습니다. 새 모델로 초기화합니다.")
        except Exception as e:
            logger.error(f"모델 로드 오류: {e}", exc_info=True)
    
    def on_closing(self):
        """창 닫기 이벤트 처리"""
        # 실행 중인 시뮬레이션 중지
        self.simulation_running = False
        
        # 모델 저장 여부 확인
        if self.agent.steps_done > 0:
            self.save_model()
        
        # 창 닫기
        self.root.destroy()