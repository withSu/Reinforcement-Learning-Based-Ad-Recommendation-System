#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN 기반 광고 추천 시스템 (AdVise 프로젝트) - 유틸리티 함수

데이터 처리 및 상태 인코딩, 편향 계산, 더미 데이터 생성 등 다양한 유틸리티 함수들을 제공합니다.

Authors: [Your Names]
Version: 2.0.0
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime

# 설정 로드
from config import *

# 로깅 설정
logger = logging.getLogger("AdRecommendation")

# CSV 데이터 전역 변수
df_age = None
df_sex = None
df_time = None
df_season = None

def load_csv_data():
    """CSV 파일을 로드하고 초기화합니다"""
    global df_age, df_sex, df_time, df_season
    
    try:
        df_age = pd.read_csv(os.path.join(DATA_DIR, "category_weight_age.csv"))
        df_sex = pd.read_csv(os.path.join(DATA_DIR, "category_weight_sex.csv"))
        df_time = pd.read_csv(os.path.join(DATA_DIR, "category_weight_time.csv"))
        df_season = pd.read_csv(os.path.join(DATA_DIR, "category_weight_season.csv"))
        logger.info("CSV 파일 로드 성공")
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
        os.makedirs(DATA_DIR, exist_ok=True)
        df_age.to_csv(os.path.join(DATA_DIR, "category_weight_age.csv"), index=False)
        df_sex.to_csv(os.path.join(DATA_DIR, "category_weight_sex.csv"), index=False)
        df_time.to_csv(os.path.join(DATA_DIR, "category_weight_time.csv"), index=False)
        df_season.to_csv(os.path.join(DATA_DIR, "category_weight_season.csv"), index=False)
        
        logger.info("더미 CSV 데이터 생성 및 저장 완료")
    
    return df_age, df_sex, df_time, df_season

# 초기에 CSV 데이터 로드
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
    # CSV 데이터 확인 및 필요시 다시 로드
    global df_age, df_sex, df_time, df_season
    if df_age is None or df_sex is None or df_time is None or df_season is None:
        df_age, df_sex, df_time, df_season = load_csv_data()
    
    # 각 특성별 중요도 가중치 (연령과 성별이 시간이나 계절보다 중요)
    feature_weights = FEATURE_WEIGHTS
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
    
    # 감정별 편향 적용 (config.py에 정의된 EMOTION_BIAS 사용)
    if emotion in EMOTION_BIAS:
        for category, bias_value in EMOTION_BIAS[emotion].items():
            if category in AD_CATEGORIES:
                idx = AD_CATEGORIES.index(category)
                emotion_bias[idx] = bias_value
    
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

def get_current_time_category():
    """현재 시간에 따른 시간 카테고리 반환"""
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "오전"
    else:
        return "오후"

def get_current_season():
    """현재 월에 따른 계절 카테고리 반환"""
    current_month = datetime.now().month
    if 3 <= current_month <= 5:
        return "봄"
    elif 6 <= current_month <= 8:
        return "여름"
    elif 9 <= current_month <= 11:
        return "가을"
    else:
        return "겨울"

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

def calculate_state_dimension():
    """상태 벡터 차원 계산"""
    return (
        len(AGE_CATEGORIES) + 
        len(GENDER_CATEGORIES) + 
        len(EMOTION_CATEGORIES) + 
        len(TIME_CATEGORIES) + 
        len(WEATHER_CATEGORIES)
    )