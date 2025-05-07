#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN 기반 광고 추천 시스템 (AdVise 프로젝트) - 설정 및 상수

이 모듈에는 시스템 전체에서 사용되는 상수와 환경 설정이 정의되어 있습니다.

Authors: [Your Names]
Version: 2.0.0
"""

import os
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,  # INFO에서 DEBUG로 변경하여 더 상세한 로그 출력
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ad_recommendation.log"),
        logging.StreamHandler()
    ]
)

# 디렉토리 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리
DATA_DIR = os.path.join(PROJECT_ROOT, "data")  # 데이터 디렉토리
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")  # 모델 저장 디렉토리

# 디렉토리 생성
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

# DQN 하이퍼파라미터
LEARNING_RATE = 1e-3  # 학습률
GAMMA = 0.99  # 할인율
EPSILON_START = 1.0  # 초기 입실론
EPSILON_FINAL = 0.01  # 최종 입실론
EPSILON_DECAY = 500  # 입실론 감소율
HIDDEN_DIM = 64  # 은닉층 차원
BATCH_SIZE = 32  # 배치 크기
TARGET_UPDATE = 10  # 타겟 네트워크 업데이트 간격
REPLAY_BUFFER_SIZE = 10000  # 리플레이 버퍼 크기

# 사용자 특성별 가중치 (CSV 기반 편향 계산용)
FEATURE_WEIGHTS = {
    "age": 0.35,
    "gender": 0.25,
    "time": 0.20,
    "weather": 0.20
}

# 감정별 광고 카테고리 휴리스틱 편향
EMOTION_BIAS = {
    "happy": {
        "Culture_Entertainment": 0.15,
        "Dining": 0.15,
        "Clothing": 0.1
    },
    "sad": {
        "Education": 0.15,
        "Durables": 0.1
    },
    "angry": {
        "Transportation": 0.15,
        "Durables": 0.1
    },
    "surprise": {
        "Culture_Entertainment": 0.15,
        "Education": 0.1
    }
}