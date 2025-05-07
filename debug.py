#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
디버그용 실행 코드

이 파일은 v5_new.py의 실행 문제를 디버깅하기 위한 코드입니다.
"""

import logging
import os
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG 레벨로 설정
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdRecommendation")

# 필요한 모듈 체크
def check_dependencies():
    missing_modules = []
    try:
        import torch
        logger.info("pytorch 로드 성공")
    except ImportError:
        missing_modules.append("torch")
    
    try:
        import numpy
        logger.info("numpy 로드 성공")
    except ImportError:
        missing_modules.append("numpy")
    
    try:
        import pandas
        logger.info("pandas 로드 성공")
    except ImportError:
        missing_modules.append("pandas")
    
    try:
        import matplotlib
        logger.info("matplotlib 로드 성공")
        # Tkinter 백엔드 가능한지 체크
        matplotlib.use('TkAgg')
    except ImportError:
        missing_modules.append("matplotlib")
    except Exception as e:
        logger.error(f"matplotlib 백엔드 설정 오류: {e}")
    
    try:
        import tkinter
        logger.info("tkinter 로드 성공")
    except ImportError:
        missing_modules.append("tkinter")
    
    if missing_modules:
        logger.error(f"필요한 모듈이 설치되지 않았습니다: {', '.join(missing_modules)}")
        return False
    return True

# 메인 함수에 try-except 추가
def main():
    logger.info("디버그 스크립트 시작")
    
    # 의존성 체크
    if not check_dependencies():
        logger.error("필요한 모듈이 설치되지 않아 종료합니다.")
        return
    
    try:
        # 원본 v5_new.py 실행
        logger.info("원본 스크립트 실행 시도")
        
        # 여기서 CSV 파일 존재 확인
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(project_root, "data")
        csv_files = [
            "category_weight_age.csv",
            "category_weight_sex.csv",
            "category_weight_time.csv",
            "category_weight_season.csv"
        ]
        
        for csv_file in csv_files:
            file_path = os.path.join(data_dir, csv_file)
            if os.path.exists(file_path):
                logger.info(f"CSV 파일 확인: {csv_file} 존재함")
            else:
                logger.warning(f"CSV 파일 없음: {csv_file}")
        
        # 여기서 GUI 초기화 시도
        try:
            import tkinter as tk
            root = tk.Tk()
            root.title("테스트 창")
            
            # 간단한 라벨 추가
            label = tk.Label(root, text="GUI 테스트")
            label.pack(padx=20, pady=20)
            
            # 종료 버튼 추가
            button = tk.Button(root, text="종료", command=root.quit)
            button.pack(pady=10)
            
            logger.info("GUI 초기화 성공, 메인루프 시작")
            root.mainloop()
            logger.info("GUI 메인루프 종료")
        except Exception as e:
            logger.error(f"GUI 초기화 오류: {e}")
            
    except Exception as e:
        logger.error(f"실행 중 예외 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()