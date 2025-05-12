# AdVise-ML: DQN 기반 광고 추천 시스템

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.6.0-orange)

<p align="center">
  <img src="/images/image (0).png" alt="AdVise-ML 시스템 아키텍처" width="700">
  <br>
  <em>AdVise-ML 시스템 아키텍처</em>
</p>

AdVise-ML은 강화학습(Deep Q-Network, DQN)을 활용하여 사용자 특성에 기반한 맞춤형 광고를 추천하는 인공지능 시스템입니다. 사용자의 나이, 성별, 감정 상태, 시간대, 계절과 같은 컨텍스트 정보를 활용하여 최적의 광고 카테고리를 선택하고, 사용자 반응(응시 시간)을 기반으로 학습하는 지능형 추천 시스템입니다.

## 프로젝트 목적

현대 디지털 광고 환경에서는 개인화된 맞춤형 광고가 중요합니다. AdVise-ML은 다음과 같은 목적을 가지고 있습니다:

1. **사용자 특성 기반 광고 추천**: 사용자의 인구통계학적 특성(나이, 성별)과 상황적 특성(감정, 시간대, 계절)을 고려한 개인화된 광고 제공
2. **실시간 학습 및 적응**: 사용자 반응(응시 시간)을 바탕으로 지속적인 학습과 모델 개선
3. **투명한 추천 시각화**: 추천 결과와 학습 과정을 직관적으로 시각화하여 추천 의사결정의 이해 제공
4. **효율적인 광고 전달**: 최적의 광고 카테고리를 선택하여 사용자 만족도와 광고 효과 극대화

## 시스템 아키텍처

AdVise-ML은 다음과 같은 핵심 컴포넌트로 구성되어 있습니다:

```
ADVISE-ML/
│
├── data/                      # 데이터 디렉토리
│   ├── category_weight_age.csv    # 연령별 광고 카테고리 가중치
│   ├── category_weight_season.csv # 계절별 광고 카테고리 가중치
│   ├── category_weight_sex.csv    # 성별 광고 카테고리 가중치
│   └── category_weight_time.csv   # 시간대별 광고 카테고리 가중치
│
├── models/                    # 모델 저장 디렉토리
│   ├── dqn_model.pth             # 학습된 DQN 모델
│   └── replay_buffer.pkl         # 경험 재생 버퍼
│
├── config.py                  # 시스템 전체 설정 및 상수
├── models.py                  # DQN 모델 정의 및 에이전트 클래스
├── utils.py                   # 유틸리티 함수 (상태 인코딩, 편향 계산 등)
├── visualizer.py              # GUI 시각화 인터페이스
├── main.py                    # 메인 실행 파일
├── debug.py                   # 디버깅 및 테스트 도구
├── environment.yml            # Conda 환경 설정 파일
└── ad_recommendation.log      # 로그 파일
```

### 핵심 컴포넌트

1. **DQN 에이전트** (models.py)

   - 심층 Q-네트워크(DQN) 기반 강화학습 에이전트
   - 사용자 특성을 입력으로 받아 최적의 광고 카테고리 선택
   - Double DQN 알고리즘 및 경험 재생(Experience Replay) 사용

2. **상태 관리** (utils.py)

   - 사용자 특성의 원-핫 인코딩
   - CSV 기반 초기 편향 계산
   - 감정 상태에 따른 휴리스틱 편향 적용

3. **시각화 인터페이스** (visualizer.py)

   - Tkinter 기반 GUI
   - 학습 그래프, 추천 결과, 사용자 특성별 보상 시각화
   - 실시간 시뮬레이션 도구

4. **메인 시스템** (main.py)
   - GUI 및 콘솔 모드 지원
   - 학습, 평가, 시뮬레이션 기능
   - 명령줄 인터페이스(CLI) 제공

## 강화학습 적용 방식

AdVise-ML은 광고 추천 문제를 강화학습의 관점에서 접근합니다:

<p align="center">
  <img src="/images/image (1).png" alt="강화학습 기반 광고 추천 프레임워크" width="700">
  <br>
  <em>강화학습 기반 광고 추천 프레임워크</em>
</p>

### 1. MDP(Markov Decision Process) 모델링

- **상태(State)**: 사용자 특성 벡터 (나이, 성별, 감정, 시간대, 계절을 원-핫 인코딩)
- **행동(Action)**: 추천할 광고 카테고리 (교양*오락*문화, 교육, 교통, 내구재, 외식, 의류)
- **보상(Reward)**: 사용자의 광고 응시 시간 (시뮬레이션에서는 0.5~5.0초 범위의 값)
- **상태 전이(Transition)**: 매 추천마다 새로운 사용자 (에피소딕 환경)

### 2. DQN(Deep Q-Network) 알고리즘

- **신경망 구조**: 3층 완전연결 신경망 (FC1-ReLU-FC2-ReLU-FC3)
- **학습 전략**:
  - Double DQN (과대평가 문제 완화)
  - 경험 재생 (시간적 상관관계 문제 해결)
  - 입실론-탐욕(ε-greedy) 정책 (탐색-활용 균형)
- **초기 편향**: CSV 데이터 기반 초기 Q-값 편향 설정으로 학습 가속화

### 3. 하이퍼파라미터

```python
# DQN 하이퍼파라미터
LEARNING_RATE = 1e-3    # 학습률
GAMMA = 0.99            # 할인율
EPSILON_START = 1.0     # 초기 입실론
EPSILON_FINAL = 0.01    # 최종 입실론
EPSILON_DECAY = 500     # 입실론 감소율
HIDDEN_DIM = 64         # 은닉층 차원
BATCH_SIZE = 32         # 배치 크기
TARGET_UPDATE = 10      # 타겟 네트워크 업데이트 간격
REPLAY_BUFFER_SIZE = 10000  # 리플레이 버퍼 크기
```

## 코드 구조 및 역할 설명

### 1. config.py

시스템 전체에서 사용되는 상수와 환경 설정을 정의합니다.

- 로깅 설정 및 디렉토리 구성
- 광고 카테고리 및 사용자 특성 카테고리 정의
- 샘플 광고 정보 (제목, 내용, 타겟, 색상)
- DQN 하이퍼파라미터
- 특성별 가중치 및 감정 편향 설정

```python
# 광고 카테고리 예시
AD_CATEGORIES_EN = ['Culture_Entertainment', 'Education', 'Transportation', 'Durables', 'Dining', 'Clothing']
AD_CATEGORIES_KO = ['교양_오락_문화', '교육', '교통', '내구재', '외식', '의류']

# 상태 인코딩을 위한 카테고리 정의
AGE_CATEGORIES = ["20세미만", "20-30세", "31-40세", "41-50세", "51-60세", "61-70세", "70세이상"]
GENDER_CATEGORIES = ["남성", "여성"]
EMOTION_CATEGORIES = ["happy", "neutral", "sad", "angry", "surprise", "fear", "disgust"]
```

### 2. models.py

DQN 신경망 모델, 리플레이 버퍼, DQN 에이전트 클래스를 정의합니다.

- **DQN 클래스**: 상태를 입력받아 각 광고 카테고리의 Q-값을 출력하는 신경망
- **ReplayBuffer 클래스**: 경험(state, action, reward, next_state, done)을 저장하고 샘플링
- **DQNAgent 클래스**:
  - 행동 선택 (입실론-탐욕 정책)
  - 신경망 업데이트 (Double DQN)
  - 모델 저장 및 로드
  - 학습 지표 추적

```python
# DQN 신경망 구조 예시
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # 가중치 초기화 (Kaiming/He 초기화)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 3. utils.py

데이터 처리 및 상태 인코딩, 편향 계산, 더미 데이터 생성 등의 유틸리티 함수를 제공합니다.

- CSV 파일 로드 및 초기화
- 사용자 특성의 원-핫 인코딩
- 초기 편향 계산 (CSV 및 감정 상태 기반)
- 현재 시간 및 계절 카테고리 반환
- 더미 사용자 생성

```python
# 상태 인코딩 예시
def encode_state(state):
    """사용자 특성 딕셔너리를 원-핫 인코딩된 벡터로 변환"""
    # 각 카테고리 원-핫 인코딩
    age_vec = one_hot_encode(state["age"], AGE_CATEGORIES)
    gender_vec = one_hot_encode(state["gender"], GENDER_CATEGORIES)
    emotion_vec = one_hot_encode(state["emotion"], EMOTION_CATEGORIES)
    time_vec = one_hot_encode(state["time"], TIME_CATEGORIES)
    weather_vec = one_hot_encode(state["weather"], WEATHER_CATEGORIES)

    # 전체 상태 벡터 생성
    state_vector = np.concatenate([age_vec, gender_vec, emotion_vec, time_vec, weather_vec])
    return torch.FloatTensor(state_vector)
```

### 4. visualizer.py

추천 결과 및 학습 과정을 시각화하는 클래스입니다.

- Tkinter 기반 GUI 인터페이스
- 학습 그래프 (보상, 손실, 입실론)
- 추천 결과 시각화 (파이 차트, 막대 그래프)
- 사용자 특성별 보상 분석
- 시뮬레이션 제어 도구

### 5. main.py

전체 시스템의 메인 실행 파일입니다.

- AdSimulator 클래스: GUI 기반 시뮬레이터
- train_console_mode: 콘솔 모드에서 학습 실행
- evaluate_console_mode: 콘솔 모드에서 평가 실행
- 명령줄 인터페이스 및 인자 처리

### 6. debug.py

시스템 디버깅 및 의존성 체크를 위한 도구입니다.

- 필요한 모듈 체크 (PyTorch, NumPy, Pandas, Matplotlib, Tkinter)
- CSV 파일 존재 확인
- GUI 초기화 테스트

## 사용 방법

### 환경 설정

1. **Conda 환경 생성**:

   ```bash
   conda env create -f environment.yml
   conda activate Advise-ML
   ```

2. **수동 설치** (Conda 환경 파일이 없는 경우):
   ```bash
   conda create -n Advise-ML python=3.10
   conda activate Advise-ML
   conda install pytorch numpy pandas matplotlib
   pip install tqdm seaborn
   ```

### 실행 방법

AdVise-ML은 세 가지 모드로 실행할 수 있습니다:

1. **GUI 모드** (기본):

   ```bash
   python main.py
   # 또는
   python main.py --mode gui
   ```

2. **학습 모드** (콘솔):

   ```bash
   python main.py --mode train --users 100 --verbose
   ```

3. **평가 모드** (콘솔):
   ```bash
   python main.py --mode eval --users 20 --verbose
   ```

### 추가 옵션

- `--users N`: 학습/평가에 사용할 사용자 수 (기본값: 100)
- `--verbose`: 상세 로깅 활성화
- `--debug`: 디버그 모드 활성화

### GUI 사용법

1. **시뮬레이션 실행**:

   - "시뮬레이션 실행" 버튼 클릭
   - 사용자 수와 시뮬레이션 속도 선택 가능

2. **추천 결과 분석**:

   - "학습 그래프" 탭: 보상, 손실, 입실론 추이 확인
   - "추천 결과" 탭: 카테고리별 추천 비율 및 평균 보상 확인
   - "속성별 보상" 탭: 사용자 특성별 평균 보상 분석
   - "사용자 데이터" 탭: 개별 사용자 추천 이력 확인

3. **모델 관리**:
   - "모델 저장" 버튼: 현재 모델을 저장
   - "모델 로드" 버튼: 저장된 모델 로드

## 데이터 설명

AdVise-ML은 다음과 같은 CSV 데이터 파일을 사용합니다:

1. **category_weight_age.csv**: 연령별 광고 카테고리 가중치

   - 열: ['연령', '교양_오락_문화', '교육', '교통', '내구재', '외식', '의류']
   - 연령 카테고리: ["20세미만", "20-30세", "31-40세", "41-50세", "51-60세", "61-70세", "70세이상"]

2. **category_weight_sex.csv**: 성별 광고 카테고리 가중치

   - 열: ['성별', '교양_오락_문화', '교육', '교통', '내구재', '외식', '의류']
   - 성별 카테고리: ["남성", "여성"]

3. **category_weight_time.csv**: 시간대별 광고 카테고리 가중치

   - 열: ['시간대', '교양_오락_문화', '교육', '교통', '내구재', '외식', '의류']
   - 시간대 카테고리: ["오전", "오후"]

4. **category_weight_season.csv**: 계절별 광고 카테고리 가중치
   - 열: ['계절', '교양_오락_문화', '교육', '교통', '내구재', '외식', '의류']
   - 계절 카테고리: ["봄", "여름", "가을", "겨울"]

각 가중치 값은 0.3~1.0 사이의 값으로, 해당 특성(예: 20대)이 특정 광고 카테고리(예: 교육)에 대해 가지는 선호도를 나타냅니다. 파일이 없는 경우 시스템은 자동으로 랜덤 가중치로 파일을 생성합니다.

## 학습 과정 설명

AdVise-ML의 DQN 에이전트는 다음과 같은 과정으로 학습합니다:

1. **초기화**: 정책 네트워크와 타겟 네트워크 초기화

2. **사용자 특성 인코딩**: 사용자 특성을 원-핫 인코딩하여 상태 벡터 생성

3. **초기 편향 계산**: CSV 데이터 기반으로 초기 편향 계산 (사전 지식 주입)

4. **행동 선택**:

   - 입실론 확률로 무작위 행동 선택 (탐색)
   - 1-입실론 확률로 최대 Q-값 행동 선택 (활용)

5. **보상 관찰**: 사용자의 광고 응시 시간을 보상으로 받음

6. **경험 저장**: (state, action, reward, next_state, done) 튜플을 리플레이 버퍼에 저장

7. **배치 학습**:

   - 리플레이 버퍼에서 무작위 배치 샘플링
   - Double DQN 알고리즘으로 타겟 Q-값 계산
   - MSE 손실 함수로 정책 네트워크 업데이트

8. **타겟 네트워크 업데이트**: 일정 간격으로 정책 네트워크의 가중치를 타겟 네트워크로 복사

<p align="center">
  <img src="/images/image (2).png" alt="학습 과정 그래프" width="700">
  <br>
  <em>학습 과정: 보상 히스토리, 손실, 입실론 감소</em>
</p>

보상 히스토리 그래프를 보면 학습 초반에는 응시 시간이 불안정하게 0.54초 사이에서 변화하다가, 약 100스텝 이후부터 평균 응시 시간이 2.53초로 증가한 것을 확인할 수 있습니다. 이는 모델이 사용자 특성에 따라 적절한 광고를 추천하는 방법을 점진적으로 학습했음을 보여줍니다. 또한 입실론 값이 지속적으로 감소하면서 탐색보다는 활용에 중점을 두는 방향으로 변화함을 확인할 수 있습니다.

## 시뮬레이션 결과 분석

### 광고 카테고리별 추천 성과

<p align="center">
  <img src="/images/image (3).png" alt="광고 카테고리별 추천 성과" width="700">
  <br>
  <em>광고 카테고리별 추천 비율 및 평균 보상</em>
</p>

시뮬레이션 결과, 광고 카테고리 중에서는 Dining(외식)이 전체 추천의 25%로 가장 많았으며, Culture_Entertainment(20%), Transportation(15%) 순으로 나타났습니다. 카테고리별 평균 보상을 살펴보면 Transportation(3.11초), Culture_Entertainment(2.93초) 등이 높은 평균 응시 시간을 기록했습니다. 이는 해당 카테고리들이 사용자로부터 더 긴 응시 시간을 유도했음을 의미합니다.

### 사용자 특성별 추천 성과

<p align="center">
  <img src="/images/image (4).png" alt="사용자 특성별 추천 성과" width="700">
  <br>
  <em>사용자 특성별 평균 보상 분석</em>
</p>

사용자 특성별 추천 성과 분석 결과는 다음과 같습니다:

1. **연령대별**: 61-70세(3.13초), 20-30세(2.97초)가 가장 긴 응시 시간 기록
2. **감정별**: Happy(3.30초), Surprise(3.02초) 상태에서 높은 응시 시간 기록
3. **시간대/계절별**: 오후 시간대와 봄철에 모두 평균 2.94초의 응시 시간 기록

이러한 분석을 통해 다양한 사용자 특성에 따른 광고 선호도를 파악할 수 있으며, 특히 중장년층(51-60세)은 교통 및 문화 콘텐츠를, 20-30대는 외식 및 의류 광고를 선호하는 경향이 확인되었습니다.

## 확장 및 개선 가능성

AdVise-ML은 다음과 같은 방향으로 확장 및 개선할 수 있습니다:

1. **추가 특성 도입**:

   - 사용자 위치, 관심사, 과거 클릭 이력 등의 특성 추가
   - 텍스트/이미지 기반 광고 내용의 특성 반영

2. **알고리즘 개선**:

   - Prioritized Experience Replay 도입
   - Dueling DQN, Rainbow DQN 등 고급 알고리즘 적용
   - 분산 강화학습 (A3C, IMPALA 등) 도입

3. **실제 데이터 연동**:

   - 실제 사용자 데이터 및 광고 응답 데이터 사용
   - A/B 테스트 프레임워크 구축

4. **모델 해석력 강화**:

   - SHAP, LIME 등 설명 가능한 AI 기법 도입
   - 추천 결정에 대한 상세한 근거 제공

5. **다중 목표 최적화**:
   - 응시 시간뿐만 아니라 클릭률, 전환율 등 다양한 지표 고려
   - 다중 목표 강화학습 알고리즘 도입

## 기여 방법

AdVise-ML 프로젝트에 기여하고 싶으시다면 다음과 같은 방법으로 참여할 수 있습니다:

1. **이슈 제출**: 버그 보고 또는 기능 요청
2. **풀 리퀘스트(PR) 제출**: 코드 수정 또는 기능 추가
3. **데이터 기여**: 더 정확한 광고 카테고리 가중치 데이터 제공
4. **문서화**: README, 주석, 튜토리얼 개선

## 라이센스

Apache License 2.0

## 연락처

bmori3@naver.com

---

_이 프로젝트는 강화학습을 활용한 광고 추천 시스템의 개념 증명(PoC)을 위해 개발되었습니다._
