# 환경 설정 및 라이브러리 임포트
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import holidays
import shap

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_error

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 1. 데이터 로드
def load_data(path):
    df = pd.read_csv(path, parse_dates=['FL_DATE'], encoding='utf-8')
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df

# 2. 데이터 전처리
def report_missing(df, threshold=0.0):
    missing = df.isna().mean()*100
    miss = missing[missing > threshold].sort_values(ascending=False)
    if not miss.empty:
        print("\n=== Missing Values (%) ===\n", miss)

def correct_time(df, col):
    if col in df.columns:
        df[col] = df[col].astype(str).str.zfill(4)
        df[col] = df[col].str[:4].apply(lambda x: int(x[:2])*60 + int(x[2:]))
    return df

def add_date_features(df, country='US'):
    df['WEEKDAY'] = df['FL_DATE'].dt.weekday
    df['IS_WEEKEND'] = df['WEEKDAY'] >= 5
    cal = holidays.CountryHoliday(country)
    df['IS_HOLIDAY'] = df['FL_DATE'].isin(cal)
    df['MONTH'] = df['FL_DATE'].dt.month
    df['QUARTER'] = df['FL_DATE'].dt.quarter
    df['YEAR'] = df['FL_DATE'].dt.year
    # season: 1=Winter, 2=Spring, 3=Summer, 4=Fall
    df['SEASON'] = ((df['MONTH']%12 + 3)//3).map({1:'Winter',2:'Spring',3:'Summer',4:'Fall'})
    return df

# 데이터 로드 및 전처리 실행
df = load_data('your_data.csv') # 실제 데이터 경로로 변경해주세요
report_missing(df)
time_cols = ['DEP_TIME', 'SCHED_DEP_TIME', 'ARR_TIME', 'SCHED_ARR_TIME']
for col in time_cols:
    df = correct_time(df, col)
df = add_date_features(df)

# A. 개별 항공편 지연 요인 분석
print("\n--- A. 개별 항공편 지연 요인 분석 ---")

# 3.1 피처 및 타겟 설정 (ARR_DELAY 예측)
features_A = [col for col in df.columns if col not in ['ARR_DELAY','FL_DATE','ORIGIN','DEST','AIRLINE', 'ARR_TIME', 'DEP_TIME']] # 시간 관련 컬럼 제거
X_A = df[features_A]
y_A = df['ARR_DELAY']

# 3.2 데이터 분할
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, random_state=42)

# 3.3 파이프라인 & 모델 정의
num_feats = X_A.select_dtypes(include=['int64','float64']).columns.tolist()
cat_feats = ['AIRLINE','ORIGIN','DEST','SEASON', 'WEEKDAY', 'IS_WEEKEND', 'IS_HOLIDAY', 'MONTH', 'QUARTER', 'YEAR'] # 범주형 변수 추가

preproc_A = ColumnTransformer([
    ('num', StandardScaler(), num_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_feats)
])

models_A = {
    'CatBoost': CatBoostRegressor(verbose=0, random_state=42),
    'RandomForest': RandomForestRegressor(n_jobs=-1, random_state=42),
    'XGBoost': XGBRegressor(n_jobs=-1, random_state=42, verbosity=0, random_state=42),
    'LightGBM': LGBMRegressor(n_jobs=-1, random_state=42)
}

# 3.4 학습 및 평가
results_A = {}
for name, mdl in models_A.items():
    pipe = Pipeline([('pre', preproc_A), ('model', mdl)])
    pipe.fit(X_train_A, y_train_A)
    preds = pipe.predict(X_test_A)
    results_A[name] = {
        'MSE': mean_squared_error(y_test_A, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test_A, preds)),
        'MAE': mean_absolute_error(y_test_A, preds),
        'R2': r2_score(y_test_A, preds)
    }
results_A_df = pd.DataFrame(results_A).T
print("\n=== A 분석 모델별 성능 ===\n", results_A_df)

# 3.5 SHAP 분석 (모든 모델에 대해 수행)
shap_values_A = {}
for name, mdl in models_A.items():
    pipe = Pipeline([('pre', preproc_A), ('model', mdl)])
    pipe.fit(X_train_A, y_train_A)
    explainer = shap.TreeExplainer(pipe[-1]) # 파이프라인의 마지막 단계 (모델)
    shap_values = explainer.shap_values(pipe[:-1].transform(X_test_A)) # 전처리된 데이터 사용
    shap_values_A[name] = shap_values
    print(f"\n=== {name} SHAP Feature Importance ===")
    shap.summary_plot(shap_values, pipe[:-1].transform(X_test_A), feature_names=pipe[:-1].get_feature_names_out(), plot_type='bar')
    plt.show()

# B. 공항/항공사 네트워크 기반 지연 패턴 분석
print("\n--- B. 공항/항공사 네트워크 기반 지연 패턴 분석 ---")

# 4.1 네트워크 그래프 구축 (공항 노드, 편 방향 엣지)
G_airport = nx.DiGraph()
# 공항 노드 추가
all_airports = pd.concat([df['ORIGIN'], df['DEST']]).unique()
for airport in all_airports:
    G_airport.add_node(airport)

# 엣지: ORIGIN->DEST, weight=평균 DEP_DELAY
edge_df_airport = df.groupby(['ORIGIN','DEST'])['DEP_DELAY'].mean().reset_index()
for _, row in edge_df_airport.iterrows():
    if row['ORIGIN'] in G_airport and row['DEST'] in G_airport:
        G_airport.add_edge(row['ORIGIN'], row['DEST'], weight=row['DEP_DELAY'])

# 4.2 중심성 계산 (공항 네트워크)
deg_cent_airport = nx.degree_centrality(G_airport)
pagerank_airport = nx.pagerank(G_airport, weight='weight', max_iter=500, tol=1e-5) # max_iter, tol 추가
btw_cent_airport = nx.betweenness_centrality(G_airport, weight='weight')
close_cent_airport = nx.closeness_centrality(G_airport, distance='weight')
eigen_cent_airport = nx.eigenvector_centrality(G_airport, weight='weight', max_iter=500, tol=1e-5)

# 데이터프레임으로 정리 (공항)
net_df_airport = pd.DataFrame({
    'airport': list(G_airport.nodes()),
    'degree_centrality': [deg_cent_airport.get(n, 0) for n in G_airport.nodes()],
    'pagerank': [pagerank_airport.get(n, 0) for n in G_airport.nodes()],
    'betweenness': [btw_cent_airport.get(n, 0) for n in G_airport.nodes()],
    'closeness': [close_cent_airport.get(n, 0) for n in G_airport.nodes()],
    'eigenvector': [eigen_cent_airport.get(n, 0) for n in G_airport.nodes()]
})
net_df_airport_sorted = net_df_airport.sort_values('pagerank', ascending=False).head(10)
print("\n=== 공항 네트워크 중심성 상위 10개 공항 ===\n", net_df_airport_sorted)

# 4.1.2 항공사 네트워크 그래프 구축 (항공사 노드, 편 방향 엣지 - 출발 공항 -> 도착 공항)
G_airline = nx.DiGraph()
all_airlines = df['AIRLINE'].unique()
for airline in all_airlines:
    G_airline.add_node(airline)

edge_df_airline = df.groupby(['AIRLINE', 'ORIGIN', 'DEST'])['DEP_DELAY'].mean().reset_index()
airline_delay_impact = edge_df_airline.groupby('AIRLINE')['DEP_DELAY'].mean().sort_values(ascending=False)
print("\n=== 항공사별 평균 출발 지연 ===\n", airline_delay_impact.head(10))

# (추가 분석: 항공사 간 연결 - 동일한 공항 쌍을 운항하는 경우)
airline_connections = df.groupby(['ORIGIN', 'DEST'])['AIRLINE'].nunique().reset_index()
print("\n=== 공항 쌍별 운항 항공사 수 상위 10개 ===\n", airline_connections.sort_values(by='AIRLINE', ascending=False).head(10))


# 4.3 지연 전파 시뮬레이션 (단계적 전파 예시 - 공항 네트워크)
def simulate_delay_airport(origin, steps=3):
    current = {origin: 1.0}  # 초기 활성화 (영향력)
    sim_delay = {origin: df[df['ORIGIN']==origin]['DEP_DELAY'].mean()}
    for step in range(steps):
        new_current = {}
        step_delays = {}
        for src, act in current.items():
            for nbr in G_airport.successors(src):
                weight = G_airport[src][nbr].get('weight', 0) # 엣지 가중치 (평균 출발 지연)
                delay_contribution = act * weight
                sim_delay[nbr] = sim_delay.get(nbr, 0) + delay_contribution
                new_current[nbr] = new_current.get(nbr, 0) + act
                step_delays[(src, nbr)] = delay_contribution # 엣지별 지연 영향 저장
        current = new_current
        print(f"\n=== Step {step+1} 전파 결과 (Origin: {origin}) ===")
        print(pd.Series(step_delays).sort_values(ascending=False).head(10))
    return pd.Series(sim_delay).sort_values(ascending=False)

# 예시: JFK에서 3단계 전파 시뮬레이션
print("\n=== JFK 공항 출발 지연 전파 시뮬레이션 (3단계) ===\n", simulate_delay_airport('JFK').head(10))

# C. 정보 중요도 분석 (A, B 결과 통합)
print("\n--- C. 정보 중요도 분석 (A, B 결과 통합) ---")

# 5.1 A 결과 (SHAP)에서 상위 특성 추출
shap_importance_all = {}
for name, shap_val in shap_values_A.items():
    feature_names = preproc_A.get_feature_names_out()
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_abs_mean': np.abs(shap_val).mean(axis=0)
    }).sort_values('shap_abs_mean', ascending=False)
    shap_importance_all[name] = shap_importance.head(10)
    print(f"\n=== {name} 모델 SHAP 상위 10개 특성 ===\n", shap_importance_all[name])

# 5.2 B 결과 (네트워크)에서 상위 노드 메트릭
print("\n=== 공항 네트워크 PageRank 상위 10개 공항 ===\n", net_df_airport.sort_values('pagerank', ascending=False).head(10))

# 5.3 통합: SHAP 특성 중 네트워크 관련 특성이 포함되는지 확인
print("\n=== SHAP 상위 특성 중 공항 관련 특성 포함 여부 ===\n")
for name, shap_df in shap_importance_all.items():
    airport_features_shap = shap_df[shap_df['feature'].str.contains('ORIGIN_|DEST_')]
    print(f"--- {name} ---")
    if not airport_features_shap.empty:
        print(airport_features_shap)
    else:
        print("공항 관련 특성 없음")

# 추가적인 통합 분석 (예시): PageRank 상위 공항의 지연 예측 시 SHAP 값 분석
top_pagerank_airports = net_df_airport.sort_values('pagerank', ascending=False)['airport'].head(5).tolist()
print(f"\n=== PageRank 상위 5개 공항: {top_pagerank_airports} ===")

# 특정 모델 (CatBoost)에 대해 PageRank 상위 공항 관련 SHAP 값 살펴보기
if 'CatBoost' in shap_values_A:
    catboost_shap_df = pd.DataFrame(shap_values_A['CatBoost'], columns=preproc_A.get_feature_names_out())
    for airport in top_pagerank_airports:
        origin_shap_cols = [col for col in catboost_shap_df.columns if f'ORIGIN_{airport}' in col]
        dest_shap_cols = [col for col in catboost_shap_df.columns if f'DEST_{airport}' in col]
        if origin_shap_cols:
            print(f"\n=== CatBoost 모델에서 {airport} 출발 관련 SHAP 값 (평균) ===\n", catboost_shap_df[origin_shap_cols].mean().sort_values(ascending=False).head(5))
        if dest_shap_cols:
            print(f"\n=== CatBoost 모델에서 {airport} 도착 관련 SHAP 값 (평균) ===\n", catboost_shap_df[dest_shap_cols].mean().sort_values(ascending=False).head(5))
