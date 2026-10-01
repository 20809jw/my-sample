import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="서울 기온 예측기", page_icon="🌡️", layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.markdown(
    "서울의 과거 기온 데이터를 바탕으로 선형 회귀 모델을 생성하고, 선택한 연도의 기온을 예측합니다."
)


# 1. 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")

    # '날짜' 열을 datetime 타입으로 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 1) 관측일 수가 300일 이상인 연도만 필터링
    valid_years = (
        df.groupby("연도")["평균기온"].count().reset_index(name="관측일수")
    )
    valid_years = valid_years[valid_years["관측일수"] >= 300]["연도"]
    df_filtered = df[df["연도"].isin(valid_years)].copy()

    # 연도별 평균기온 계산
    yearly_df = (
        df_filtered.groupby("연도")["평균기온"].mean().reset_index()
    )

    # 2) 2025년 이하 데이터만 필터링 (수업 기준 기간)
    yearly_df = yearly_df[yearly_df["연도"] <= 2025].copy()

    # 독립변수 X: 1908년부터 경과한 연수 (연도 - 1908)
    yearly_df["경과연수"] = yearly_df["연도"] - 1908

    return yearly_df


# 데이터 불러오기
yearly_data = load_and_preprocess_data()

# 데이터 기본 정보 산출
num_years = len(yearly_data)
start_year = int(yearly_data["연도"].min())
end_year = int(yearly_data["연도"].max())

# 2. 선형 회귀 계산 (1908년 기준 경과연수를 독립변수로 설정)
X = yearly_data["경과연수"].values
y = yearly_data["평균기온"].values

# 1차 선형 회귀 계수 산출 (y = slope * X + intercept)
slope, intercept = np.polyfit(X, y, 1)

# Pearson 상관계수 계산
corr = np.corrcoef(yearly_data["연도"], y)[0, 1]

# 3. 데이터 요약 통계 출력
st.sidebar.header("📊 분석 데이터 정보")
st.sidebar.info(
    f"""
- **학습 데이터 수**: {num_years}개 연도
- **분석 시작 연도**: {start_year}년
- **분석 끝 연도**: {end_year}년
- **상관계수(r)**: `{corr:.4f}`
"""
)

# 4. 연도 선택 슬라이더 및 예측값 출력
st.subheader("🔮 예측 연도 선택")
target_year = st.slider(
    "예측하고 싶은 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2030,
    step=1,
)

# 선택 연도의 경과연수 계산 및 예측
target_elapsed = target_year - 1908
predicted_temp = slope * target_elapsed + intercept

col1, col2 = st.columns([1, 2])
with col1:
    st.metric(
        label=f"🎯 {target_year}년 예상 평균기온",
        value=f"{predicted_temp:.2f} °C",
    )

# 5. Plotly 그래프 생성
# 회귀선 데이터를 위한 연도 범위 생성
plot_years = np.arange(1900, 2101)
plot_elapsed = plot_years - 1908
plot_pred = slope * plot_elapsed + intercept

fig = go.Figure()

# 산점도 (실제 관측 데이터)
fig.add_trace(
    go.Scatter(
        x=yearly_data["연도"],
        y=yearly_data["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=8, color="#1f77b4"),
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=plot_pred,
        mode="lines",
        name="선형 회귀선",
        line=dict(color="#ff7f0e", width=2),
    )
)

# 선택한 연도 예측 포인트 강조 표시
fig.add_trace(
    go.Scatter(
        x=[target_year],
        y=[predicted_temp],
        mode="markers+text",
        name="선택 연도 예측값",
        marker=dict(size=14, color="red", symbol="star"),
        text=[f"{predicted_temp:.2f}°C"],
        textposition="top center",
    )
)

fig.update_layout(
    title="서울 연도별 평균기온 및 선형 회귀 예측",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(range=[1895, 2105]),
    hovermode="x unified",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)

# 6. 기준 요약 정보 표시
st.markdown("---")
st.markdown(
    f"📌 **모델 학습 기준 정보**: 총 **{num_years}개**의 연도 데이터를 사용하여 회귀 모델을 만들었습니다. (분석 기간: **{start_year}년 ~ {end_year}년**)"
)
