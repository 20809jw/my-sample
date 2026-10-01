import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="서울 기온 예측기", page_icon="🌡️", layout="wide"
)

st.title("🌡️ 서울 기온 예측 및 온난화 속도 비교")
st.markdown(
    "서울의 연평균기온 데이터를 바탕으로 전체 기간과 최근 20년 동안의 기온 상승 속도(100년당 °C)를 비교합니다."
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

# 2. 회귀선 계산

# (1) 전체 기간 회귀 모델
X_full = yearly_data["경과연수"].values
y_full = yearly_data["평균기온"].values
slope_full, intercept_full = np.polyfit(X_full, y_full, 1)

# (2) 최근 20년 회귀 모델 (최대 연도 기준 최근 20개 연도)
recent_20_data = yearly_data.sort_values("연도").tail(20)
X_recent = recent_20_data["경과연수"].values
y_recent = recent_20_data["평균기온"].values
slope_recent, intercept_recent = np.polyfit(X_recent, y_recent, 1)

# 100년당 기온 변화량 (°C / 100년)
rate_full_100y = slope_full * 100
rate_recent_100y = slope_recent * 100

# Pearson 상관계수 (전체 기간)
corr_full = np.corrcoef(yearly_data["연도"], y_full)[0, 1]

# 3. 주요 지표(100년당 기온 상승량 비교) 메트릭 출력
st.subheader("🔥 기온 상승 속도 비교 (100년 기준)")
m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.metric(
        label=f"🌐 전체 기간 ({start_year}~{end_year}) 상승 속도",
        value=f"{rate_full_100y:+.2f} °C / 100년",
        help="전체 분석 기간의 연평균기온 추세를 100년 단위 변화량으로 환산한 값입니다.",
    )

with m_col2:
    st.metric(
        label=f"⚡ 최근 20년 ({recent_20_data['연도'].min()}~{end_year}) 상승 속도",
        value=f"{rate_recent_100y:+.2f} °C / 100년",
        delta=f"{rate_recent_100y - rate_full_100y:+.2f} °C (전체 대비)",
        help="최근 20년간의 데이터로만 계산한 100년당 기온 변화율입니다.",
    )

with m_col3:
    st.metric(
        label="📊 전체 기간 상관계수(r)",
        value=f"{corr_full:.4f}",
    )

st.markdown("---")

# 4. 연도 선택 슬라이더 및 예측값 출력
st.subheader("🔮 연도별 기온 예측")
target_year = st.slider(
    "예측하고 싶은 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2030,
    step=1,
)

target_elapsed = target_year - 1908
pred_full = slope_full * target_elapsed + intercept_full
pred_recent = slope_recent * target_elapsed + intercept_recent

p_col1, p_col2 = st.columns(2)
with p_col1:
    st.metric(
        label=f"🎯 {target_year}년 예상 기온 (전체 기간 모델 기준)",
        value=f"{pred_full:.2f} °C",
    )
with p_col2:
    st.metric(
        label=f"🚀 {target_year}년 예상 기온 (최근 20년 모델 기준)",
        value=f"{pred_recent:.2f} °C",
        delta=f"{pred_recent - pred_full:+.2f} °C (전체 모델 대비)",
    )

# 5. Plotly 그래프 생성
plot_years = np.arange(1900, 2101)
plot_elapsed = plot_years - 1908

plot_pred_full = slope_full * plot_elapsed + intercept_full
plot_pred_recent = slope_recent * plot_elapsed + intercept_recent

fig = go.Figure()

# 산점도 (전체 관측 데이터)
fig.add_trace(
    go.Scatter(
        x=yearly_data["연도"],
        y=yearly_data["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=8, color="#1f77b4", opacity=0.7),
    )
)

# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=plot_pred_full,
        mode="lines",
        name=f"전체 기간 회귀선 ({rate_full_100y:+.2f}°C/100년)",
        line=dict(color="#ff7f0e", width=3),
    )
)

# 최근 20년 회귀선
fig.add_trace(
    go.Scatter(
        x=plot_years,
        y=plot_pred_recent,
        mode="lines",
        name=f"최근 20년 회귀선 ({rate_recent_100y:+.2f}°C/100년)",
        line=dict(color="#d62728", width=3, dash="dash"),
    )
)

# 선택 연도 예측값 포인트 강조 (전체 모델)
fig.add_trace(
    go.Scatter(
        x=[target_year],
        y=[pred_full],
        mode="markers+text",
        name="전체 모델 예측점",
        marker=dict(size=12, color="#ff7f0e", symbol="star"),
        text=[f"{pred_full:.2f}°C"],
        textposition="top center",
    )
)

# 선택 연도 예측값 포인트 강조 (최근 20년 모델)
fig.add_trace(
    go.Scatter(
        x=[target_year],
        y=[pred_recent],
        mode="markers+text",
        name="최근 20년 모델 예측점",
        marker=dict(size=12, color="#d62728", symbol="diamond"),
        text=[f"{pred_recent:.2f}°C"],
        textposition="bottom center",
    )
)

fig.update_layout(
    title="서울 연도별 평균기온 및 추세선 비교 (전체 기간 vs 최근 20년)",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(range=[1895, 2105]),
    hovermode="x unified",
    template="plotly_white",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    ),
)

st.plotly_chart(fig, use_container_width=True)

# 6. 기준 요약 정보 표시
st.markdown("---")
st.markdown(
    f"📌 **모델 학습 기준 정보**: 총 **{num_years}개**의 연도 데이터를 사용했습니다. (분석 기간: **{start_year}년 ~ {end_year}년**)"
)
