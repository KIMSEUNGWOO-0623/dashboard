import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt
import panel as pn
import os
import base64

#12월 일사량 누적, 온도, 풍속 평균 그래프 시각화
# --------------------------
# 1. EPW 파일 불러오기
# --------------------------
epw_path = 'data/KOR_KG_Seoul-Seongnam.AP.471110_TMYx.2004-2018/KOR_KG_Seoul-Seongnam.AP.471110_TMYx.2004-2018.epw'
df = pd.read_csv(epw_path, skiprows=8, header=None)

# --------------------------
# 2. 날짜 및 시간 처리
# --------------------------
df['Month'] = df[1]
df['Day'] = df[2]
df['Hour'] = df[3] - 1  # EPW의 시간은 1~24로 표기 → 0~23으로 보정

# 날짜 컬럼 추가 (선택적, 타임스탬프 만들기용)
df['Datetime'] = pd.to_datetime({
    'year': 2023,
    'month': df['Month'],
    'day': df['Day'],
    'hour': df['Hour']
}, errors='coerce')

# 1. 주요 기상 변수 추출
df['GHI_kWh'] = df[13] / 1000
df['Temp'] = df[6]

# 2. 데이터 필터링 (2010년 12월 데이터)
dec_2010 = df[(df[0] == 2010) & (df['Month'] == 12)].copy()

# 3. 시간 필터링 (06시~18시 사용)
hourly_avg_filtered = dec_2010[(dec_2010['Hour'] >= 6) & (dec_2010['Hour'] <= 18)].groupby('Hour')[['GHI_kWh', 'Temp']].mean()

# 4. X축 Label
hour_labels = [f"{hour:02d}:00" for hour in hourly_avg_filtered.index]

# 5. 그래프 시각화
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Bar plot (Radiation)
fig.add_trace(
    go.Bar(
        x=hour_labels,
        y=hourly_avg_filtered['GHI_kWh'],
        name='Radiation (kWh/m²)',
        marker_color='gray',
        opacity=0.8  # Bar 투명도 추가
    ),
    secondary_y=False,
)

# Temperature (Temp → 점+선)
fig.add_trace(
    go.Scatter(
        x=hour_labels,
        y=hourly_avg_filtered['Temp'],
        name='Temperature (°C)',
        mode='lines+markers',
        line=dict(color='skyblue', width=3)  # 라인 두께 3로 설정
    ),
    secondary_y=True,
)

# Layout 설정
fig.update_layout(
    legend=dict(
        x=0.85,
        y=0.98,
        xanchor='left',
        yanchor='top',
        font=dict(family="Noto Sans KR Medium", size=20, color="black"),
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="black",
        borderwidth=0
    ),
    margin=dict(l=80, r=80, t=120, b=120),
    plot_bgcolor='white',
    paper_bgcolor='whitesmoke'
)

# X축 설정
fig.update_xaxes(
    title_text="Time",
    type='category',
    tickangle=0,
    showgrid=True,
    title_font=dict(size=20, family="Noto Sans KR Medium", color="black"),
    tickfont=dict(size=18, family="Noto Sans KR Medium", color="black")  # 글자 크기 조절 추가
)

# Y축 Radiation
fig.update_yaxes(
    title_text="Radiation (kWh/m²)",
    title_font=dict(size=20, family="Noto Sans KR Medium", color="black"),
    tickfont=dict(size=15, family="Noto Sans KR Medium", color="black"),  # 여기 추가!
    secondary_y=False,
    range=[0, 0.5],
    showgrid=True,
    gridcolor='rgba(0,0,0,0.5)',
    gridwidth=1,
    tickvals=[round(x * 0.05, 8) for x in range(11)],
    ticktext=[f"{round(x * 0.05, 2):.2f}" for x in range(11)],
    tickmode='array'
)

# Y축 Temp → grid 표시 안함
fig.update_yaxes(
    title_text="Temperature (°C)",
    title_font=dict(size=20, family="Noto Sans KR Medium", color="black"),
    tickfont=dict(size=15, family="Noto Sans KR Medium", color="black"),  # 여기 추가!
    secondary_y=True,
    range=[-4, 2],
    showgrid=False
)

# Show the plot
fig.show()

# Save as HTML
#fig.write_html("Data Analysis.html")


pn.extension()

# ① 폴더 경로
folder_path = r"C:\Users\USER\Desktop\#캡스톤\#패널\radiation analysis"

# ② 총 프레임 수
num_frames = 372

# ③ base64 변환 함수
def load_base64_png(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# ④ base64 이미지 리스트 만들기
frame_images = []
for i in range(1, num_frames + 1):
    file_path = os.path.join(folder_path, f"frame ({i}).png")  # 공백 주의 → 실제 파일명 확인 필요
    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
    else:
        frame_images.append(load_base64_png(file_path))

# ⑤ 초기 이미지 Pane → HTML 사용
img_pane = pn.pane.HTML(f'<img src="{frame_images[0]}" width="800">')

# ⑥ 슬라이더
current_frame = pn.widgets.IntSlider(name='Frame', start=1, end=len(frame_images), step=1)

# ⑦ Play 상태 관리
playing = [False]

# ⑧ 이미지 업데이트 함수
def update_image(event):
    frame_idx = event.new
    img_pane.object = f'<img src="{frame_images[frame_idx - 1]}" width="800">'

current_frame.param.watch(update_image, 'value')

# ⑨ Prev button
def prev_callback(event):
    if current_frame.value > 1:
        current_frame.value -= 1

# ⑩ Next button
def next_callback(event):
    if current_frame.value < len(frame_images):
        current_frame.value += 1

# ⑪ Play/Pause button
def animate():
    if playing[0]:
        if current_frame.value < len(frame_images):
            current_frame.value += 1
        else:
            current_frame.value = 1

prev_button = pn.widgets.Button(name='⏮ Prev', button_type='primary')
prev_button.on_click(prev_callback)

next_button = pn.widgets.Button(name='Next ⏭', button_type='primary')
next_button.on_click(next_callback)

play_button = pn.widgets.Toggle(name='▶ Play / ⏸ Pause', button_type='success')

cb = pn.state.add_periodic_callback(animate, period=500)    # period: frmae당 시간
cb.stop()

def play_callback(event):
    if event.new:
        playing[0] = True
        cb.start()
    else:
        playing[0] = False
        cb.stop()

play_button.param.watch(play_callback, 'value')

# ⑫ 레이아웃 구성
controls = pn.Row(prev_button, play_button, next_button)
layout = pn.Column(current_frame, controls, img_pane)

# ⑬ HTML 저장 → base64라서 HTML 파일 하나로 동작 가능!
#layout.save("animation_buttons_panel_base64_FINAL.html", embed=True)

# ⑭ Panel serve로 확인도 가능
layout.servable()

# app.py
import dash
from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ① Plotly 그래프 → 이미 fig로 확보됨 → 여기서는 예시로 다시 정의
# → 당신은 기존 코드에서 fig 변수를 그대로 import 하거나 복붙하면 됨
# → 여기서는 예시용으로 다시 구성 (생략 가능, 그냥 기존 fig 사용 가능)

# hour_labels, ghi_values, temp_values → 기존 코드에서 그대로 사용 가능
hour_labels = ['06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
               '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00']

# 예시 데이터 → 기존 코드의 hourly_avg_filtered['GHI_kWh'], ['Temp'] 사용 가능
ghi_values = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.22, 0.20, 0.15, 0.10, 0.05, 0.02, 0.0]
temp_values = [-5, -4, -3, -1, 0, 2, 3, 2, 1, 0, -1, -3, -4]

# fig 구성
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=hour_labels, y=ghi_values, name='Radiation (kWh/m²)', marker_color='gray', opacity=0.8), secondary_y=False)
fig.add_trace(go.Scatter(x=hour_labels, y=temp_values, name='Temperature (°C)', mode='lines+markers', line=dict(color='skyblue', width=3)), secondary_y=True)
fig.update_layout(title="12월 시간대별 Radiation & Temperature", margin=dict(l=80, r=80, t=120, b=120))
fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="Radiation (kWh/m²)", secondary_y=False)
fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)

# ② Dash 앱 구성
app = dash.Dash(__name__)

app.layout = html.Div([

    # 제목 + 제작자
    html.Div([
        html.H1("디지털트윈을 활용한 빙판길 사고 예방", style={'textAlign': 'center'}),
        html.H3("스마트시티융합학과 202335220 김승우", style={'textAlign': 'center', 'color': 'gray'}),
    ], style={'margin-bottom': '30px'}),

    # 구성1: Animation Iframe
    html.Div([
        html.H2("구성 1. 일사량 분석 모델링 (애니메이션)", style={'textAlign': 'left'}),
        
        html.Iframe(src='animation_buttons_panel_base64_FINAL.html', width='100%', height='700px'),

        html.H3("추가 이미지 예시", style={'margin-top': '20px'}),
        
        html.Div([
            html.Img(src='assets/sample1.png', style={'width': '32%', 'padding': '5px'}),
            html.Img(src='assets/sample2.png', style={'width': '32%', 'padding': '5px'}),
            html.Img(src='assets/sample3.png', style={'width': '32%', 'padding': '5px'}),
        ], style={'display': 'flex', 'justify-content': 'center'})
    ], style={'margin-bottom': '50px'}),

    # 구성2: EPW 기반 그래프
    html.Div([
        html.H2("구성 2. EPW 데이터를 바탕으로 한 12월 시간별 일사량 누적 평균 & 기온 평균", style={'textAlign': 'left'}),
        dcc.Graph(figure=fig)
    ])
])

# ③ 실행
if __name__ == '__main__':
    app.run(debug=True, port=8051)
