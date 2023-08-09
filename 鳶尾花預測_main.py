### 生物學案例 - 鳶尾花預測 ###
#%% (IV) Streamlit App 製作
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.preprocessing import scale, LabelEncoder # for scaling the data
import sklearn.metrics as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)

#%%# (I) 情境介紹與前處理 - 資料標準化
def set_vars(df):
    # 定義X變數 與目標變數y
    X = df.iloc[:, :-1] #:-1代表到最後一欄，但不包含最後
    y = df.iloc[:, -1] #-1代表最後一欄

    X_scaled = scale(X) #數據標準化

    le = LabelEncoder()
    y_encoded = le.fit_transform(y) #將不可量化的單位進行編碼
    
    return X, y, X_scaled, y_encoded

#%%## (II) 模型訓練與視覺化結果 - 模型建制與訓練
# shift + enter
def modeling(X_scaled):
    clustering = KMeans(n_clusters=3, random_state=42) #花種有3個
    clustering.fit(X_scaled) #fit the dataset
    return clustering
    
#%%# (II) 模型訓練與視覺化結果 - 預測結果
# F9
def get_prediction(X, clustering):
    # 預測結果
    X['prediction'] = clustering.labels_
    return X
    
#%%## (II) 模型訓練與視覺化結果 - 視覺化呈現: 預測
# shift + enter
def plot_reseult(X, y, y_encoded, predict=True):
    
    if predict:
        color_index = X.prediction
        title_ = " "
        
    else:
        color_index = y_encoded
        title_ = " "
          
    colors = np.array(["Red","Orange","Violet"])
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

    fig.add_trace(
        go.Scatter(
            x=X["petal_length"], y=X["petal_width"],
            name = "petal",
            mode='markers',
            marker_color=colors[color_index]
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=X["sepal_length"], y=X["sepal_width"],
            mode='markers',
            name = "sepal",
            marker_color=colors[color_index]
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text= f"<b>{title_}</b>",
        title_x=0.1,
        width=900,
        height=600,
        font=dict(
            family="Courier New, monospace",
            size=20,
        ),
    )

    return fig

#%%
def backend(df):
    
    # 特徵變數 & 目標變數：資料處理
    X, y, X_scaled, y_encoded = set_vars(df)
    
    # 建制並訓練 K-means 分群模型
    clustering = modeling(X_scaled)
    
    # 利用該模型進行分群預測
    X = get_prediction(X, clustering)
    
    # 繪製預測與實際的視覺圖
    predict_fig = plot_reseult(X, y, y_encoded, predict=True)
    actual_fig = plot_reseult(X, y, y_encoded, predict=False)
    
    return predict_fig, actual_fig


def upload():
    data_file = st.file_uploader("請上傳鳶尾花案例資料: IRIS.csv")
    if data_file is not None:
        return pd.read_csv(data_file, encoding='utf-8-sig')

def show_img():
    
    from PIL import Image
    image = Image.open('christina-brinza-TXmV4YYrzxg-unsplash.jpg')
    img_caption = "Photo by Christina Brinza on Unsplash"
    
    return st.image(image, caption=img_caption)
    
def main_page():

    # Frontend: Text
    st.markdown("""
                # 生物學案例 - 鳶尾花預測
                ##
                """, unsafe_allow_html=True)
    st.write("""
             **IRIS 資料集**是資料科學界中非常知名的資料集。常作為**分類問題**的招牌示範資料。
             在1936 年，英國統計學家和生物學家 Ronald Fisher 在論文中曾引入此多變量資料集。
             此資料是埃德加安德森收集三種鳶尾花種的不同特徵，因此有時作安德森鳶尾花資料集。
             該資料集包含 **三種鳶尾花（山鳶尾、變色鳶尾和維吉尼亞鳶尾）** 各 50 個樣本。
             從每個樣本中測量四種特徵：**萼片和花瓣的長度和寬度**，以厘米為單位。
             """, unsafe_allow_html=True)
    st.text("")
    
    # Frontend: col 排版
    col1, col2 = st.columns([3,2])
    
    with col1:
        st.write("""**分析目標: 從萼片與花瓣長寬度預測花種**""")
        df = upload()
        
    with col2:
        show_img()

    if df is not None:
        
        # frontend: 上傳資料呈現
        st.markdown("### 資料樣貌")
        with st.expander("前五筆資料"):  
            st.dataframe(data=df.head())
        
        # backend
        predict_fig, actual_fig = backend(df)
        
        # frontend: 視覺化預測結果
        st.markdown("### <br> K-means 預測結果<br><br>", unsafe_allow_html=True)
        st.markdown("#### 預測")
        st.plotly_chart(predict_fig, use_container_width = True)
        st.markdown("#### 實際")
        st.plotly_chart(actual_fig, use_container_width = True)

        st.success("恭喜成功!")


if __name__ == "__main__":
    st.set_page_config(page_title="生物學案例 - 鳶尾花預測", page_icon=":hibiscus:")
    main_page()
    