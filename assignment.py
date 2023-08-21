import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 

#%%
# 設定頁面顯示
## 請用st.set_page_config()來設定頁面顯示
## 若忘記可參考單元6


#%%
# 設定左側邊欄選單頁面
menu = st.sidebar.selectbox("Menu",["Home","Plot"])

## 請於Home頁面設定title為"我的第一個Streamlit Web App"
## 請將Home裡面放入「assignment_home.py」的內容
## 並將'常用streamlit基礎功能'、'終端機Streamlit指令'、'Streamlit的動態視覺圖'分為三欄作呈現
## 提示: 利用st.columns()及with col1, with col2...

## 請將Plot裡面放入「02_app_plotly.py」的內容
## 提示：利用if, else來作設定