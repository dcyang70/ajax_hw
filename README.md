# 使用 Restful API 與 Ajax 來進行機器學習模型訓練與預測

##
Python 3.12.10

## 安裝uv
Windows PowerShell  
`
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
`

## 安裝套件
- Flask==3.1.3
- pandas==3.0.3
- scikit-learn==1.9.0
- numpy==2.5.0
- joblib==1.5.3
...

## 執行方法
cd titanic/  
uv run init_data.py  
uv run app.py  
打開瀏覽器, 輸入: http://127.0.0.1:5000

## 說明
目前使用 GridsearchCV 搭配 隨機森林（Random Forest）分類器,  
鐵達尼號有很多類別型欄位（如登船港口、性別）與缺失值（如年齡），  
隨機森林對這些特徵的包容度高，不容易過擬合（Overfitting,  
且不需要太複雜的特徵縮放。   
 - 內部調整的超參數有: n_estimator, max_depth 和 min_split_node  
 - 目前暫時調整出超參數: n_estimators: 50, max_depth: 100, min_samples_split: 20

## 成果
#### 過程截圖
1. 起始頁  
   <img src="/screenshot/01.jpg" width="50%" alt="起始頁">
2. 主執行頁面  
   <img src="/screenshot/02.jpg" width="50%" alt="起始頁">
3. 參考頁  
   <img src="/screenshot/03.jpg" width="50%" alt="起始頁">

#### Demo  

&emsp;<a href="https://drive.google.com/file/d/1nYqfEPrdjfdQuYTcKvUvmpZqyEbEpN-1/view?usp=sharing">
        <img src="/screenshot/start-page.jpg" width="400" alt="播放Demo影片">
      </a>


## 其它
N/A
