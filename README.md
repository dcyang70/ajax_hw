# 使用 Restful API 與 Ajax 來進行機器學習模型訓練與預測

## 安裝套件
- Flask==3.1.3
- pandas==3.0.3
- scikit-learn==1.9.0
- numpy==
(版本號可用 pip list，或是 conda list 來檢視，上面是範例，你的可能跟我不一樣，請自行調整)
...

## 執行方法
python init_db.py
python app.py
(或是其它你覺得比較好的執行方式，請自行調整)

## 說明
(介紹你使用的模型，理想的超參數是哪些，剩下可以自由發揮)
隨機森林（Random Forest）
鐵達尼號有很多類別型欄位（如登船港口、性別）與缺失值（如年齡），
隨機森林對這些特徵的包容度高，不容易過擬合（Overfitting）, 且不需要太複雜的特徵縮放。
內部調整的超參數有:n_estimator, max_depth 和 min_split_node

## 成果
![](執行過程的擷圖或說明圖片)
...
[影片名稱或其它標題](你的影片連結)
...

## 其它你想要補充的資訊
...x
