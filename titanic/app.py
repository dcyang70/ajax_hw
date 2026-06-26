import sqlite3
from flask import Flask, jsonify, request, render_template
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import time

app = Flask(__name__)

DATABASE = 'my_db.db'
db = sqlite3.connect(DATABASE, check_same_thread=False)
# sqlite3 用單一執行續
# 這個check 設定成false, 才能用各自的request thread ??

db.row_factory = sqlite3.Row 
#這樣才會用key value 的形式, 不然取出來是index
# 讓我們在讀取資料庫時，可以直接用 row["欄位名稱"] 的方式來存取資料，
# 而不是 row[0]、row[1] 這樣的 index。

#(作業新增)........ 核心路徑設定 ---
# 1. 取得目前 app.py 所在的資料夾路徑
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 定位到上一層資料夾 (Parent Folder)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
# 3. 在上一層資料夾建立 models 目錄路徑
MODEL_DIR = os.path.join(PARENT_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')

# 與 app.py 同一層的 train.csv 路徑
TRAIN_DATA_PATH = os.path.join(CURRENT_DIR, 'titanic_train.csv')

# print(CURRENT_DIR, PARENT_DIR, MODEL_DIR, MODEL_PATH, TRAIN_DATA_PATH)

# 確保上一層的 models 資料夾存在
os.makedirs(MODEL_DIR, exist_ok=True)

# feature_cols = ['Pclass', 'Sex', 'Age', 'Fare']
# feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q','Embarked_S']
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked_S']

def preprocess_data(df, is_training=True):
    """
    極簡極限版 Titanic 特徵工程：
    只挑選最關鍵的數值特徵，並處理缺失值與欄位轉換
    """
    # 複製一份避免修改到原始 DataFrame
    data = df.copy()
    
    # 1. 填補年齡缺失值 (用中位數)
    data['Age'] = data['Age'].fillna(28.0)
    data['Fare'] = data['Fare'].fillna(14.0)
    
    # 2. 將性別轉換為 0 與 1 (male=1, female=0)
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0}).fillna(1)
    
    # 3. 挑選模型需要的特徵欄位 (必須全是數值)
    # global :  feature_cols = ['Pclass', 'Sex', 'Age', 'Fare']
    
    X = data[feature_cols]
    
    if is_training:
        # 訓練時需要特別拔出目標標籤 'Survived'
        y = data['Survived']
        return X, y
    
    return X

def train_and_save_model():
    """
    讀取與 app.py 同一層的 train.csv，透過 GridSearchCV 訓練並儲存
    """
    # 1. 讀取真實的 train.csv 檔案
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"在 {TRAIN_DATA_PATH} 找不到 train.csv 檔案！")
        
    raw_train_df = pd.read_csv(TRAIN_DATA_PATH)
    
    # 2. 進行特徵工程清洗
    X_train, y_train = preprocess_data(raw_train_df, is_training=True)
    
    # 3. 定義基礎模型與超參數網格
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        # 'n_estimators': [50, 100],
        # 'max_depth': [3, 5, None],
        # 'min_samples_split': [2, 5]

        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 100, None],
        'min_samples_split': [2, 5, 20, 50, 100]
    }
    print('before gridsearch cv')
    start_time = time.time()  # 記錄開始時間

    # 4. 建立 GridSearchCV 開始尋找最佳組合
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    end_time = time.time()    # 記錄結束時間
    training_time = end_time - start_time  # 計算總時間（秒）
    print(f"訓練總時間：{training_time:.2f} 秒")

    # 5. 提取最棒的模型並儲存到上一層的 models/ 中
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, MODEL_PATH)
#......
 
def row_to_dict(row):
    return dict(row)

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/passengers/new')
def new_passenger_page():
    return render_template('new.html')

@app.route('/passengers/<int:passenger_id>/edit')
def edit_passenger_page(passenger_id):
    return render_template(
        "edit.html",
        passenger_id=passenger_id
    )
# 以後自己寫可加v1.0 v1.1 ...
@app.route('/api/passengers', methods=['GET'])
def get_passengers():
                #get 不到參數就1, 設定轉成int, 不用怕奇怪字串
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)

    search = request.args.get('search', '')
    offset = (page - 1) * per_page

    if search != '':
        total_row = db.execute(
            """
            SELECT COUNT(*) AS total
            FROM titanic
            WHERE Name LIKE ?  
            """,
            (f"%{search}%",) # %xxx% 這個就是SQL 裡面找字串的wild card, 不要搞混了
                             # Tuple 一個值還是要加 ',' 不然會錯 
        ).fetchone()

        rows = db.execute(
            """
            SELECT *
            FROM titanic
            WHERE Name LIKE ?
            ORDER BY PassengerId
            LIMIT ?
            OFFSET ?
            """,
            (f"%{search}%", per_page, offset)            
        ).fetchall()
    else:
        total_row = db.execute(
            """
            SELECT COUNT(*) AS total
            FROM titanic
            """
        ).fetchone()

        rows = db.execute(
            """
            SELECT *
            FROM titanic
            ORDER BY PassengerId
            LIMIT ?
            OFFSET ?
            """,
            (per_page, offset)            
        ).fetchall()

    total = total_row['total']

    return jsonify({
        'message': 'ok',
        'items': [row_to_dict(row) for row in rows],
        'page': page,
        'per_page':per_page,
        'total': total
    }), 200

@app.route('/api/passengers/<int:passenger_id>', methods = ["GET"])
def get_passenger(passenger_id):
    row = db.execute(
        "SELECT * FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    ).fetchone()

    if row is None:
        return jsonify({'error':"找不到資料"}), 404
    
    return jsonify({
        'message':'ok',
        'item':row_to_dict(row)
    }), 200


@app.route('/api/passengers', methods=['POST'])
def create_passenger():
    data = request.get_json()

    cursor = db.execute(   #老師習慣5個5個對..
        """
        INSERT INTO titanic (
            Survived, Pclass, Name, Sex, Age,
            SibSp, Parch, Ticket, Fare, Cabin,
            Embarked
        )
        VALUES (
            ?, ?, ?, ?, ?, 
            ?, ?, ?, ?, ?, 
            ?
        )
        """,
        (
            data['Survived'],
            data['Pclass'],
            data['Name'],
            data['Sex'],
            data['Age'],
            data['SibSp'],
            data['Parch'],
            data['Ticket'],
            data['Fare'],
            data['Cabin'],
            data['Embarked'],
        )
    )
    
    db.commit()

    new_id = cursor.lastrowid  #檢查一下被影響的資料

    row = db.execute(
        'SELECT * from titanic WHERE PassengerId = ?',
        (new_id,)
    ).fetchone()

    return jsonify({
        'message': 'created',
        'item': row_to_dict(row)
    }), 201


@app.route('/api/passengers/<int:passenger_id>', methods=['PUT'])
def update_passenger(passenger_id):
    data = request.get_json()
    # update 這種破壞性的指令, 要特別小心 一定要加where
    cursor = db.execute(
        '''
        UPDATE titanic
        SET
            Survived = ?,
            Pclass = ?,
            Name = ?,
            Sex = ?,
            Age = ?,
            SibSp = ?,
            Parch = ?,
            Ticket = ?,
            Fare = ?,
            Cabin = ?,
            Embarked = ?
        WHERE PassengerId = ?
        ''',
        (
            data["Survived"],
            data["Pclass"],
            data["Name"],
            data["Sex"],
            data["Age"],
            data["SibSp"],
            data["Parch"],
            data["Ticket"],
            data["Fare"],
            data["Cabin"],
            data["Embarked"],
            passenger_id
        )
    )
    db.commit()

    if cursor.rowcount == 0:  #檢查一下被影響的資料
        return jsonify({'error': '找不到資料'}) ,404
    
    row = db.execute(
        'SELECT * FROM titanic WHERE PassengerId = ?',
        (passenger_id,)
    ).fetchone()

    if row is None:
        return jsonify({'error': '找不到資料'}) ,404

    return jsonify({
        'message': 'updated',
        'item': row_to_dict(row)
    }), 200

@app.route('/api/passengers/<int:passenger_id>', methods=['DELETE'])
def delete_passenger(passenger_id):
    cursor = db.execute(
        "DELETE FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    )

    db.commit()
    # 判斷影響的行數或資料
    if cursor.rowcount == 0:
        return jsonify({'error':'找不到資料'}), 404
    
    return jsonify({'message': 'deleted'}), 200


#(作業新增) .....
# --- 功能一：檢查/載入模型 ---
@app.route('/load_model', methods=['POST'])
def load_model_api():
    if not os.path.exists(MODEL_PATH):
        try:
            train_and_save_model()
            return jsonify({"status": "training", "message": "開始訓練"})
        except Exception as e:
            return jsonify({"status": "error", "message": f"訓練失敗: {str(e)}"}), 500
        
    try:
        model = joblib.load(MODEL_PATH)
        params = model.get_params()
        important_params = {
            "n_estimators": params.get("n_estimators"),
            "max_depth": params.get("max_depth"),
            "min_samples_split": params.get("min_samples_split")
        }
        return jsonify({"status": "success", "params": important_params})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
# --- 功能二：上傳 CSV 並進行現場預測 ---
@app.route('/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files: return "沒有選取檔案", 400
    file = request.files['file']
    if file.filename == '': return "未選擇檔案", 400
    
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        if not os.path.exists(MODEL_PATH):
            return "錯誤：模型尚未訓練或已被刪除，無法進行預測！", 400
            
        model = joblib.load(MODEL_PATH)

        print(model.feature_importances_)
        try:
            X_predict = preprocess_data(df, is_training=False)
            predictions = model.predict(X_predict) 
            probabilities = model.predict_proba(X_predict)[:, 1]
            df['生存預測'] = predictions
            df['生存機率'] = [f"{prob * 100:.2f}%" for prob in probabilities]
        except Exception as e:
            return f"預測失敗: {str(e)}", 500
        
        html_table = df.to_html(index=False)
        return render_template('predict.html', table_html=html_table)
    
@app.route('/feature_importance', methods=['POST'])
def feature_importance():
    if not os.path.exists(MODEL_PATH):
        return "錯誤：模型尚未訓練，無法顯示特徵重要性！", 400

    # 1. 現場載入模型
    model = joblib.load(MODEL_PATH)
    importances = model.feature_importances_
    
    # 2. 打包成純粹的 Python 字典 (Dictionary)
    # 結果會像這樣： {'Pclass': 0.12, 'Sex': 0.45, 'Age': 0.23, 'Fare': 0.20}
    feat_data = dict(zip(feature_cols, importances))
    feat_data = {k: round(v, 4) for k, v in sorted(feat_data.items(), key=lambda x: x[1], reverse=True) }
    print(feat_data)

    # 3. 分別取出排序完美的 標籤列表 與 數值列表
    sorted_labels = list(feat_data.keys())
    sorted_values = list(feat_data.values())

    # 4. 打包成一個大字典傳給前端
    chart_data = {
        "labels": sorted_labels,
        "values": sorted_values
    }

    # 3. 把資料傳給專門用來畫圖的 HTML 樣板
    return render_template('chart.html', data=chart_data)
    
# 【新增功能】：刪除硬碟中的模型檔案 ---
@app.route('/delete_model', methods=['POST'])
def delete_model_api():
    try:
        if os.path.exists(MODEL_PATH):
            # 直接刪除上一層 models 資料夾底下的 joblib 檔案
            os.remove(MODEL_PATH)
            return jsonify({"status": "success", "message": "模型檔案已成功刪除！"})
        else:
            return jsonify({"status": "not_found", "message": "找不到模型檔案，本來就是空的。"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"刪除失敗: {str(e)}"}), 500


@app.route('/predict_sys')
def predict_sys():
    return render_template('predict.html') # 放原本寫好的預測模型網頁
#..................



if __name__ == '__main__':
    app.run(
        debug=True,
        host='127.0.0.1', #對外就是設0.0.0.0
        port=5000
    )

#老師建議之後自己加上 rollback 機制