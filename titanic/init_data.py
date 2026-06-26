import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
import os

DB_PATH = 'my_db.db'
# CSV 檔案路徑
CSV_PATH = "titanic.csv"

# 建立資料表的 SQL 語句，包含欄位定義和約束條件
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS titanic (
    PassengerId INTEGER PRIMARY KEY,
    Survived INTEGER NOT NULL CHECK (Survived IN (0, 1)),
    Pclass INTEGER NOT NULL CHECK (Pclass IN (1, 2, 3)),
    Name TEXT NOT NULL CHECK (length(Name) <= 100),
    Sex TEXT NOT NULL CHECK (Sex IN ('male', 'female')),
    Age REAL CHECK (Age IS NULL OR (Age >= 0 AND Age <= 120)),
    SibSp INTEGER NOT NULL DEFAULT 0 CHECK (SibSp >= 0),
    Parch INTEGER NOT NULL DEFAULT 0 CHECK (Parch >= 0),
    Ticket TEXT NOT NULL CHECK (length(Ticket) <= 30),
    Fare REAL NOT NULL DEFAULT 0 CHECK (Fare >= 0),
    Cabin TEXT CHECK (Cabin IS NULL OR length(Cabin) <= 30),
    Embarked TEXT CHECK (Embarked IS NULL OR Embarked IN ('C', 'Q', 'S'))
);
"""

#建索引
CREATE_INDEX_SQL = [
     "CREATE INDEX IF NOT EXISTS idx_titanic_name ON titanic(Name);"
]

def init_db():
    conn = sqlite3.connect(DB_PATH)

    try:
        cursor = conn.cursor()
        
        cursor.execute("DROP TABLE IF EXISTS titanic;")
        cursor.execute(CREATE_TABLE_SQL)

        for sql in CREATE_INDEX_SQL:
            cursor.execute(sql)

        df = pd.read_csv(CSV_PATH)
        print(df.head(5))

        df = df.where(pd.notnull(df), None) #空的填None ?
        df.to_sql("titanic", conn, if_exists="append", index=False)
        
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback() #出錯時做還原
        print(f"資料庫初始化失敗: {e}")
    finally:
        conn.close()
        print('my_db.db 建立完成, titanic 資料表已匯入' )


def split_train_test():
    df = pd.read_csv(CSV_PATH)
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['Survived'], random_state=0)

    df_test['Embarked'] = df_test['Embarked'].fillna(df_train['Embarked'].mode()[0])
    df_test = pd.get_dummies(df_test, columns=['Embarked'], drop_first=False, dtype=int)

    df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])
    df_train = pd.get_dummies(df_train, columns=['Embarked'], drop_first=False, dtype=int)

    df_train.to_csv('titanic_train.csv')
    df_test.to_csv('./static/download/titanic_test.csv')
                    

if __name__ == "__main__":
    # 取得目前工作目錄
    cwd = os.getcwd()
    print("目前的工作目錄是：", cwd)
    print("init db")
    init_db()
    print("split train test set")
    split_train_test()
    print('done')
    