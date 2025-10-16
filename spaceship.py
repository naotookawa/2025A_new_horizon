import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import numpy as np
import os

# --- 設定 ---
DATA_DIR = './spaceship-titanic-ut-komaba-2025/'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
OUTPUT_FILE = 'submission_lgbm.csv'
TARGET_COLUMN = 'Transported'

# --- データの読み込み ---
try:
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))
    test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))
except FileNotFoundError as e:
    print(f"ファイルが見つかりません: {e.filename}。ディレクトリとファイル名を確認してください。")
    exit()

# 目的変数の変換 (True/False -> 1/0)
train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].astype(int)

# 説明変数と目的変数の分離
X = train_df.drop(TARGET_COLUMN, axis=1)
y = train_df[TARGET_COLUMN]
X_test = test_df.copy()

# --- 特徴量の定義 ---
# 数値特徴量: 欠損値補完後にスケーリング
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# カテゴリ特徴量: 欠損値補完後にOne-Hot Encoding
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

# シンプルな特徴量セットを使用
all_features = numeric_features + categorical_features
X = X[all_features]
X_test_processed = X_test[all_features]

# --- 前処理 Pipeline の定義 ---

# 数値特徴量の処理
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# カテゴリ特徴量の処理
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=Falseで密行列を生成
])

# ColumnTransformerで特徴量ごとに異なる前処理を適用
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# --- モデル定義と全体の Pipeline 構築 ---

# LightGBMモデルの定義 (分類問題のため'binary'を使用)
# random_stateを設定し再現性を確保
lgbm_model = lgb.LGBMClassifier(random_state=42, n_estimators=1000, n_jobs=-1, objective='binary', verbose=-1)

# 全体の Pipeline: 前処理 -> LightGBMモデル
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgbm_model)
])

# --- ステップ1: クロスバリデーションによる手元でのスコア確認 ---

# StratifiedKFold (分類問題に適した分割)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# クロスバリデーションの実行 (Accuracyを評価)
# `X` と `y` を渡すと、Pipelineが自動的に前処理も適用しながら交差検証を行います。
print("--- ステップ1: 5分割クロスバリデーション開始 ---")
cv_scores = cross_val_score(full_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

print("--- クロスバリデーションスコア ---")
print(f"各CVのAccuracy: {cv_scores}")
print(f"平均Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# --- ステップ2: 全データで再学習し、test.csvに対する出力を生成 ---

# 全学習データでPipelineを再学習
print("\n--- ステップ2: 全学習データで再学習 ---")
full_pipeline.fit(X, y)
print("学習完了。")

# testデータで予測
# LightGBMはデフォルトでクラス確率（float）を返すため、`predict`でクラスラベル（0または1）を取得
test_predictions = full_pipeline.predict(X_test_processed)

# 予測結果を True/False のブール値に戻す
predicted_transported = test_predictions.astype(bool)

# 提出ファイルの形式に整形
submission_df = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Transported': predicted_transported
})

# ファイルに出力
submission_df.to_csv(os.path.join(DATA_DIR, OUTPUT_FILE), index=False)

print(f"\n提出用ファイル '{os.path.join(DATA_DIR, OUTPUT_FILE)}' を生成しました。")