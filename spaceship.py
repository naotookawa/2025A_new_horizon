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
OUTPUT_FILE = 'submission_cv_lgbm.csv'
TARGET_COLUMN = 'Transported'
RANDOM_STATE = 42
N_SPLITS = 5 # CV分割数

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

# --- 1. 特徴量エンジニアリング関数 ---
def feature_engineer(df):
    """
    データフレームに特徴量エンジニアリングを適用します。
    - Cabin情報 (Deck, Num, Side) を抽出
    - TotalSpent (総支出額) を計算
    - PassengerIDからグループごとのIDを抽出
    """
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['TotalSpent'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
    df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]

    group_size_map = df.groupby('Group')['PassengerId'].transform('count')
    df['GroupSize'] = group_size_map

    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

     # --- NEW: Ageのカテゴリ化 ---
    # 0 < Age <= 12, 12 < Age <= 17, 17 < Age
    # df['Age'].max() + 1 を使うことで、17歳以上の最大値まで全て含める
    bins = [0, 12, 17, df['Age'].max() + 1] 
    labels = ['Child (0-12)', 'Teen (13-17)', 'Adult (18+)' ]
    

    df['AgeGroup'] = pd.cut(
        df['Age'], 
        bins=bins, 
        labels=labels, 
        right=True,        # (a, b] の区間設定
        include_lowest=True # 0を含むように
    ).astype(object) # カテゴリとして処理するためにobject型に変換

    # 元の 'Cabin' は不要なので削除
    df = df.drop('Cabin', axis=1, errors='ignore')
    return df

# 特徴量エンジニアリングの適用
X = feature_engineer(X.copy())
X_test = feature_engineer(X_test.copy())


# --- 2. 特徴量の定義（FE後） ---
numeric_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpent', 'GroupSize']
# 新たなカテゴリ特徴量 'Deck', 'Side' を追加
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'IsAlone', 'AgeGroup'] 
# 'Num' は数値だが、カテゴリ的な性質が強いため、今回はシンプルに無視します（より高度なFEが必要）

# 使用する特徴量のみを選択（PassengerId, Nameなどを除く）
all_features = numeric_features + categorical_features
X_processed = X[all_features]
X_test_processed = X_test[all_features]


# --- 3. 前処理 Pipeline の定義（共通） ---
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# --- 4. CV評価関数 ---
def evaluate_lgbm_config(params, X_data, y_data):
    """
    与えられたLightGBMパラメータでクロスバリデーションを実行し、スコアを返します。
    """
    print(f"\n--- CV評価開始: Params = {params} ---")
    
    # LightGBMモデルの定義 (渡されたparamsを使用)
    lgbm_model = lgb.LGBMClassifier(
        random_state=RANDOM_STATE, 
        n_estimators=1000, 
        n_jobs=-1, 
        objective='binary', 
        verbose=-1,
        **params # パラメータを展開
    )

    # 全体の Pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgbm_model)
    ])

    # StratifiedKFold
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # クロスバリデーションの実行
    cv_scores = cross_val_score(full_pipeline, X_data, y_data, cv=cv, scoring='accuracy', n_jobs=-1)
    
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()

    print("--- CVスコア結果 ---")
    print(f"各CVのAccuracy: {cv_scores}")
    print(f"平均Accuracy: {mean_score:.4f} (±{std_score:.4f})")
    
    return mean_score, std_score, lgbm_model


# --- 5. 実験と最良モデルの選択 ---
print("--- ステップ1: クロスバリデーションによるハイパーパラメータ実験開始 ---")

# (A) ベースライン設定 (現在のデフォルト設定)
params_baseline = {} 

# (B) チューニング設定 (例: 学習率を下げ、木の深さを制限して汎化性能を向上)
params_tuned = {
    'learning_rate': 0.015045884263884837,
    'max_depth': 4,
    'num_leaves': 46,
    'reg_alpha': 0.0012331367058110389,
    'reg_lambda': 0.000153443533884356,
    'min_child_samples': 17,
    'subsample': 0.8449842887788588,
    'colsample_bytree': 0.8597784530564854
}

results = {}
best_score = -1
best_model_params = None

# 実験の実行
for name, params in [('Baseline', params_baseline), ('Tuned', params_tuned)]:
    mean, std, model = evaluate_lgbm_config(params, X_processed, y)
    results[name] = {'mean_score': mean, 'std_score': std, 'model': model, 'params': params}
    
    if mean > best_score:
        best_score = mean
        best_model_params = params
        
print("\n" + "="*50)
print(f"⭐ 最良のCVスコア: {best_score:.4f}")
print(f"⭐ 選択された設定: {best_model_params}")
print("="*50)

# --- 6. ステップ2: 最良モデルでの最終学習と予測生成 ---

# 最良の設定で最終的なパイプラインを構築
best_lgbm_model = lgb.LGBMClassifier(
    random_state=RANDOM_STATE, 
    n_estimators=1000, 
    n_jobs=-1, 
    objective='binary', 
    verbose=-1,
    **best_model_params
)

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_lgbm_model)
])


print("\n--- ステップ2: 最良設定で全学習データで再学習 ---")
final_pipeline.fit(X_processed, y)
print("学習完了。")

# testデータで予測
test_predictions = final_pipeline.predict(X_test_processed)

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