import pandas as pd
import datetime
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.utils import resample

def main():
    print('Target Action Prediction Pipeline')

    # Загрузка данных с явным указанием типов данных для избежания DtypeWarning
    df_sessions = pd.read_csv('data/ga_sessions.csv', dtype={'client_id': str})
    df_hits = pd.read_csv('data/ga_hits.csv')

    # Определение целевых действий
    target_actions = [
        'sub_car_claim_click', 'sub_car_claim_submit_click',
        'sub_open_dialog_click', 'sub_custom_question_submit_click',
        'sub_call_number_click', 'sub_callback_submit_click',
        'sub_submit_success', 'sub_car_request_submit_click'
    ]
    df_hits['target'] = df_hits['event_action'].isin(target_actions).astype(int)

    df_combined = pd.merge(df_sessions, df_hits, on='session_id', how='left')

    # Агрегация df_hits на уровне сессий
    agg_hits = df_combined.groupby('session_id').agg(
        target=('target', 'max'),  # Была ли хоть одна целевая активность
        n_hits=('hit_type', 'count'),
        n_pages=('hit_page_path', 'nunique'),
        n_events=('event_action', 'count'),
        n_event_types=('event_action', 'nunique'),
        n_hit_types=('hit_type', 'nunique'),
        session_duration=('hit_time', lambda x: x.max() - x.min())
    ).reset_index()
    df_combined = agg_hits.merge(df_sessions, on='session_id', how='left')
    df_combined['target'] = df_combined['target'].fillna(0).astype(int)

    # Новая фильтрация: удаление аномальных пользователей по количеству действий
    client_session_counts = df_sessions['client_id'].value_counts()
    bots = client_session_counts[client_session_counts > 100].index
    df_combined = df_combined[~df_combined['client_id'].isin(bots)]

    df_combined = df_combined.drop(columns=['n_hit_types','n_event_types','n_events', 'n_pages', 'n_hits', 'session_duration'])

    # Заполнение пропусков
    df_combined['device_model'] = df_combined['device_model'].fillna('other')
    df_combined['device_os'] = df_combined['device_os'].fillna('other')
    df_combined['device_brand'] = df_combined['device_brand'].fillna('other')

    # Список столбцов для заполнения
    utm_columns = ['utm_keyword', 'utm_adcontent', 'utm_campaign', 'utm_source']

    for col in utm_columns:
        mode_value = df_combined[col].mode()[0]
        df_combined[col] = df_combined[col].fillna(mode_value)

    df_combined = df_combined.dropna(subset=['client_id'])

    # Удаление дубликатов
    df_combined = df_combined.drop_duplicates(keep='last')

    # Преобразование временных признаков
    df_combined['visit_date'] = pd.to_datetime(df_combined['visit_date'], format='%Y-%m-%d', utc=True)
    df_combined['visit_time'] = pd.to_datetime(df_combined['visit_time'], format='%H:%M:%S', utc=True)

    #удаление выбросов
    visit_counts = df_combined.groupby('visit_date')['visit_date'].count()
    visit_counts.name = 'count'
    visit_counts = visit_counts.reset_index()
    Q1 = visit_counts['count'].quantile(0.25)
    Q3 = visit_counts['count'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = visit_counts[(visit_counts['count'] < lower_bound) | (visit_counts['count'] > upper_bound)]
    df_filtered = df_combined[~df_combined['visit_date'].isin(outliers['visit_date'])]

    target_counts = df_filtered[df_filtered['target'] == 1].groupby('visit_date').size().reset_index(name='target_count')
    mean_target_count = target_counts['target_count'].mean()
    std_target_count = target_counts['target_count'].std()
    outliers_target = target_counts[target_counts['target_count'] > mean_target_count + 3 * std_target_count]
    df_filtered = df_filtered[~df_filtered['visit_date'].isin(outliers_target['visit_date'])]

    # Feature Engineering

    def get_time_period(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'day'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    df_filtered['time_period'] = df_filtered['visit_time'].dt.hour.apply(get_time_period)

    df_filtered['month'] = df_filtered['visit_date'].dt.month
    df_filtered['day'] = df_filtered['visit_date'].dt.day
    df_filtered['day_of_week'] = df_filtered['visit_date'].dt.weekday
    df_filtered['is_weekend'] = df_filtered['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    df_filtered['time_only'] = df_filtered['visit_time'].dt.time
    df_filtered['hour'] = pd.to_datetime(df_filtered['time_only'], format='%H:%M:%S').dt.hour.astype(int)

    # Ограничение категориальных признаков топ-20 значениями
    for col in ['utm_keyword', 'utm_source', 'utm_adcontent', 'device_model', 'device_brand', 'device_browser', 'geo_country', 'geo_city', 'utm_campaign']:
        top_values = df_filtered[col].value_counts().head(20).index
        df_filtered[col] = df_filtered[col].apply(lambda x: x if x in top_values else 'other')

    # Обработка device_screen_resolution
    df_filtered[['width', 'height']] = df_filtered['device_screen_resolution'].str.split('x', expand=True)
    df_filtered['width'] = pd.to_numeric(df_filtered['width'], errors='coerce')
    df_filtered['height'] = pd.to_numeric(df_filtered['height'], errors='coerce')
    df_filtered['width'] = df_filtered['width'].fillna(df_filtered['width'].median(skipna=True)).astype(int)
    df_filtered['height'] = df_filtered['height'].fillna(df_filtered['height'].median(skipna=True)).astype(int)

    # Кодирование device_category
    df_filtered['device_category'] = df_filtered['device_category'].map({'mobile': 0, 'desktop': 1, 'tablet': 2})

    # Удаление ненужных столбцов
    df_filtered = df_filtered.drop(columns=['session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number', 'device_screen_resolution', 'time_only'])
    

    # Определение категориальных и числовых признаков
    categorical_features = [
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
        'device_os', 'device_brand', 'device_model', 'device_browser',
        'time_period', 'geo_country', 'geo_city'
    ]

    df_filtered[categorical_features] = df_filtered[categorical_features].astype('category')
    
    numerical_features = ['month', 'day', 'hour', 'day_of_week', 'is_weekend', 'width', 'height', 'device_category']
    
    for col in numerical_features:
        df_filtered[col] = pd.to_numeric(df_filtered[col], downcast='integer')

    #уменьшаем количество отрицательных целевых действий для балансировки тренировочного датасета

    df_majority = df_filtered[df_filtered['target'] == 0]
    df_minority = df_filtered[df_filtered['target'] == 1]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=len(df_minority)*5,  # или 3, зависит от баланса
                                       random_state=42)

    df_filtered = pd.concat([df_majority_downsampled, df_minority])

    # Определение X и y
    X = df_filtered.drop(columns=['target'])
    y = df_filtered['target']

    # Преобразователи для pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    # Модели с лучшими гиперпараметрами
    models = (
        LogisticRegression(max_iter=1000, random_state=42),
        RandomForestClassifier(
            class_weight=None, max_depth=20, min_samples_leaf=2,
            min_samples_split=2, n_estimators=200, random_state=42
        ),
        LGBMClassifier(
            is_unbalance=False, learning_rate=0.05, max_depth=-1,
            n_estimators=100, num_leaves=100, random_state=42, verbose=-1
        ),
        CatBoostClassifier(
            auto_class_weights='Balanced', depth=10, iterations=200,
            l2_leaf_reg=3, learning_rate=0.1, random_state=42, verbose=0
        )
    )

    # Определение метрики ROC-AUC
    auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

    best_score = 0.0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring=auc_scorer)
        print(f'model: {type(model).__name__}, auc_mean: {score.mean():.4f}, auc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    # Обучение лучшей модели
    best_pipe.fit(X, y)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, auc: {best_score:.4f}')

    # Сохранение модели
    joblib.dump({
        'model': best_pipe,
        'metadata': {
            'name': 'target action prediction pipeline',
            'author': 'Lilia Filippova',
            'version': 1,
            'date': datetime.datetime.now(),
            'type': type(best_pipe.named_steps["classifier"]).__name__,
            'roc_auc': best_score
        }
    }, 'model/model.pkl')

if __name__ == '__main__':
    main()