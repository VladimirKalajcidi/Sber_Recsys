import argparse
import os
import pickle
from pathlib import Path
from typing import Tuple
import pandas as pd
from my_model import ALSModel
import numpy as np
import xgboost as xgb

cfg_data = {
    "user_column": "user_id",
    "item_column": "item_id",
    "date_column": "timestamp",
    "rating_column": "weight",
    "weighted": False,
    "dataset_names": ["smm", "zvuk"],
    "data_dir": "./",
    "model_dir": "./saved_models",
}


def sort_item_ids(row, df):
    '''
        Функция для сортировки айтемов по предиктам ранкера
    '''

    user_id = row['user_id']
    predictions = df[df['user_id'] == user_id]['predictions'].values
    
    predictions = predictions[0]
    item_ids = row['item_id']
    
    paired_list = sorted(zip(predictions, item_ids), key=lambda x: x[0], reverse=True)
    
    return [item_id for _, item_id in paired_list]


def predict_ranker(model, df):
    # Предикт ранкера
    return model.predict(df.loc[:, ~df.columns.isin(['user_id'])])


def process_datasets(train_smm, train_zvuk):
    '''
        Изначальная обработка данных.
        После нее имеются колонки [mean_user_rating_smm, mean_user_rating_zvuk, mean_item_rating]
    '''

    train_smm['timestamp'] -= 1673740803033000000
    train_zvuk['timestamp'] -= 1673740803033000000
    train_smm['timestamp'] /= 10**6
    train_zvuk['timestamp'] /= 10**6

    mean_user_rating_smm = train_smm.groupby('user_id')['rating'].mean().reset_index()
    mean_user_rating_smm.rename(columns={'rating': 'mean_user_rating_smm'}, inplace=True)

    mean_user_rating_zvuk = train_zvuk.groupby('user_id')['rating'].mean().reset_index()
    mean_user_rating_zvuk.rename(columns={'rating': 'mean_user_rating_zvuk'}, inplace=True)

    user_meta = pd.merge(mean_user_rating_smm, mean_user_rating_zvuk, on='user_id', how='outer')

    user_meta['mean_user_rating_smm'] = user_meta['mean_user_rating_smm'].fillna(user_meta['mean_user_rating_smm'].mean())
    user_meta['mean_user_rating_zvuk'] = user_meta['mean_user_rating_zvuk'].fillna(user_meta['mean_user_rating_zvuk'].mean())

    mean_item_rating_smm = train_smm.groupby('item_id')['rating'].mean().reset_index()
    mean_item_rating_smm.rename(columns={'rating': 'mean_item_rating'}, inplace=True)
    
    mean_item_rating_zvuk = train_zvuk.groupby('item_id')['rating'].mean().reset_index()
    mean_item_rating_zvuk.rename(columns={'rating': 'mean_item_rating'}, inplace=True)

    item_meta_zvuk = mean_item_rating_zvuk
    item_meta_smm = mean_item_rating_smm

    item_meta_zvuk['mean_item_rating'] = item_meta_zvuk['mean_item_rating'].fillna(item_meta_zvuk['mean_item_rating'].mean())
    item_meta_smm['mean_item_rating'] = item_meta_smm['mean_item_rating'].fillna(item_meta_smm['mean_item_rating'].mean())

    train_smm.sort_values(by='timestamp', inplace=True)
    train_zvuk.sort_values(by='timestamp', inplace=True)

    return user_meta, item_meta_zvuk, item_meta_smm


def create_intersection_dataset(
    smm_events: pd.DataFrame,
    zvuk_events: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    smm_item_count = smm_events["item_id"].nunique()
    zvuk_item_count = zvuk_events["item_id"].nunique()

    zvuk_events["item_id"] += smm_item_count
    merged_events = pd.concat([smm_events, zvuk_events])
    item_indices_info = pd.DataFrame(
        {"left_bound": [0, smm_item_count],
         "right_bound": [smm_item_count, smm_item_count + zvuk_item_count]},
        index=["smm", "zvuk"]
    )
    user_ids = set(merged_events["user_id"])
    encoder = {id: n for n, id in enumerate(user_ids)}
    merged_events["user_id"] = merged_events["user_id"].map(encoder)
    return merged_events, item_indices_info, encoder


def fit_als() -> None:
    smm_path = os.path.join(cfg_data["data_dir"], "train_smm.parquet")
    zvuk_path = os.path.join(cfg_data["data_dir"], "train_zvuk.parquet")
    print("Train smm-events:", smm_path)
    print("Train zvuk-events:", zvuk_path)
    smm_events = pd.read_parquet(smm_path)
    zvuk_events = pd.read_parquet(zvuk_path)
    
    train_events, indices_info, encoder = create_intersection_dataset(smm_events, zvuk_events)
    train_events["weight"] = 1
    
    my_model = ALSModel(
        cfg_data,
        factors=20,
        regularization=0.002,
        iterations=1,
        alpha=20,
    )
    my_model.fit(train_events)
    my_model.users_encoder = encoder
    
    md = Path(cfg_data["model_dir"])
    md.mkdir(parents=True, exist_ok=True)
    with open(md / "als.pickle", "bw") as f:
        pickle.dump(my_model, f)
    indices_info.to_parquet(md / "indices_info.parquet")


def predict(subset_name: str) -> None:
    with open(Path(cfg_data["model_dir"]) / "als.pickle", "br") as f:
        my_model: ALSModel = pickle.load(f)
    
    my_model.model = my_model.model.to_cpu()
    encoder = my_model.users_encoder
    decoder = {n: id for id, n in encoder.items()}
    indices_info = pd.read_parquet(Path(cfg_data["model_dir"]) / "indices_info.parquet")

    test_data = pd.read_parquet(os.path.join(cfg_data["data_dir"], f"test_{subset_name}.parquet"))

    test_data["user_id"] = test_data["user_id"].map(encoder)
    
    test_data["weight"] = 1

    left_bound, right_bound = (
        indices_info["left_bound"][subset_name],
        indices_info["right_bound"][subset_name],
    )

    my_model.model.item_factors[:left_bound, :] = 0
    my_model.model.item_factors[right_bound:, :] = 0
    recs, user_ids = my_model.recommend_k(test_data, k=20)
    recs = pd.Series([np.array(x - left_bound) for x in recs.tolist()], index=user_ids)
    recs = recs.reset_index()
    recs.columns = ["user_id", "item_id"]
    recs["user_id"] = recs["user_id"].map(decoder)

    prediction_path = Path(cfg_data["data_dir"]) / f"submission_{subset_name}_als.parquet"
    recs.to_parquet(prediction_path)


def fit_ranker() -> None:
    # Считывание данных

    train_smm = pd.read_parquet("train_smm.parquet")
    train_zvuk = pd.read_parquet("train_zvuk.parquet")

    test_smm = pd.read_parquet("test_smm.parquet")
    test_zvuk = pd.read_parquet("test_zvuk.parquet")

    # Обработка данных

    user_meta, item_meta_zvuk, item_meta_smm = process_datasets(train_smm, train_zvuk)

    train_smm = train_smm[train_smm['user_id'].isin(test_smm['user_id'])]
    train_zvuk = train_zvuk[train_zvuk['user_id'].isin(test_zvuk['user_id'])]

    groups_smm = train_smm.groupby('user_id').size().to_frame('size')['size'].to_numpy()
    groups_zvuk = train_zvuk.groupby('user_id').size().to_frame('size')['size'].to_numpy()

    train_smm = train_smm.merge(user_meta, on='user_id', how='left')
    train_smm = train_smm.merge(item_meta_smm, on='item_id', how='left')
    train_smm = train_smm[["user_id", "mean_user_rating_smm", "mean_item_rating", "rating"]].rename(columns={"mean_user_rating_smm": "mean_user_rating"})

    train_zvuk = train_zvuk.merge(user_meta, on='user_id', how='left')
    train_zvuk = train_zvuk.merge(item_meta_zvuk, on='item_id', how='left')
    train_zvuk = train_zvuk[["user_id", "mean_user_rating_zvuk", "mean_item_rating", "rating"]].rename(columns={"mean_user_rating_zvuk": "mean_user_rating"})

    # Инициализация ранкеров
    
    model_smm = xgb.XGBRanker(
        tree_method='hist',
        booster='gbtree',
        objective='rank:pairwise',
        n_estimators=5,
    )

    model_zvuk = xgb.XGBRanker(
        tree_method='hist',
        booster='gbtree',
        objective='rank:pairwise',
        n_estimators=5,
    )

    # Обучение

    model_smm.fit(train_smm[['mean_user_rating', 'mean_item_rating']], train_smm['rating'], group=groups_smm, verbose=True)
    model_zvuk.fit(train_zvuk[['mean_user_rating', 'mean_item_rating']], train_zvuk['rating'], group=groups_zvuk, verbose=True)

    # Считывание и обрабокта данных после отбора кандидатов с помощью als

    sub_smm = pd.read_parquet("submission_smm_als.parquet")
    sub_zvuk = pd.read_parquet("submission_zvuk_als.parquet")

    sub_smm = sub_smm.explode("item_id")
    sub_zvuk = sub_zvuk.explode("item_id")

    sub_smm = sub_smm.merge(user_meta, on='user_id', how='left')
    sub_smm = sub_smm.merge(item_meta_smm, on='item_id', how='left')
    sub_smm = sub_smm[["user_id", "mean_user_rating_smm", "mean_item_rating"]].rename(columns={"mean_user_rating_smm": "mean_user_rating"})

    sub_zvuk = sub_zvuk.merge(user_meta, on='user_id', how='left')
    sub_zvuk = sub_zvuk.merge(item_meta_zvuk, on='item_id', how='left')
    sub_zvuk = sub_zvuk[["user_id", "mean_user_rating_zvuk", "mean_item_rating"]].rename(columns={"mean_user_rating_zvuk": "mean_user_rating"})

    # Предикты для этих данных
    
    predictions_smm = (sub_smm.groupby('user_id')
                .apply(lambda x: predict_ranker(model_smm, x)))

    predictions_zvuk = (sub_zvuk.groupby('user_id')
                .apply(lambda x: predict_ranker(model_zvuk, x)))

    # Приведение данных к нужному для саабмита виду

    df_predictions_zvuk = pd.DataFrame({'user_id':predictions_zvuk.index, 'predictions':predictions_zvuk.values})
    df_predictions_smm = pd.DataFrame({'user_id':predictions_smm.index, 'predictions':predictions_smm.values})

    sub_smm_final = pd.read_parquet('submission_smm_als.parquet')
    sub_zvuk_final = pd.read_parquet('submission_zvuk_als.parquet')

    sub_zvuk_final['sorted_item_id'] = sub_zvuk_final.apply(lambda row: sort_item_ids(row, df_predictions_zvuk), axis=1)
    result_df_zvuk = sub_zvuk_final[['user_id', 'sorted_item_id']]

    sub_smm_final['sorted_item_id'] = sub_smm_final.apply(lambda row: sort_item_ids(row, df_predictions_smm), axis=1)
    result_df_smm = sub_smm_final[['user_id', 'sorted_item_id']]

    result_df_zvuk = result_df_zvuk.rename(columns={"sorted_item_id":"item_id"})
    result_df_smm = result_df_smm.rename(columns={"sorted_item_id":"item_id"})

    result_df_smm.to_parquet('submission_smm.parquet', index=False)
    result_df_zvuk.to_parquet('submission_zvuk.parquet', index=False)


def main():
    fit_als()

    for subset_name in cfg_data["dataset_names"]:
        predict(subset_name)

    fit_ranker()

if __name__ == "__main__":
    main()
