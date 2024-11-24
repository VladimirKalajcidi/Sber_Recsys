"""Implicit ALS and BPR models embedding."""

import pickle
import time
from pathlib import Path
from typing import Optional, Mapping, Any, Tuple, Union
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import implicit
from tqdm import tqdm
import pandas as pd
from implicit.als import AlternatingLeastSquares
import xgboost as xgb


def create_sparse_matrix(
    data: pd.DataFrame,
    rating_col: str,
    user_col: str,
    item_col: str,
    weighted: bool = True,
    data_shape: tuple = None,
    sparse_type: str = "csr",
) -> Union[csr_matrix, Tuple[csr_matrix, csr_matrix]]:
    """
    Create a sparse matrix from the input DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        rating_col (str): The name of the column containing the ratings.
        user_col (str): The name of the column containing the user IDs.
        item_col (str): The name of the column containing the item IDs.
        weighted (bool, optional): Whether to create a weighted matrix based
        on ratings (default=False).
        data_shape (tuple): Desired shape of sparse matrix.
        sparse_type (str): type of the sparse matrix

    Returns:
        Union[csr_matrix, tuple[csr_matrix, csr_matrix]]: The sparse matrix or tuple of sparse
        matrix and weights.
    """
    if sparse_type == "csr":
        matrix = csr_matrix
    elif sparse_type == "coo":
        matrix = coo_matrix
    else:
        matrix = csr_matrix
    if data_shape is not None:
        interactions = matrix(
            (
                np.where(data[rating_col] > 0, 1, 0),
                (data[user_col].to_numpy(), data[item_col].to_numpy()),
            ),
            shape=data_shape,
        )
    else:
        interactions = matrix(
            (
                np.where(data[rating_col] > 0, 1, 0),
                (data[user_col].to_numpy(), data[item_col].to_numpy()),
            ),
        )

    if weighted:
        weights = interactions.copy()
        weights.data = data[rating_col].to_numpy()
        return interactions, weights
    else:
        return interactions, None


def get_saving_path(path: Path) -> Path:
    """Save the model. If previously saved model exists,
    creating a new directory with higher ordering number

    Args:
        path (Path): Original path to the model.

    Returns:
        Path: New path with the name.
    """
    # Check if the directory exists
    if not path.exists():
        path.mkdir(parents=True)
    # Check max model version and create new dir
    i = 0
    for model_path in path.glob("model_*"):
        i = max(int(model_path.stem.split("_")[1]) + 1, i)
    model_dir = path.joinpath(f"model_{i}")
    model_dir.mkdir()
    return model_dir


DTYPE = np.float32


class ALSModel:
    """
    Implicit Model Bench base class for model training, optimization, and evaluation.
    """

    def __init__(self, cfg_data, **model_params: Optional[Mapping[str, Any]]) -> None:
        """
        Initialize the ImplicitAlsBench instance.

        Args:
            model (AlternatingLeastSquares or BayesianPersonalizedRanking):
            The ALS or BPR model instance.
            model_params (Mapping[str, Any]): Model parameters.
        """
        self.cfg_data = cfg_data
        self.model = AlternatingLeastSquares(**model_params)
        self.model_params = model_params

    def fit(
        self,
        interactions: coo_matrix,
        weights: Optional[coo_matrix] = None,
        show_progress: bool = True,
    ) -> None:
        """
        Fit the Implicit ALS or BPR model to the training data.

        Args:
            interactions (coo_matrix): Training interactions matrix (user-item interactions).
            weights (coo_matrix, optional): Weight matrix for training interactions.
            show_progress (bool, optional): Whether to show the progress during model training.
            callback (function, optional): Callback function to be executed during training.

        Returns:
            None
        """
        # We need this in the COO format.
        interactions, _ = create_sparse_matrix(
            interactions,
            self.cfg_data["rating_column"],
            self.cfg_data["user_column"],
            self.cfg_data["item_column"],
            False,
            None,
            "coo",
        )
        self.shape = interactions.shape

        if interactions.dtype != DTYPE:
            interactions.data = interactions.data.astype(DTYPE)

        weight_data = self._process_weight(interactions, weights)
        self.model.fit(
            user_items=weight_data.tocsr(),
            show_progress=show_progress,
        )

    def recommend_k(
        self,
        test: pd.DataFrame,
        k: int,
        filter_already_liked_items: bool = True,
        filter_items=None,
        recalculate_user: bool = False,
    ) -> np.ndarray:

        """
        Recommend top k items for users.

        Args:
            test: pd.DataFrame: 
                test dataset for removing seen items from prediction
            k (int): The number of results to return.
            train_interactions (csr_matrix or coo_matrix):
                Sparse matrix of shape (users, number_items)
                representing the user-item interactions for training.
            filter_already_liked_items (bool, optional): When True, don't return items present in
                the training set that were rated by the specified user.
            filter_items (array_like, optional): List of extra item IDs to filter out from
                the output.
            recalculate_user (bool, optional): When True, recalculates
                factors for a batch of users.

        Returns:
            np.ndarray: 2-dimensional array with a row of item IDs for each user.
        """
        userids = np.sort(test.user_id.unique())

        interactions, _ = create_sparse_matrix(
            test, 
            self.cfg_data['rating_column'], self.cfg_data['user_column'], self.cfg_data['item_column'], 
            False, None, "coo"
        )
        interactions = interactions.tocsr()
        
        ids, _ = self.model.recommend(
            userid=userids,
            user_items=interactions[userids],
            N=k,
            filter_already_liked_items=filter_already_liked_items,
            filter_items=filter_items,
            recalculate_user=recalculate_user,
        )
        return ids, userids

    def _process_weight(self, interactions, weights) -> coo_matrix:
        """
        Process the weights matrix.

        This method allows you to feed interactions and weights separately
        to models from implicit libraries. If weights is None, interactions used.
        If weights is not None, than shape and DTYPE is checked to match
        the interactions matrix.and weights returned.

        Args:
            interactions (coo_matrix): Sparse interactions matrix.
            weights (Optional[coo_matrix]): Sparse sample weight matrix.

        Returns:
            coo_matrix: Processed weight matrix.

        Raises:
            ValueError: If the shape and order of the weights matrix do not match
                the interactions matrix.
        """
        if not isinstance(interactions, coo_matrix):
            if not isinstance(interactions, csr_matrix):
                raise ValueError("interactions must be a COO matrix.")
            interactions = interactions.tocoo()

        if weights is not None:
            if not isinstance(weights, coo_matrix):
                if not isinstance(weights, csr_matrix):
                    raise ValueError("Sample_weight must be a COO matrix.")
                weights = weights.tocoo()

            if weights.shape != interactions.shape:
                raise ValueError(
                    "Sample weight and interactions matrices must be the same shape"
                )

            if not (
                np.array_equal(interactions.row, weights.row)
                and np.array_equal(interactions.col, weights.col)
            ):
                raise ValueError(
                    "Sample weight and interaction matrix "
                    "entries must be in the same order"
                )

            if weights.data.dtype != DTYPE:
                weight_data = weights.data.astype(DTYPE)
            else:
                weight_data = weights.data
        else:
            if np.array_equiv(interactions.data, 1.0):
                # Re-use interactions data if they are all ones
                weight_data = interactions.data
            else:
                # Otherwise allocate a new array of ones
                weight_data = np.ones_like(interactions.data, dtype=DTYPE)
        return coo_matrix((weight_data, (interactions.row, interactions.col)))

    def save_model(self, path: Path) -> None:
        """
        Save the Implicit ALS model to a file.

        Args:
            path (str): Path to the directory where the model file should be saved.

        Returns:
            None
        """
        model_dir = get_saving_path(path)

        with open(model_dir.joinpath("model.pcl"), "wb") as file:
            pickle.dump(self.model, file)
        with open(model_dir.joinpath("cfg_data.pcl"), "wb") as file:
            pickle.dump(self.cfg_data, file)
        with open(model_dir.joinpath("params.pcl"), "wb") as file:
            pickle.dump(self.model_params, file)

    @classmethod
    def load_model(cls, path: Path) -> None:
        """
        Save the Implicit ALS model to a file.

        Args:
            path (str): Path to the directory where the model file should be saved.

        Returns:
            None
        """
        model_dir = get_saving_path(path)

        with open(model_dir.joinpath("model.pcl"), "rb") as file:
            model = pickle.load(file)
        with open(model_dir.joinpath("cfg_data.pcl"), "rb") as file:
            cfg = pickle.load(file)
        with open(model_dir.joinpath("params.pcl"), "rb") as file:
            model_params = pickle.load(file)
        obj = cls(cfg, **model_params)
        obj.model = model
        return obj


class XGBRanker:
    def __init__(self, cfg_data, **model_params: Optional[Mapping[str, Any]]) -> None:
        self.cfg_data = cfg_data
        self.model = xgb.XGBRanker(**model_params)
        self.model_params = model_params
