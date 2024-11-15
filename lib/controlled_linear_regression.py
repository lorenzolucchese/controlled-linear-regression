import numpy as np
from typing import Union
import torch

from sklearn.linear_model._base import LinearModel, LinearRegression

class LinearPredictionModel(LinearModel):
    def __init__(self, coef=None, intercept=None):
        super().__init__()
        if coef is not None:
            coef = np.array(coef)
            if intercept is None:
                intercept = np.zeros(coef.shape[0])
            else:
                pass
        else:
            if intercept is None:
                raise ValueError("Provide at least one of coef and intercept")
            else:
                pass
        self.intercept_ = intercept
        self.coef_ = coef
        
        self.intercept = intercept
        self.coef = coef

    def fit(self, X, y):
        raise NotImplementedError("model is only for prediction")
    
    def predict(self, X):
        if self.coef_ is None and X is not None:
            raise ValueError("model is constant, set X = None.")
        elif self.coef_ is None and X is None:
            return self.intercept_
        else:
            return super().predict(X)


def _fit_controlled_linear_regression_numpy(X: Union[np.ndarray, None], Y: np.ndarray, Z: Union[np.ndarray, None], fit_intercept: bool = True, method: str = 'control-joint-OLS') -> LinearPredictionModel:    
    # Note this has same logic as sklearn.linear_regression, i.e. fits multi-target to same features
    if Z is not None and Z.ndim == 1:
        Z = Z.reshape((-1, 1))
    if X is not None and X.ndim == 1:
        X = X.reshape((-1, 1))

    if X is not None:
        # add feature dimension
        if X.ndim == 1:
            X = X.reshape((-1, 1))
    if Z is not None:
        # add feature dimension
        if Z.ndim == 1:
            Z.reshape((-1, 1))
        if method == 'control-basic':
            Y_c = Y - Z @ np.linalg.inv(Z.T @ Z) @ Z.T @ Y

    lm_fit = LinearRegression(fit_intercept=fit_intercept)
    if X is not None and Z is not None:
        if method == 'control-basic':
            lm_fit.fit(X, Y_c)
        elif method == 'control-joint-OLS':            
            XZ = np.concatenate([X, Z], axis=-1)
            lm_fit.fit(XZ, Y)
        else:
            raise ValueError(f'Unknown method = {method}')
        coef_X = lm_fit.coef_[:, :X.shape[1]] if Y.ndim == 2 else lm_fit.coef_[:X.shape[1]]
        intercept = lm_fit.intercept_
    elif X is None and Z is not None:
        if method == 'control-basic':
            intercept = Y_c.mean(axis=0) if Y_c.ndim == 2 else Y_c.mean()
        elif method == 'control-joint-OLS':
            lm_fit.fit(Z, Y)            
            intercept = lm_fit.intercept_
        else:
            raise ValueError(f'Unknown method = {method}')
        coef_X = None
    elif X is not None and Z is None:
        lm_fit.fit(X, Y)
        coef_X = lm_fit.coef_
        intercept = lm_fit.intercept_
    else:
        coef_X = None
        if fit_intercept:
            intercept = Y.mean(axis=0) if Y.ndim == 2 else Y.mean()
        else:
            intercept = np.zeros(Y.shape[1]) if Y.ndim == 2 else 0.0

    lm_pred = LinearPredictionModel(coef=coef_X, intercept=intercept)
    return lm_pred


class TorchLinearRegression(LinearRegression):
    def __init__(self, parallel: bool = False, fit_intercept: bool = True):
        self.parallel = parallel
        self.fit_intercept = fit_intercept

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        assert X.shape[0] == y.shape[0]
        if self.parallel:
            # X has shape (n_samples, n_features, n_targets) and y has shape (n_samples, n_targets)
            assert X.ndim == 3 and y.ndim == 2
            assert X.shape[-1] == y.shape[-1]
        else:
            # X has shape (n_samples, n_features) and y has shape (n_samples,) or (n_samples, n_targets) 
            assert X.ndim == 2

        if self.fit_intercept:
            if self.parallel:
                X = torch.cat([torch.ones(X.shape[0], 1, X.shape[-1]).to(X.device), X], dim=1)
            else:
                X = torch.cat([torch.ones(X.shape[0], 1).to(X.device), X], dim=1)

        if self.parallel:
            # set n_targets as first dim
            X = X.permute(2, 0, 1)
            y = y.transpose(0, 1)

            # set non-invertible entries so that betas are set to zero (and intercept to mean)
            XtX = X.transpose(-1, -2) @ X
            mask = XtX.det() == 0
            XtX[mask] = torch.eye(XtX.shape[-1]).to(X.device)
            X[mask] = torch.zeros_like(X[0]).to(X.device)

            # beta has shape [n_targets, n_features (+ 1)]
            if self.fit_intercept:
                X[mask, 0] = 1.
            beta = torch.bmm(
                torch.inverse(XtX), 
                (X.transpose(-1, -2) @ y.unsqueeze(-1))
            ).squeeze(-1)
        else:
            beta = torch.inverse(X.T @ X) @ X.T @ y
            
            # beta has shape [n_features (+ 1)] or [n_targets, n_features (+ 1)]
            if y.ndim == 2:
                beta = beta.T

        if self.fit_intercept:
            if y.ndim == 2:
                self.intercept_ = beta[:, 0]
                self.coef_ = beta[:, 1:]
            else:
                self.intercept_ = beta[0]
                self.coef_ = beta[1:]
        else:
            self.coef_ = beta

        return self
    
    def predict(self, X: torch.Tensor):
        if self.parallel:
            assert X.ndim == 3 and self.coef_.ndim == 2
            # X has shape (n_samples, n_features, n_targets) and self.coef_ has shape (n_targets, n_features)
            X = X.permute(2, 0, 1)
            pred = (X @ self.coef_.unsqueeze(-1)).squeeze(-1)
            # pred has shape (n_samples, n_targets)
            pred = pred.transpose(0, 1)
        else:
            # X has shape (n_samples, n_features) and self.coef_ has shape (n_features,) or (n_targets, n_features)
            assert X.ndim == 2
            # pred has shape (n_samples, n_targets) or (n_samples,)
            pred = X @ self.coef_.T if self.coef_.ndim == 2 else X @ self.coef_
        if self.fit_intercept:
            # pred has shape (n_samples, n_targets) and self.intercept_ has shape (n_targets,) or pred has shape (n_samples,) and self.intercept_ is scalar
            return self.intercept_.unsqueeze(0) + pred if pred.ndim == 2 else self.intercept_ + pred
        else:
            return pred
    

class TorchLinearPredictionModel(LinearModel):
    def __init__(self, coef=None, intercept=None, parallel: bool = False):
        super().__init__()
        self.parallel = parallel
        if coef is not None:
            if not isinstance(coef, torch.Tensor):
                coef = torch.tensor(coef)
            if intercept is None:
                intercept = torch.zeros(coef.shape[0]).to(coef.device)
            else:
                pass
        else:
            if intercept is None:
                raise ValueError("Provide at least one of coef and intercept")
            else:
                pass
        self.intercept_ = intercept
        self.coef_ = coef
        
        self.intercept = intercept
        self.coef = coef

    def fit(self, X, y):
        raise NotImplementedError("model is only for prediction")
    
    def predict(self, X):
        if self.coef_ is None and X is not None:
            raise ValueError("model is constant, set X = None.")
        elif self.coef_ is None and X is None:
            return self.intercept_
        else:
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
            if self.parallel:
                # X is expected to have shape (n_samples, n_targets) or (n_samples, n_features, n_targets)
                # self.coef_ has shape (n_targets, n_features)
                # self.intercept_ is scaar or has shape (n_targets,)
                assert X.shape[-1] == self.coef_.shape[0]
                if X.ndim == 2:
                    X = X[:, None, :]
                return self.intercept_ + torch.einsum('nik,ki->nk', X, self.coef_)
            else:
                # X is expected to have shape (n_samples,) or (n_samples, n_features)
                # self.coef_ has shape (n_features,) or (n_targets, n_features)
                # self.intercept_ is scaar or has shape (n_targets,)
                if X.ndim == 1:
                    X = X[:, None]
                return self.intercept_ + X @ self.coef_.T if self.coef_.ndim == 2 else self.intercept_ + X @ self.coef_
        

def _fit_controlled_linear_regression_torch(X: Union[torch.Tensor, None], Y: torch.Tensor, Z: Union[torch.Tensor, None], fit_intercept: bool = True, parallel: bool = False, method: str = 'control-joint-OLS') -> TorchLinearPredictionModel:    
    if X is not None:
        if parallel:
            assert Y.ndim == 2
            assert X.shape[0] == Y.shape[0] and X.shape[-1] == Y.shape[-1]
            # add feature dimension
            if X.ndim == 2:
                X = X[:, None, :]
        else:
            # add feature dimension
            if X.ndim == 1:
                X = X[:, None]
    if Z is not None:
        if parallel:
            assert Y.ndim == 2
            assert Z.shape[0] == Y.shape[0] and Z.shape[-1] == Y.shape[-1]
            # add feature dimension
            if Z.ndim == 2:
                Z = Z[:, None, :]
            if method == 'control-basic':
                Y_c = Y - torch.einsum('nik,kij,jk->nk', Z, torch.inverse(torch.einsum('nik,njk->kij', Z, Z)), torch.einsum('njk,nk->jk', Z, Y))
        else:
            # add feature dimension
            if Z.ndim == 1:
                Z = Z[:, None]
            if method == 'control-basic':
                Y_c = Y - Z @ torch.inverse(Z.T @ Z) @ Z.T @ Y

    lm_fit = TorchLinearRegression(fit_intercept=fit_intercept, parallel=parallel)
    if X is not None and Z is not None:
        if method == 'control-basic':
            lm_fit.fit(X, Y_c)
        elif method == 'control-joint-OLS':
            XZ = torch.cat([X, Z], dim=1)
            lm_fit.fit(XZ, Y)
        else:
            raise ValueError(f'Unknown method = {method}')
        coef_X = lm_fit.coef_[:, :X.shape[1]] if Y.ndim == 2 else lm_fit.coef_[:X.shape[1]]
        intercept = lm_fit.intercept_
    elif X is None and Z is not None:
        if method == 'control-basic':
            intercept = torch.mean(Y_c, dim=0) if Y_c.ndim == 2 else torch.mean(Y_c)
        elif method == 'control-joint-OLS':
            lm_fit.fit(Z, Y)            
            intercept = lm_fit.intercept_
        else:
            raise ValueError(f'Unknown method = {method}')
        coef_X = None
    elif X is not None and Z is None:
        lm_fit.fit(X, Y)
        coef_X = lm_fit.coef_
        intercept = lm_fit.intercept_
    else:
        coef_X = None
        if fit_intercept:
            intercept = torch.mean(Y, dim=0) if Y.ndim == 2 else torch.mean(Y)
        else:
            intercept = torch.zeros(Y.shape[1]).to(Y.device) if Y.ndim == 2 else 0.0

    lm_pred = TorchLinearPredictionModel(coef=coef_X, intercept=intercept, parallel=parallel)
    return lm_pred


def fit_controlled_linear_regression(X: Union[np.ndarray, torch.Tensor, None], Y: Union[np.ndarray, torch.Tensor], Z: Union[np.ndarray, torch.Tensor, None], fit_intercept: bool = True, parallel: bool = False, method: str = 'control-joint-OLS') -> Union[LinearPredictionModel, TorchLinearPredictionModel]:    
    if all([isinstance(M, np.ndarray) or M is None for M in [X, Y, Z]]):
        if parallel:
            raise ValueError('parallel not supported for numpy arrays, current implementation uses scikit-learn which uses all rgressor across all targets.')
        return _fit_controlled_linear_regression_numpy(X, Y, Z, fit_intercept=fit_intercept, method=method)
    elif all([isinstance(M, torch.Tensor) or M is None for M in [X, Y, Z]]):
        return _fit_controlled_linear_regression_torch(X, Y, Z, fit_intercept=fit_intercept, parallel=parallel, method=method)
    else:
        raise ValueError('X, Y and Z must all be of the same type (or None): np.ndarray or torch.Tensor.')