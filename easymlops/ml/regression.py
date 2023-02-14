from ..base import *
import scipy.sparse as sp
import numpy as np
import pandas as pd


class RegressionBase(PipeObject):
    def __init__(self, y: series_type = None, cols="all", pred_name="pred", skip_check_transform_type=True,
                 drop_input_data=True,
                 native_init_params=None, native_fit_params=None,
                 **kwargs):
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        self.cols = cols
        self.drop_input_data = drop_input_data
        self.y = copy.copy(y)
        self.pred_name = pred_name
        # 底层模型自带参数
        self.native_init_params = native_init_params
        self.native_fit_params = native_fit_params
        if self.native_init_params is None:
            self.native_init_params = dict()
        if self.native_fit_params is None:
            self.native_fit_params = dict()

    def before_fit(self, s: dataframe_type) -> dataframe_type:
        s = super().before_fit(s)
        assert self.y is not None
        if str(self.cols).lower() in ["none", "all", "null", "nan"]:
            self.cols = s.columns.tolist()
            return s
        else:
            assert type(self.cols) == list
            return s[self.cols]

    def before_transform(self, s: dataframe_type) -> dataframe_type:
        s = super().before_transform(s)
        return s[self.cols]

    def transform(self, s: dataframe_type) -> dataframe_type:
        if not self.drop_input_data:
            s_ = self.before_transform(s)
            s_ = self.after_transform(self._transform(s_))
            for col in s_.columns:
                s[col] = s_[col]
            return s
        else:
            return self.after_transform(self._transform(self.before_transform(s)))

    def before_transform_single(self, s: dict_type) -> dict_type:
        s = super().before_transform_single(s)
        return self.extract_dict(s, self.cols)

    def transform_single(self, s: dict_type) -> dict_type:
        if not self.drop_input_data:
            s_ = self.before_transform_single(s)
            s_ = self.after_transform_single(self._transform_single(s_))
            s.update(s_)
            return s
        else:
            return self.after_transform_single(self._transform_single(self.before_transform_single(s)))

    def _fit(self, s):
        return self

    def _transform(self, s):
        return s

    def _transform_single(self, s: dict_type):
        input_dataframe = pd.DataFrame([s])
        return self._transform(input_dataframe).to_dict("record")[0]

    def _get_params(self) -> dict_type:
        return {"pred_name": self.pred_name, "cols": self.cols,
                "drop_input_data": self.drop_input_data, "native_init_params": self.native_init_params,
                "native_fit_params": self.native_fit_params}

    def _set_params(self, params: dict):
        self.pred_name = params["pred_name"]
        self.cols = params["cols"]
        self.drop_input_data = params["drop_input_data"]
        self.native_init_params = params["native_init_params"]
        self.native_fit_params = params["native_fit_params"]

    @staticmethod
    def pd2csr(data: dataframe_type):
        try:
            sparse_matrix = sp.csr_matrix(data.sparse.to_coo().tocsr(), dtype=np.float32)
        except:
            sparse_matrix = sp.csr_matrix(data, dtype=np.float32)
        return sparse_matrix

    @staticmethod
    def pd2dense(data: dataframe_type):
        return data.values


class LGBMRegression(RegressionBase):
    """
    input type:pandas.dataframe
    input like:
    | 0 | 1 | 2 | 3 |
    |0.2|0.3|0.4|0.5|
    |0.3|0.2|0.6|0.1|
    label type:pandas:series
    label like:
    |1|
    |0|
    |0|
    -------------------------
    output type:pandas.dataframe
    output like:
    |pred|
    |0.5 |
    |0.5 |
    """

    def __init__(self, y=None, verbose=-1, objective="regression",
                 use_faster_predictor=True, **kwargs):
        super().__init__(y=y, **kwargs)
        self.native_init_params.update({
            'objective': objective,
            'verbose': verbose
        })
        self.lgb_model = None
        self.use_faster_predictor = use_faster_predictor
        self.lgb_model_faster_predictor_params = None
        self.lgb_model_faster_predictor = None

    def _fit(self, s: dataframe_type) -> dataframe_type:
        import lightgbm as lgb
        s_ = self.pd2csr(s)  # 转稀疏矩阵后会丢失columns信息
        self.lgb_model = lgb.train(params=self.native_init_params, train_set=lgb.Dataset(data=s_, label=self.y),
                                   **self.native_fit_params)
        if self.use_faster_predictor:
            from ..utils import FasterLgbSinglePredictor
            self.lgb_model_faster_predictor_params = self.lgb_model.dump_model()
            self.lgb_model_faster_predictor_params["feature_names"] = s.columns.tolist()
            self.lgb_model_faster_predictor = FasterLgbSinglePredictor(
                model=self.lgb_model_faster_predictor_params, cache_num=10)
        return self

    def _transform(self, s: dataframe_type) -> dataframe_type:
        s = self.pd2csr(s)
        result = pandas.DataFrame(self.lgb_model.predict(s), columns=[self.pred_name])
        return result

    def _transform_single(self, s: dict_type):
        if self.use_faster_predictor:
            return {self.pred_name: self.lgb_model_faster_predictor.predict(s).get("score")}
        else:
            input_dataframe = pd.DataFrame([s])
            return self._transform(input_dataframe).to_dict("record")[0]

    def _get_params(self) -> dict_type:
        params = {"lgb_model": self.lgb_model, "use_faster_predictor": self.use_faster_predictor}
        if self.use_faster_predictor:
            params["lgb_model_faster_predictor_params"] = self.lgb_model_faster_predictor_params
        return params

    def _set_params(self, params: dict_type):
        self.lgb_model = params["lgb_model"]
        self.use_faster_predictor = params["use_faster_predictor"]
        if self.use_faster_predictor:
            from ..utils import FasterLgbSinglePredictor
            self.lgb_model_faster_predictor_params = params["lgb_model_faster_predictor_params"]
            self.lgb_model_faster_predictor = FasterLgbSinglePredictor(
                model=self.lgb_model_faster_predictor_params, cache_num=10)

    def get_contrib(self, s: dict_type) -> dict_type:
        """
        获取sabaas特征重要性
        """
        assert self.use_faster_predictor
        return self.lgb_model_faster_predictor.predict(s).get("contrib")
