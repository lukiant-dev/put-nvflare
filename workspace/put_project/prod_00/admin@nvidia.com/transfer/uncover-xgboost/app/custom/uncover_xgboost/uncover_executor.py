# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from abc import ABC, abstractmethod
from typing import Tuple

import xgboost as xgb
import numpy as np

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey,Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable

from nvflare.app_common.app_constant import AppConstants
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.security.logging import secure_format_exception
from nvflare.app_common.widgets.streaming import AnalyticsSender
from uncover_xgboost.uncover_shareable_generator import update_model, XGBUncoverModelShareableGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import torch.utils.tensorboard as tb

class FedXGBTreeUncoverExecutor(Executor, ABC):
    def __init__(
        self,
        dataset_root,
        all_rows_num,
        random_seed,
        analytic_sender_id: str,
        training_mode: str = "bagging", 
        lr_scale=1,  
        num_client_bagging: int = 1,
        lr_mode: str = "uniform",
        local_model_path: str = "local_model.json",
        global_model_path: str = "model_global.json",
        learning_rate: float = 0.1,
        objective: str = "binary:logistic",
        num_local_parallel_tree: int = 1,
        local_subsample: float = 1,
        max_depth: int = 8,
        eval_metric: str = "auc",
        nthread: int = 16,
        tree_method: str = "auto",
        train_task_name: str = AppConstants.TASK_TRAIN,
    ):
        super().__init__()
        self.client_id = None
        self.writer = None
        self.all_rows_num = all_rows_num
        self.random_seed = random_seed
        
        self.training_mode = training_mode
        self.num_client_bagging = num_client_bagging
        self.lr = None
        self.lr_scale = lr_scale
        self.base_lr = learning_rate
        self.lr_mode = lr_mode
        self.num_local_parallel_tree = num_local_parallel_tree
        self.local_subsample = local_subsample
        self.local_model_path = local_model_path
        self.global_model_path = global_model_path
        self.objective = objective
        self.max_depth = max_depth
        self.eval_metric = eval_metric
        self.nthread = nthread
        self.tree_method = tree_method
        self.train_task_name = train_task_name
        self.num_local_round = 1

        self.bst = None
        self.global_model_as_dict = None
        self.config = None
        self.local_model = None

        self.dmat_train = None
        self.dmat_valid = None
        
        self.dataset_root= dataset_root
        self.analytic_sender_id = analytic_sender_id
        self.best_score = 0

        # use dynamic shrinkage - adjusted by personalized scaling factor
        if lr_mode not in ["uniform", "scaled"]:
            raise ValueError(f"Only support [uniform] or [scaled] mode, but got {lr_mode}")

    def load_data(self):
        
        self.data_file = os.path.join(self.dataset_root, f"{self.client_id[-1]}.csv")

        full_dataset = pd.read_csv(self.data_file)
        full_dataset = full_dataset.drop(columns=['did_the_patient_expire_in_hospital'])
        full_labels = full_dataset['admission_disposition']
        full_dataset = full_dataset.drop(columns=['admission_disposition'])
        
        self.train_set, self.val_set, self.train_labels, self.val_labels = train_test_split(full_dataset, full_labels, train_size=0.8, random_state=self.random_seed)

        # training
        dmat_train = xgb.DMatrix(self.train_set, label=self.train_labels)

        # validation
        dmat_valid = xgb.DMatrix(self.val_set, label=self.val_labels)

        self.lr_scale = len(full_dataset.values) / self.all_rows_num
        
        return dmat_train, dmat_valid

    def initialize(self, fl_ctx: FLContext):
        # set the paths according to fl_ctx
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_dir = ws.get_app_dir(fl_ctx.get_job_id())
        self.local_model_path = os.path.join(app_dir, self.local_model_path)
        self.global_model_path = os.path.join(app_dir, self.global_model_path)

        # get and print the args
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized with args: \n {fl_args}",
        )

        try:
            engine = fl_ctx.get_engine()
            self.analytic_sender = engine.get_component(self.analytic_sender_id)
            self.writer = self.analytic_sender

            if not isinstance(self.analytic_sender, AnalyticsSender):
                raise TypeError(f"Analytic sender must be AnalyticsSender type. Got: {type(self.analytic_sender)}")
        except Exception as e:
            self.log_exception(fl_ctx, f"Uncover Executor initialize exception: {secure_format_exception(e)}")

        if self.training_mode not in ["cyclic", "bagging"]:
            self.system_panic(f"Only support [cyclic] or [bagging] mode, but got {self.training_mode}", fl_ctx)
            return

        # load data and lr_scale, this is task/site-specific
        self.dmat_train, self.dmat_valid = self.load_data()
        self.lr = self._get_effective_learning_rate()

    def _get_effective_learning_rate(self):
        if self.training_mode == "bagging":
            # Bagging mode
            if self.lr_mode == "uniform":
                # uniform lr, global learning_rate scaled by num_client_bagging for bagging
                lr = self.base_lr / self.num_client_bagging
            else:
                # scaled lr, global learning_rate scaled by data size percentage
                lr = self.base_lr * self.lr_scale
        else:
            # Cyclic mode, directly use the base learning_rate
            lr = self.base_lr
        return lr

    def _get_train_params(self):
        param = {
            "objective": self.objective,
            "eta": self.lr,
            "max_depth": self.max_depth,
            "eval_metric": self.eval_metric,
            "nthread": self.nthread,
            "num_parallel_tree": self.num_local_parallel_tree,
            "subsample": self.local_subsample,
            "tree_method": self.tree_method,
        }
        return param

    def _local_boost_bagging(self, fl_ctx: FLContext):
        eval_results = self.bst.eval_set(
            evals=[(self.dmat_train, "train"), (self.dmat_valid, "valid")], iteration=self.bst.num_boosted_rounds() - 1
        )

        train_pred = np.rint(self.bst.predict(self.dmat_train))
        valid_pred = np.rint(self.bst.predict(self.dmat_valid))

        train_acc = accuracy_score(self.train_labels.to_numpy(), train_pred)
        valid_acc = accuracy_score(self.val_labels.to_numpy(), valid_pred)

        self.log_info(fl_ctx, eval_results)
        quality_metric = float(eval_results.split("\t")[2].split(":")[1])
        for i in range(self.num_local_round):
            self.bst.update(self.dmat_train, self.bst.num_boosted_rounds())

        # extract newly added self.num_local_round using xgboost slicing api
        bst = self.bst[self.bst.num_boosted_rounds() - self.num_local_round : self.bst.num_boosted_rounds()]

        self.log_info(
            fl_ctx,
            f"Global METRIC {quality_metric}",
        )
        if self.writer:
            # note: writing auc before current training step, for passed in global model
            self.writer.add_scalar(
                f"uncover_xgboost_{self.eval_metric}", quality_metric, int(self.bst.num_boosted_rounds())
            )
            self.writer.add_scalar(
                "uncover_xgboost_train_acc", train_acc, int(self.bst.num_boosted_rounds())
            )
            # self.writer.add_scalar(
            #     "uncover_xgboost_valid_acc", valid_acc, int(self.bst.num_boosted_rounds())
            # )
        return bst, eval_results, quality_metric

    def _local_boost_cyclic(self, fl_ctx: FLContext):
        # Cyclic mode
        # starting from global model
        # return the whole boosting tree series
        self.bst.update(self.dmat_train, self.bst.num_boosted_rounds())
        eval_results = self.bst.eval_set(
            evals=[(self.dmat_train, "train"), (self.dmat_valid, "valid")], iteration=self.bst.num_boosted_rounds() - 1
        )
        self.log_info(fl_ctx, eval_results)
        auc = float(eval_results.split("\t")[2].split(":")[1])
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} AUC after training: {auc}",
        )
        if self.writer:
            self.writer.add_scalar("AUC", auc, self.bst.num_boosted_rounds() - 1)
        return self.bst, auc

    def train(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        acc = 0 
        if abort_signal.triggered:
            self.finalize(fl_ctx)
            return make_reply(ReturnCode.TASK_ABORTED)

        # retrieve current global model download from server's shareable
        dxo = from_shareable(shareable)
        model_update = dxo.data

        # xgboost parameters
        param = self._get_train_params()

        if not self.bst:
            # First round
            self.log_info(
                fl_ctx,
                f"Client {self.client_id} initial training from scratch",
            )
            if not model_update:
                bst = xgb.train(
                    param,
                    self.dmat_train,
                    num_boost_round=self.num_local_round,
                    evals=[(self.dmat_valid, "validate"), (self.dmat_train, "train")],
                )
            else:
                loadable_model = bytearray(model_update["model_data"])
                bst = xgb.train(
                    param,
                    self.dmat_train,
                    num_boost_round=self.num_local_round,
                    xgb_model=loadable_model,
                    evals=[(self.dmat_valid, "validate"), (self.dmat_train, "train")],
                )
            self.config = bst.save_config()
            self.bst = bst
        else:
            self.log_info(
                fl_ctx,
                f"Client {self.client_id} model updates received from server",
            )
            if self.training_mode == "bagging":
                model_updates = model_update["model_data"]
                for update in model_updates:
                    self.global_model_as_dict = update_model(self.global_model_as_dict, json.loads(update))

                loadable_model = bytearray(json.dumps(self.global_model_as_dict), "utf-8")
            else:
                loadable_model = bytearray(model_update["model_data"])

            self.log_info(
                fl_ctx,
                f"Client {self.client_id} converted global model to json ",
            )

            self.bst.load_model(loadable_model)
            self.bst.load_config(self.config)

            self.log_info(
                fl_ctx,
                f"Client {self.client_id} loaded global model into booster ",
            )

            # train local model starting with global model
            if self.training_mode == "bagging":
                bst, eval_results, acc = self._local_boost_bagging(fl_ctx)
            else:
                bst = self._local_boost_cyclic(fl_ctx)

        self.local_model = bst.save_raw("json")

        if acc > self.best_score:
            self.best_score = acc
            with open(self.local_model_path, "w") as f:
                # save bytearray to file 
                f.write(self.local_model.decode("utf-8"))
                
        # report updated model in shareable
        dxo = DXO(data_kind=DataKind.WEIGHTS, data={"model_data": self.local_model})
        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        new_shareable = dxo.to_shareable()

        if self.writer:
            self.writer.flush()
        return new_shareable

    def finalize(self, fl_ctx: FLContext):
        # freeing resources in finalize avoids seg fault during shutdown of gpu mode
        del self.bst
        del self.dmat_train
        del self.dmat_valid
        self.log_info(fl_ctx, "Freed training resources")

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        self.log_info(fl_ctx, f"Client trainer got task: {task_name}")

        try:
            if task_name == "train":
                return self.train(shareable, fl_ctx, abort_signal)
            elif task_name == "validate":
                return self.validate(shareable, fl_ctx, abort_signal)
            elif task_name =="submit_model":
                return self.submit_model(shareable, fl_ctx, abort_signal)
            else:
                self.log_error(fl_ctx, f"Could not handle task: {task_name}")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            # Task execution error, return EXECUTION_EXCEPTION Shareable
            self.log_exception(fl_ctx, f"execute exception: {secure_format_exception(e)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.initialize(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.finalize(fl_ctx)


    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get validation information
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(AppConstants.MODEL_OWNER)
        if model_owner:
            self.log_info(fl_ctx, f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()}")
        else:
            model_owner = "global_model"  # evaluating global model during training

        # retrieve current global model download from server's shareable
        dxo = from_shareable(shareable)
        model_update = dxo.data

        # xgboost parameters
        param = self._get_train_params()
        
        loadable_model = bytearray(model_update["model_data"])
        self.bst.load_model(loadable_model)
        self.bst.load_config(self.config)
            
        train_pred = np.rint(self.bst.predict(self.dmat_train))
        val_pred = np.rint(self.bst.predict(self.dmat_valid))

        self.log_info(fl_ctx, f"TRAINP type: {type(train_pred)} {train_pred}")
        self.log_info(fl_ctx, f"TRAINL type: {type(self.train_labels)} {self.train_labels.to_numpy()}")

        train_acc = accuracy_score(self.train_labels.to_numpy(), train_pred)
        val_acc = accuracy_score(self.val_labels.to_numpy(), val_pred)

        # train_acc = sum(train_pred == self.train_labels) / len(self.train_labels)
        # val_acc = sum(val_pred == self.val_labels) / len(self.val_labels)

        eval_results = self.bst.eval_set(
            evals=[(self.dmat_train, "train"), (self.dmat_valid, "valid")], iteration=self.bst.num_boosted_rounds() - 1
        )
        self.log_info(fl_ctx, f"VALIDATION: {eval_results}")
        train_auc = float(eval_results.split("\t")[1].split(":")[1])
        val_auc = float(eval_results.split("\t")[2].split(":")[1])

        self.log_info(fl_ctx, "Evaluation finished. Returning shareable")

        val_results = {"train_accuracy": train_acc, "val_accuracy": val_acc}
        self.log_info(fl_ctx, f"XVALIDATION ACCURACY: {val_results}")

        metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
        return metric_dxo.to_shareable()

    def submit_model(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if os.path.exists(self.local_model_path):
            self.logger.info(f"Loading model for local submission: {self.local_model_path}")
            with open(self.local_model_path, "r") as json_file:
                model = json.load(json_file)
            model_learnable = make_model_learnable(weights=model, meta_props=dict())
            sg = XGBUncoverModelShareableGenerator()
            shareable = sg.learnable_to_shareable(model_learnable, fl_ctx)
            return shareable
        else:
            self.logger.error("No local model found.")
            return

        

        