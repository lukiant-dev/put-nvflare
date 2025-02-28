# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
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

import os
from typing import List, Union

from nvflare.apis.dxo import DXO, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import model_learnable_to_dxo
from nvflare.app_common.abstract.model_locator import ModelLocator
from uncover_xgboost.uncover_model_persistor import UncoverXGBModelPersistor
from uncover_xgboost.uncover_shareable_generator import XGBUncoverModelShareableGenerator

class UncoverXGBoostModelLocator(ModelLocator):
    def __init__(self):
        super().__init__()

    def get_model_names(self, fl_ctx: FLContext) -> List[str]:
        return ["xgboost_model"]

    def locate_model(self, model_name, fl_ctx: FLContext) -> Union[DXO, None]:
        if model_name == "xgboost_model":
            try:
                server_run_dir = fl_ctx.get_engine().get_workspace().get_app_dir(fl_ctx.get_job_id())
                model_path = os.path.join(server_run_dir, "xgboost_model.json")
                if not os.path.exists(model_path):
                    return None

                mp = UncoverXGBModelPersistor()
                mp._initialize(fl_ctx=fl_ctx)
                model_learnable = mp.load_model(fl_ctx=fl_ctx)
                sg = XGBUncoverModelShareableGenerator()
                shareable = sg.learnable_to_shareable(model_learnable, fl_ctx)

                # Create dxo and return
                return from_shareable(shareable)
            except Exception as e:
                self.log_error(fl_ctx, f"Error in retrieving {model_name}: {e}.", fire_event=False)
                return None
        else:
            self.log_error(fl_ctx, f"UncoverXGBoostModelLocator doesn't recognize name: {model_name}", fire_event=False)
            return None
