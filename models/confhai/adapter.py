# file: models/confhai/adapter.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression

# 导入您上传的源代码中的模型和工具函数
# 请确保路径正确，可能需要调整
from .models import Net
from .utils import train_confhai
from .utils import test_hai  # 我们借用test_hai中的逻辑来实现predict
from .utils import train_ips, train_confips


class ConfHAI_Adapter:
    """
    ConfHAI 适配器类。
    它封装了来自 utils.py 的函数式调用，提供了标准的 fit/predict 接口。
    """

    def __init__(self, gamma, C=0.0, lr=0.01, nepoch=1000, hidd=2):
        self.gamma = gamma
        self.C = C
        self.lr = lr
        self.nepoch = nepoch
        self.hidd = hidd

        # 这些将在fit方法中被训练和赋值
        self.trained_model_ = None
        self.trained_router_ = None
        self.propensity_model_ = None

    def fit(self, ds_train, ds_valid=None):  # ds_valid在这里没用到，但保持接口一致
        print(f"\n--- Fitting ConfHAI Adapter for gamma={self.gamma:.4f} ---")

        # 1. 准备数据加载器 (DataLoader)
        # ConfHAI的训练函数需要倾向性得分 q，我们需要先训练一个模型来估计它
        print("  - Training a temporary propensity score model for ConfHAI data format...")
        self.propensity_model_ = LogisticRegression(solver='liblinear')
        self.propensity_model_.fit(ds_train.x, ds_train.t)

        # 预测倾向性得分q
        q_train = self.propensity_model_.predict_proba(ds_train.x)

        # 创建符合 train_confhai 要求的 TensorDataset 和 DataLoader
        # loader需要提供 (x, q, t, y)
        confhai_dataset = TensorDataset(
            torch.from_numpy(ds_train.x).float(),
            torch.from_numpy(q_train).float(),
            torch.from_numpy(ds_train.t).float(),
            torch.from_numpy(ds_train.y).float()
        )
        # train_confhai 内部是以整个batch进行训练的
        loader = DataLoader(confhai_dataset, batch_size=len(confhai_dataset))

        # 2. 初始化模型和路由器
        # 使用您上传的 ConfoundL2D/models.py 中的 Net
        input_dim = ds_train.x.shape[1]
        model = Net(input_dim, num_classes=2, hidden=self.hidd)  # 2个动作
        router = Net(input_dim, num_classes=1, hidden=self.hidd)  # 1个输出用于sigmoid推迟概率

        # 3. 调用真实的训练函数
        print(f"  - Calling train_confhai from utils.py...")
        # train_confhai 返回训练好的 model, router, 和一个收敛指示器
        trained_model, trained_router, _ = train_confhai(
            model=model,
            router=router,
            input_dim=input_dim,
            output_dim=2,
            loader=loader,
            num_epochs=self.nepoch,
            lr=self.lr,
            gamma=self.gamma,
            C=self.C,
            hidd=self.hidd
        )

        self.trained_model_ = trained_model.cpu()
        self.trained_router_ = trained_router.cpu()

        print("--- ConfHAI Adapter fitting complete. ---")
        return self

    @torch.no_grad()
    def predict(self, ds_test) -> torch.Tensor:
        if not self.trained_model_ or not self.trained_router_:
            raise RuntimeError("The model has not been fitted yet. Call .fit() first.")

        print("  - Predicting with fitted ConfHAI model...")
        self.trained_model_.eval()
        self.trained_router_.eval()

        x_test_tensor = torch.from_numpy(ds_test.x).float()

        # 这里的逻辑借鉴了 utils.py 中的 test_hai 函数
        # 算法策略
        pred_algo = torch.argmax(self.trained_model_(x_test_tensor), dim=1)

        # 推迟决策
        # router输出一个logit，通过sigmoid变成概率，大于0.5则推迟
        defer_prob = torch.sigmoid(self.trained_router_(x_test_tensor))
        should_defer = (defer_prob > 0.5).squeeze()

        # 合成最终决策
        # 0: 不推迟, 1: 推迟
        # 我们需要返回 0, 1, 2 的格式
        final_preds = pred_algo.clone()
        final_preds[should_defer] = 2  # 将需要推迟的决策设为2

        return final_preds


class CRM_Adapter:
    """
    CRM (aka AO) 适配器类。
    封装了 train_ips 函数。
    """

    def __init__(self, lr=0.01, nepoch=1000, hidd=2):
        self.lr = lr
        self.nepoch = nepoch
        self.hidd = hidd
        self.trained_model_ = None
        self.propensity_model_ = None

    def _prepare_loader(self, ds_train):
        # 内部函数：为训练准备数据和倾向性得分
        print("  - CRM_Adapter: Training temporary propensity model...")
        self.propensity_model_ = LogisticRegression(solver='liblinear', max_iter=1000)
        self.propensity_model_.fit(ds_train.x, ds_train.t)
        q_train = self.propensity_model_.predict_proba(ds_train.x)

        dataset = TensorDataset(
            torch.from_numpy(ds_train.x).float(),
            torch.from_numpy(q_train).float(),
            torch.from_numpy(ds_train.t).float(),
            torch.from_numpy(ds_train.y).float()
        )
        return DataLoader(dataset, batch_size=len(dataset))

    def fit(self, ds_train):
        print("\n--- Fitting CRM_Adapter (AO) ---")
        loader = self._prepare_loader(ds_train)
        input_dim = ds_train.x.shape[1]

        print("  - Calling train_ips from utils.py...")
        self.trained_model_ = train_ips(
            input_dim=input_dim,
            output_dim=2,
            loader=loader,
            num_epochs=self.nepoch,
            lr=self.lr,
            hidd=self.hidd
        )
        print("--- CRM_Adapter fitting complete. ---")
        return self

    @torch.no_grad()
    def predict(self, ds_test) -> torch.Tensor:
        if not self.trained_model_:
            raise RuntimeError("CRM_Adapter has not been fitted yet.")

        self.trained_model_.eval()
        x_test_tensor = torch.from_numpy(ds_test.x).float()
        predictions = torch.argmax(self.trained_model_(x_test_tensor), dim=1)
        return predictions


class ConfAO_Adapter(CRM_Adapter):  # <--- 让 ConfAO 继承 CRM_Adapter 以复用 _prepare_loader
    """
    ConfAO 适配器类。
    封装了 train_confips 函数。
    """

    def __init__(self, gamma, lr=0.01, nepoch=1000, hidd=2):
        super().__init__(lr, nepoch, hidd)  # 调用父类的 __init__
        self.gamma = gamma

    def fit(self, ds_train):
        print(f"\n--- Fitting ConfAO_Adapter for gamma={self.gamma:.4f} ---")
        loader = self._prepare_loader(ds_train)
        input_dim = ds_train.x.shape[1]

        print("  - Calling train_confips from utils.py...")
        # train_confips 返回 model 和 con_ind
        model, _ = train_confips(
            input_dim=input_dim,
            output_dim=2,
            loader=loader,
            num_epochs=self.nepoch,
            lr=self.lr,
            gamma=self.gamma,
            hidd=self.hidd
        )
        self.trained_model_ = model
        print("--- ConfAO_Adapter fitting complete. ---")
        return self

    # predict 方法直接从父类 CRM_Adapter 继承，无需重写