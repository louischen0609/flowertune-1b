"""flowertune-1B-v1: A Flower / FlowerTune app."""

import os
from datetime import datetime
import random
import numpy as np
import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.common.config import unflatten_dict
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from omegaconf import DictConfig

from flowertune_1b_v1.models import get_model, get_parameters, set_parameters
from flowertune_1b_v1.dataset import replace_keys
from flwr.server.strategy import FedAvg

# 添加 LM Eval Harness 相關導入
import lm_eval
from lm_eval.models.huggingface import HFLM

def seed_all(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 時使用
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[INFO] Global seed set to: {seed}")

def get_mmlu_tasks_by_category(subset="all"):
    """
    根據類別返回 MMLU 任務列表
    """
    stem_tasks = [
        "mmlu_abstract_algebra", "mmlu_astronomy", "mmlu_college_biology",
        "mmlu_college_chemistry", "mmlu_college_computer_science", 
        "mmlu_college_mathematics", "mmlu_college_physics", "mmlu_computer_security",
        "mmlu_conceptual_physics", "mmlu_econometrics", "mmlu_electrical_engineering",
        "mmlu_elementary_mathematics", "mmlu_formal_logic", "mmlu_high_school_biology",
        "mmlu_high_school_chemistry", "mmlu_high_school_computer_science",
        "mmlu_high_school_mathematics", "mmlu_high_school_physics", 
        "mmlu_high_school_statistics", "mmlu_machine_learning"
    ]
    
    humanities_tasks = [
        "mmlu_formal_logic", "mmlu_high_school_european_history", 
        "mmlu_high_school_us_history", "mmlu_high_school_world_history",
        "mmlu_international_law", "mmlu_jurisprudence", "mmlu_logical_fallacies",
        "mmlu_moral_disputes", "mmlu_moral_scenarios", "mmlu_philosophy",
        "mmlu_prehistory", "mmlu_professional_law", "mmlu_world_religions"
    ]
    
    social_tasks = [
        "mmlu_econometrics", "mmlu_high_school_geography", 
        "mmlu_high_school_government_and_politics", "mmlu_high_school_macroeconomics",
        "mmlu_high_school_microeconomics", "mmlu_high_school_psychology",
        "mmlu_human_sexuality", "mmlu_professional_psychology", 
        "mmlu_public_relations", "mmlu_security_studies", "mmlu_sociology",
        "mmlu_us_foreign_policy"
    ]
    
    other_tasks = [
        "mmlu_anatomy", "mmlu_business_ethics", "mmlu_clinical_knowledge",
        "mmlu_college_medicine", "mmlu_global_facts", "mmlu_human_aging",
        "mmlu_management", "mmlu_marketing", "mmlu_medical_genetics",
        "mmlu_miscellaneous", "mmlu_nutrition", "mmlu_professional_accounting",
        "mmlu_professional_medicine", "mmlu_virology"
    ]
    
    all_tasks = stem_tasks + humanities_tasks + social_tasks + other_tasks
    
    if subset == "stem":
        return stem_tasks
    elif subset == "humanities":
        return humanities_tasks
    elif subset == "social":
        return social_tasks
    elif subset == "other":
        return other_tasks
    else:
        return all_tasks

def run_mmlu_evaluation(model, model_path, eval_config, device="cuda"):
    """
    運行 MMLU 評估並返回結果
    """
    try:
        # 檢查是否啟用 MMLU 評估
        if not eval_config.get("enable_mmlu", True):
            print("[INFO] MMLU 評估已停用")
            return 0.0, {}
            
        # 保存模型到臨時路徑（如果還沒保存的話）
        if not os.path.exists(model_path):
            model.save_pretrained(model_path)
        
        # 使用 LM Eval Harness 創建模型包裝器
        eval_model = HFLM(
            pretrained=model_path,
            device=device,
            dtype="auto",
            trust_remote_code=True
        )
        
        # 獲取要評估的任務
        tasks_subset = eval_config.get("mmlu_tasks_subset", "all")
        mmlu_tasks = get_mmlu_tasks_by_category(tasks_subset)
        
        print(f"[INFO] 開始運行 MMLU 評估 ({tasks_subset})，共 {len(mmlu_tasks)} 個任務...")
        
        # 設定評估參數
        num_fewshot = eval_config.get("mmlu_num_fewshot", 5)
        batch_size = eval_config.get("mmlu_batch_size", "auto")
        limit = eval_config.get("mmlu_limit", "")
        
        # 處理 limit 參數
        if limit and str(limit).isdigit():
            limit = int(limit)
        else:
            limit = None
        
        # 執行評估
        results = lm_eval.simple_evaluate(
            model=eval_model,
            tasks=mmlu_tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
        )
        
        # 計算平均 MMLU 分數
        mmlu_scores = []
        for task_name in mmlu_tasks:
            if task_name in results["results"]:
                acc = results["results"][task_name].get("acc,none", 0)
                mmlu_scores.append(acc)
        
        avg_mmlu_score = sum(mmlu_scores) / len(mmlu_scores) if mmlu_scores else 0.0
        
        print(f"[INFO] MMLU 評估完成，平均分數: {avg_mmlu_score:.4f}")
        
        # 清理模型以釋放 GPU 記憶體
        del eval_model
        torch.cuda.empty_cache()
        
        return avg_mmlu_score, results["results"]
        
    except Exception as e:
        print(f"[ERROR] MMLU 評估失敗: {str(e)}")
        return 0.0, {}

# Get function that will be executed by the strategy's evaluate() method
# 修改評估函數以包含 MMLU 評估
def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path, eval_config=None):
    """Return an evaluation function for saving global model and running MMLU evaluation."""

    def evaluate(server_round: int, parameters, config):
        # 保存模型（每輪都保存以便評估）
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # 初始化模型
            model = get_model(model_cfg)
            set_parameters(model, parameters)

            model_save_path = f"{save_path}/peft_{server_round}"
            model.save_pretrained(model_save_path)
            print(f"[INFO] 模型已保存到: {model_save_path}")
            
            # 運行 MMLU 評估（如果啟用）
            mmlu_score = 0.0
            detailed_results = {}
            
            if eval_config and eval_config.get("enable_mmlu", True):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                mmlu_score, detailed_results = run_mmlu_evaluation(
                    model, 
                    model_save_path, 
                    eval_config,
                    device=device
                )
            
            # 清理模型以節省記憶體
            del model
            torch.cuda.empty_cache()
            
            print(f"[INFO] Round {server_round} MMLU Score: {mmlu_score:.4f}")
            
            return mmlu_score, {
                "mmlu_score": mmlu_score,
                "mmlu_detailed": detailed_results,
                "round": server_round
            }
        
        # 如果不是評估輪次，返回預設值
        return 0.0, {}

    return evaluate


def get_on_fit_config(save_path):
    """Return a function that will be used to construct the config that the
    client's fit() method will receive."""

    def fit_config_fn(server_round: int):
        fit_config = {}
        fit_config["current_round"] = server_round
        fit_config["save_path"] = save_path
        return fit_config

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregate (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    seed = cfg.get("seed", 42)
    seed_all(seed)
    # Get initial model weights
    init_model = get_model(cfg.model)
    init_model_parameters = get_parameters(init_model)
    init_model_parameters = ndarrays_to_parameters(init_model_parameters)

    # 準備評估配置
    eval_config = {
        "enable_mmlu": cfg.get("evaluation", {}).get("enable_mmlu", True),
        "mmlu_num_fewshot": cfg.get("evaluation", {}).get("mmlu_num_fewshot", 5),
        "mmlu_batch_size": cfg.get("evaluation", {}).get("mmlu_batch_size", "auto"),
        "mmlu_limit": cfg.get("evaluation", {}).get("mmlu_limit", ""),
        "mmlu_tasks_subset": cfg.get("evaluation", {}).get("mmlu_tasks_subset", "all"),
    }

    # Define strategy
    strategy = FedAvg(
        fraction_fit=cfg.strategy.fraction_fit,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        on_fit_config_fn=get_on_fit_config(save_path),
        fit_metrics_aggregation_fn=fit_weighted_average,
        initial_parameters=init_model_parameters,
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, num_rounds, save_path, eval_config
        ),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)
