#!/usr/bin/env python3
"""
Merlion评估器实现
集成anomaly_predict.py的功能，提供更精确的F1分数计算
"""

import numpy as np
import torch
import time
from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events
from merlion.evaluate.anomaly import accumulate_tsad_score, ScoreType
from merlion.utils import TimeSeries
import pandas as pd


class MerlionEvaluator:
    """
    Merlion评估器 - 提供多种异常检测评估方法
    整合了原anomaly_predict.py的功能，优化了阈值选择和F1分数计算
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self._last_search = None

    def _scan_thresholds(self, scores, labels, nu_range=None, step_size=0.01):
        """扫描阈值范围，复现 CutAddPaste 中 ad_floating 的逻辑"""
        start_time = time.time()

        if nu_range is None:
            nu_range = (-3, 3)

        normalized_scores = self.z_score_normalize(scores)
        if_aff = np.count_nonzero(labels)

        if if_aff:
            events_gt = convert_vector_to_events(labels)
        else:
            events_gt = []

        target_ts = TimeSeries.from_pd(pd.DataFrame(labels))
        nu_list = np.arange(nu_range[0], nu_range[1], step_size)

        affiliation_list = []
        affiliation_f1_list = []
        score_list = []
        predict_list = []
        rpa_f1_list, pa_f1_list, pw_f1_list = [], [], []

        for nu in nu_list:
            predict = np.int64(normalized_scores > nu)
            predict_list.append(predict)

            if if_aff:
                events_pred = convert_vector_to_events(predict)
                Trange = (0, len(predict))
                affiliation_dict = pr_from_events(events_pred, events_gt, Trange)
                precision = affiliation_dict.get("precision", 0.0)
                recall = affiliation_dict.get("recall", 0.0)
                if precision + recall > 0:
                    affiliation_f1 = 2 * precision * recall / (precision + recall)
                else:
                    affiliation_f1 = 0.0
            else:
                affiliation_dict = {"precision": 0.0, "recall": 0.0}
                affiliation_f1 = 0.0

            affiliation_list.append(affiliation_dict)
            affiliation_f1_list.append(affiliation_f1)

            predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
            score = accumulate_tsad_score(ground_truth=target_ts, predict=predict_ts)
            score_list.append(score)
            rpa_f1_list.append(score.f1(ScoreType.RevisedPointAdjusted))
            pa_f1_list.append(score.f1(ScoreType.PointAdjusted))
            pw_f1_list.append(score.f1(ScoreType.Pointwise))

        if len(nu_list) == 0:
            raise ValueError("阈值搜索范围为空")

        affiliation_f1_arr = np.nan_to_num(affiliation_f1_list)
        rpa_f1_arr = np.nan_to_num(rpa_f1_list)
        pa_f1_arr = np.nan_to_num(pa_f1_list)
        pw_f1_arr = np.nan_to_num(pw_f1_list)

        affiliation_idx = int(np.nanargmax(affiliation_f1_arr))
        rpa_idx = int(np.nanargmax(rpa_f1_arr))
        pa_idx = int(np.nanargmax(pa_f1_arr))
        pw_idx = int(np.nanargmax(pw_f1_arr))

        search_duration = time.time() - start_time

        if self.verbose:
            print(f"Best affiliation threshold: {nu_list[affiliation_idx]:.2f}")
            print(f"Best RPA threshold: {nu_list[rpa_idx]:.2f}")
            print(f"Best PA threshold: {nu_list[pa_idx]:.2f}")
            print(f"Execution time: {search_duration:.5f} seconds")

        result = {
            "affiliation_max": affiliation_list[affiliation_idx],
            "rpa_score_max": score_list[rpa_idx],
            "pa_score_max": score_list[pa_idx],
            "pw_score_max": score_list[pw_idx],
            "predict": predict_list[rpa_idx],
            "thresholds": {
                "affiliation": nu_list[affiliation_idx],
                "rpa": nu_list[rpa_idx],
                "pa": nu_list[pa_idx],
                "pw": nu_list[pw_idx]
            },
            "search_duration": search_duration
        }

        self._last_search = result
        return result

    def optimize_threshold_floating(self, scores, labels, nu_range=None, step_size=0.01):
        """
        使用浮动阈值优化来寻找最佳RPA F1和PA F1分数
        """
        search_result = self._scan_thresholds(scores, labels, nu_range, step_size)
        best_rpa_f1 = search_result["rpa_score_max"].f1(ScoreType.RevisedPointAdjusted)
        best_pa_f1 = search_result["pa_score_max"].f1(ScoreType.PointAdjusted)
        best_threshold = search_result["thresholds"]["rpa"]
        return best_rpa_f1, best_pa_f1, best_threshold
    
    def evaluate_with_fixed_threshold(self, scores, labels, threshold=0.5):
        """
        使用固定阈值进行评估
        
        Args:
            scores: 异常分数
            labels: 真实标签
            threshold: 固定阈值
            
        Returns:
            evaluation_results: 评估结果字典
        """
        # 标准化分数
        scores = self.z_score_normalize(scores)
        
        # 预测
        predictions = np.int64(scores > threshold)
        
        # 计算Merlion指标
        target_ts = TimeSeries.from_pd(pd.DataFrame(labels))
        predict_ts = TimeSeries.from_pd(pd.DataFrame(predictions))
        score = accumulate_tsad_score(ground_truth=target_ts, predict=predict_ts)
        
        # 计算affiliation指标
        if_aff = np.count_nonzero(labels)
        if if_aff != 0:
            events_gt = convert_vector_to_events(labels)
            events_pred = convert_vector_to_events(predictions)
            Trange = (0, len(predictions))
            affiliation_dict = pr_from_events(events_pred, events_gt, Trange)
        else:
            affiliation_dict = {"precision": 0.0, "recall": 0.0}
        
        return {
            "rpa_f1": score.f1(ScoreType.RevisedPointAdjusted),
            "pa_f1": score.f1(ScoreType.PointAdjusted),
            "pointwise_f1": score.f1(ScoreType.Pointwise),
            "affiliation_precision": affiliation_dict.get("precision", 0.0),
            "affiliation_recall": affiliation_dict.get("recall", 0.0),
            "threshold": threshold,
            "anomaly_count": int(np.sum(labels)),
            "predicted_anomaly_count": int(np.sum(predictions))
        }
    
    def evaluate_comprehensive(self, scores, labels, optimization_method="floating"):
        """
        综合评估方法，提供完整的性能分析
        
        Args:
            scores: 异常分数
            labels: 真实标签
            optimization_method: 优化方法 ("floating", "fixed", "percentile")
            
        Returns:
            comprehensive_results: 综合评估结果
        """
        if optimization_method == "floating":
            best_rpa_f1, best_pa_f1, best_threshold = self.optimize_threshold_floating(scores, labels)
            search_result = self._last_search or self._scan_thresholds(scores, labels)
            affiliation = search_result["affiliation_max"]
            predict = search_result["predict"]
            detailed_results = {
                "rpa_f1": best_rpa_f1,
                "pa_f1": best_pa_f1,
                "pointwise_f1": search_result["pw_score_max"].f1(ScoreType.Pointwise),
                "affiliation_precision": affiliation.get("precision", 0.0),
                "affiliation_recall": affiliation.get("recall", 0.0),
                "threshold": best_threshold,
                "anomaly_count": int(np.sum(labels)),
                "predicted_anomaly_count": int(np.sum(predict))
            }
        elif optimization_method == "fixed":
            detailed_results = self.evaluate_with_fixed_threshold(scores, labels, threshold=0.5)
            best_rpa_f1 = detailed_results["rpa_f1"]
            best_pa_f1 = detailed_results["pa_f1"]
            best_threshold = detailed_results["threshold"]
        else:
            # Percentile-based threshold
            threshold = np.percentile(scores, 95)  # Top 5% as anomalies
            detailed_results = self.evaluate_with_fixed_threshold(scores, labels, threshold)
            best_rpa_f1 = detailed_results["rpa_f1"]
            best_pa_f1 = detailed_results["pa_f1"]
            best_threshold = threshold
        
        return {
            "best_rpa_f1": best_rpa_f1,
            "best_pa_f1": best_pa_f1,
            "best_threshold": best_threshold,
            "optimization_method": optimization_method,
            **detailed_results
        }
    
    @staticmethod
    def z_score_normalize(scores):
        """Z-score标准化异常分数"""
        mean = np.mean(scores)
        std = np.std(scores)
        if std != 0:
            return (scores - mean) / std
        return scores
    
    def batch_evaluate(self, dataset_results):
        """
        批量评估多个数据集的结果
        
        Args:
            dataset_results: 字典，格式为 {dataset_name: (scores, labels)}
            
        Returns:
            batch_results: 批量评估结果
        """
        results = {}
        total_rpa_f1 = 0.0
        total_pa_f1 = 0.0
        valid_datasets = 0
        
        for dataset_name, (scores, labels) in dataset_results.items():
            try:
                result = self.evaluate_comprehensive(scores, labels)
                results[dataset_name] = result
                
                if result["best_rpa_f1"] > 0:
                    total_rpa_f1 += result["best_rpa_f1"]
                    total_pa_f1 += result["best_pa_f1"]
                    valid_datasets += 1
                    
            except Exception as e:
                if self.verbose:
                    print(f"数据集 {dataset_name} 评估失败: {e}")
                results[dataset_name] = {"error": str(e)}
        
        # 计算平均性能
        avg_rpa_f1 = total_rpa_f1 / valid_datasets if valid_datasets > 0 else 0.0
        avg_pa_f1 = total_pa_f1 / valid_datasets if valid_datasets > 0 else 0.0
        
        if self.verbose:
            print(f"\n批量评估完成:")
            print(f"  有效数据集: {valid_datasets}/{len(dataset_results)}")
            print(f"  平均RPA F1: {avg_rpa_f1:.4f}")
            print(f"  平均PA F1: {avg_pa_f1:.4f}")
        
        return {
            "individual_results": results,
            "summary": {
                "valid_datasets": valid_datasets,
                "total_datasets": len(dataset_results),
                "average_rpa_f1": avg_rpa_f1,
                "average_pa_f1": avg_pa_f1
            }
        }


# 向后兼容的函数接口
def ad_predict(target, scores, mode='floating', nu=0.05):
    """
    向后兼容的异常预测函数
    """
    evaluator = MerlionEvaluator(verbose=False)
    
    if mode == 'floating':
        search_result = evaluator._scan_thresholds(scores, target)
        return (
            search_result["affiliation_max"],
            search_result["rpa_score_max"],
            search_result["pa_score_max"],
            search_result["pw_score_max"],
            search_result["predict"],
        )
    
    else:
        # 其他模式的简化实现
        result = evaluator.evaluate_with_fixed_threshold(scores, target, threshold=0.5)
        
        affiliation_dict = {
            "precision": result["affiliation_precision"], 
            "recall": result["affiliation_recall"]
        }
        
        target_ts = TimeSeries.from_pd(pd.DataFrame(target))
        predict = np.int64(scores > 0.5)
        predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
        score_obj = accumulate_tsad_score(ground_truth=target_ts, predict=predict_ts)
        
        return affiliation_dict, score_obj, score_obj, score_obj, predict
