import json
import re
import ast
import numpy as np
from datetime import datetime, timedelta
from smolagents import CodeAgent, OpenAIServerModel, tool

class FutureXEvaluationAgent:
    def __init__(self, api_key: str, base_url: str):
        self.model = OpenAIServerModel(model_id="gpt-4.1", api_key=api_key, api_base=base_url)
        self.agent = CodeAgent(tools=[self._search_tool()], model=self.model, add_base_tools=True)
    def _search_tool(self):
        @tool
        def search_historical_data(indicator: str, end_date: str) -> str:
            """
            搜索指标的过去7天历史数据
            Args:
                indicator: 指标名称，如"上海航运交易所煤炭运价指数"
                end_date: 结束日期,格式为YYYY-MM-DD
            Returns:
                搜索到的历史数据信息
            """
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=6)).strftime('%Y-%m-%d')
            return f"搜索 {indicator} 从 {start_date} 到 {end_date} 的每日数据"
        return search_historical_data
    def get_std_for_level4(self, question: str, end_time: str) -> float:
        """计算Level 4的标准差"""
        end_date = end_time.split('T')[0] if 'T' in end_time else end_time
        indicator = re.search(r'([^，]*(?:指数|价格|率|CPI))', question)
        indicator = indicator.group(1) if indicator else question.split('？')[0]
        prompt = f"""
        搜索 {indicator} 在 {end_date} 前7天的历史数据，提取数值并计算标准差。
        步骤：
        1. 搜索历史数据
        2. 提取7个数值
        3. 计算标准差
        最终格式：标准差: [数值]
        """
        try:
            response = self.agent.run(prompt, max_steps=5)
            std_match = re.search(r'标准差[:：]\s*([\d.]+)', response)
            return float(std_match.group(1)) if std_match else 1.0
        except:
            return 1.0
    def evaluate_predictions(self, predictions_file: str, dataset_file: str) -> dict:
        """评估预测结果"""
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = {item['question_id']: item for item in json.load(f)}
        level_scores = {1: [], 2: [], 3: [], 4: []}
        for pred in predictions:
            if pred['error'] or not pred['prediction']:
                continue
            item = dataset.get(pred['question_id'])
            if not item:
                continue
            score = self._evaluate_item(pred, item)
            level_scores[pred['level']].append(score)
        weights = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
        overall = sum(np.mean(scores) * weights[level] for level, scores in level_scores.items() if scores)
        return {
            'overall_score': overall,
            'level_scores': {k: np.mean(v) if v else 0 for k, v in level_scores.items()}
        }
    def _evaluate_item(self, pred: dict, item: dict) -> float:
        prediction = pred['prediction']
        ground_truth = ast.literal_eval(item['answer'])
        level = pred['level']
        if level == 1:  
            pred_option = re.findall(r'\b[A-Z]\b', prediction.upper())
            return 1.0 if pred_option and pred_option[0] == ground_truth[0] else 0.0
        elif level == 2:  
            pred_set = set(re.findall(r'\b[A-Z]\b', prediction.upper()))
            truth_set = set(ground_truth)
            if not pred_set and not truth_set:
                return 1.0
            intersection = len(pred_set & truth_set)
            precision = intersection / len(pred_set) if pred_set else 0
            recall = intersection / len(truth_set) if truth_set else 0
            return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        elif level in [3, 4]:  
            if isinstance(ground_truth[0], (int, float)):
                try:
                    pred_value = float(re.findall(r'[\d.]+', prediction)[0])
                    true_value = ground_truth[0]
                    std_dev = self.get_std_for_level4(item['question'], item['end-time'])
                    if std_dev == 0:
                        return 1.0 if pred_value == true_value else 0.0
                    normalized_error = ((true_value - pred_value) / std_dev) ** 2
                    return max(0, 1 - normalized_error)
                except:
                    return 0.0
            else:

                try:
                    pred_list = ast.literal_eval(prediction) if '[' in prediction else prediction.split(',')
                    pred_list = [item.strip() for item in pred_list]  
                    if pred_list == ground_truth:
                        return 1.0
                    intersection = len(set(pred_list) & set(ground_truth))
                    union = len(set(pred_list) | set(ground_truth))
                    return 0.8 * intersection / union if union > 0 else 0.0
                except:
                    return 0.0
        return 0.0