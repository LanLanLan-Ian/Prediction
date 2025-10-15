import re
import json
import numpy as np
import argparse
import logging
# from smolagents import OpenAIServerModel
from openai import OpenAI
import os
import ipdb

class FutureXEvaluator:
    def __init__(self, api_key: str = None, base_url: str = None, file_name: str = None):
        """
        Args:
            api_key: OpenAI API key (可选，如果不提供则不使用LLM纠正)
            base_url: API base URL
        """
        self.use_llm = api_key is not None
        self.file_name = file_name
        
        # 设置日志
        self._setup_logging()
        
        if self.use_llm:
            self.client = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=api_key,
            )
    
    def _setup_logging(self):
        """设置日志配置"""
        # 确保evaluation目录存在
        eval_dir = "evaluation"
        os.makedirs(eval_dir, exist_ok=True)
        
        # 设置日志文件路径
        if self.file_name:
            log_filename = self.file_name.replace('.json', '.log')
        else:
            log_filename = 'evaluation.log'
        
        log_path = os.path.join(eval_dir, log_filename)
        
        # 配置logger
        self.logger = logging.getLogger(f'FutureXEvaluator_{id(self)}')
        self.logger.setLevel(logging.INFO)
        
        # 清除已有的handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 创建文件handler
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建formatter
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_and_print(self, message):
        """同时记录到日志文件和控制台"""
        self.logger.info(message)
    
    def _normalize_numerical_answer(self, pred: str, truth: str) -> tuple:
        """
        使用LLM将预测值和真实值标准化到同一单位（以ground truth为准）
        Returns:
            (pred_value: float, truth_value: float)
        """
        if not self.use_llm:
            return (self._extract_number(pred), self._extract_number(truth))
        
        prompt = f""" 
Ground Truth: {truth}
Prediction: {pred}

Task: Convert the prediction to match the SAME format and unit as the ground truth.

Examples:
- Ground Truth: "10362 millions yuan", Prediction: "10362000000" 
  → Ground Truth Value: 10362, Prediction Value: 10362
  
- Ground Truth: "4522.61", Prediction: "4.52261 thousand"
  → Ground Truth Value: 4522.61, Prediction Value: 4522.61

- Ground Truth: "12740 million", Prediction: "12.74 billion"
  → Ground Truth Value: 12740, Prediction Value: 12740

- Ground Truth: "15.5 billion", Prediction: "15500 million"
  → Ground Truth Value: 15.5, Prediction Value: 15.5

IMPORTANT: 
1. Keep the SAME unit as ground truth (if GT is "millions yuan", convert prediction to millions yuan) ！
2. Extract only the numeric value, ignore unit text
3. If there is no numeric value in Prediction, set PREDICTION_VALUE to 0 directly
4. Return ONLY two numbers in this exact format:
   GROUND_TRUTH_VALUE: [number]
   PREDICTION_VALUE: [number]
5. No explanation, just the two numbers.

Now normalize:
"""
        
        try:
            completion = self.client.chat.completions.create(
                # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
                temperature=0.2,
                model="deepseek-v3-1-terminus",
                messages=[
                    {"role": "system", "content": "You are a precise number normalization assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            response = completion.choices[0].message.content
            truth_match = re.search(r'GROUND_TRUTH_VALUE:\s*([\d.]+)', response)
            pred_match = re.search(r'PREDICTION_VALUE:\s*([\d.]+)', response)
            
            if truth_match and pred_match:
                truth_value = float(truth_match.group(1))
                pred_value = float(pred_match.group(1))
                return (pred_value, truth_value)
            else:
                # LLM提取失败，回退到简单提取
                self.log_and_print(f"Warning: LLM normalization failed, falling back. Response: {response}")
                return (self._extract_number(pred), self._extract_number(truth))
        except Exception as e:
            self.log_and_print(f"Error in LLM normalization: {e}")
            return (self._extract_number(pred), self._extract_number(truth))
    
    def _eval_numerical(self, pred: str, truth: str, std: float) -> float:
        """
        Level 3/4 数值型:
        score(Y, Ŷ) = max(0, 1 - ((Y - Ŷ) / σ(Y))²)
        """
        try:
            pred_value, true_value = self._normalize_numerical_answer(pred, truth)
            # print(f"  Pred Value: {pred_value}, True Value: {true_value}")
            if std == 0:
                return 1.0 if abs(pred_value - true_value) < 1e-6 else 0.0
            
            normalized_error = ((true_value - pred_value) / std) ** 2
            score = max(0.0, 1.0 - normalized_error)
            
            self.log_and_print(f"    LLM-Truth: {true_value}\n    LLM-Pred: {pred_value}\n    Std: {std}, Score: {score:.4f}")
            # with open(f"{self.file_name}_L4.txt", "a", encoding="utf-8") as f:
            #     f.write(f"Truth: {true_value} ｜ Pred: {pred_value}\n")
            
            return score
        except Exception as e:
            self.log_and_print(f"Error in numerical evaluation: {e}")
            return 0.0
    
    def _eval_single_choice(self, pred: str, truth: str) -> float:
        """Level 1: score(Y, Ŷ) = 𝕀(Y = Ŷ)"""
        pred_options = self._extract_options(pred)
        pred_option = next(iter(pred_options)) if pred_options else ""
        truth_option = truth.strip()
        score = 1.0 if pred_option.upper() == truth_option.strip().upper() else 0.0
        self.log_and_print(f"  Truth: {truth_option}, Pred: {pred_option}, Score: {score:.4f}")
        return score
    
    def _eval_multi_choice(self, pred: str, truth: str) -> float:
        """Level 2: score(Y, Ŷ) = F1-Score(Y, Ŷ)"""
        pred_set = self._extract_options(pred)
        truth_set = self._extract_options(truth)
        
        if not pred_set and not truth_set:
            return 1.0
        
        intersection = len(pred_set & truth_set)
        precision = intersection / len(pred_set) if pred_set else 0
        recall = intersection / len(truth_set) if truth_set else 0
        
        if precision + recall == 0:
            score = 0.0
        else:
            score = 2 * precision * recall / (precision + recall)
        
        self.log_and_print(f"  Truth: {truth_set}, Pred: {pred_set}, F1: {score:.4f}")
        return score
    
    def _eval_ranking(self, pred: str, truth: str) -> float:
        """
        Level 3/4 排序题:
        score = 1 if 完全匹配（顺序也对）
        score = 0.8 × |交集| / k otherwise
        """
        pred_list = self._extract_list(pred)
        truth_list = self._extract_list(truth)
    
        if pred_list == truth_list:
            score = 1.0
        else:
            # 部分匹配：0.8 × (交集大小 / k)
            intersection = len(set(pred_list) & set(truth_list))
            k = len(truth_list)
            score = 0.8 * intersection / k if k > 0 else 0.0
        
        self.log_and_print(f"    Truth: {truth_list}")
        self.log_and_print(f"    Pred: {pred_list}")
        self.log_and_print(f"    Score: {score:.4f}")
        return score
    
    
    def _extract_options(self, text: str) -> set:
        """
        提取选项（支持单选与多选）：
        - 优先提取单个字母选项 A/B/C...
        - 若未匹配到字母选项且内容为单词（如 Yes/No），返回该词（大写）作为集合
        """
        if not text:
            return set()

        boxed = re.search(r'\\boxed\{([^}]+)\}', text)
        content = boxed.group(1) if boxed else text

        # 优先提取字母选项（A/B/C...）
        letters = re.findall(r'\b([A-Z])\b', content.upper())
        if letters:
            return set(letters)

        # 兜底：如果是单词（如 YES/NO），返回该词
        token = content.strip()
        if re.fullmatch(r'[A-Za-z]+', token):
            return {token.upper()}

        return set()
    
    def _extract_number(self, text: str) -> float:
        """提取数值（fallback方法，不考虑单位）"""
        numbers = re.findall(r'[\d.]+', text)
        if numbers:
            return float(numbers[0])
        raise ValueError("No number found")
    
    def _extract_list(self, text: str) -> list:
        """提取排序列表"""
      
        boxed = re.search(r'\\boxed\{([^}]+)\}', text)
        if not boxed:
            content = text
        else:
            content = boxed.group(1)
        
        content = re.sub(r'\d+(?:st|nd|rd|th)?[:：\.\s]+', '', content)
        
        items = re.split(r'[,，、;；\n]', content)
        items = [item.strip() for item in items if item.strip()]
        return items
    
    def evaluate_prediction(self, pred_answer: str, item: dict) -> float:
        """
        Args:
            pred_answer: 模型预测的答案（包含\\boxed{}格式）
            item: ground_truth数据项
        Returns:
            score: 0-1之间的分数
        """
        ground_truth = item['ground_truth']
        level = item['level']
        
        self.log_and_print(f"\n[Level {level}] ID: {item['id']}")
        self.log_and_print(f"  Ground Truth: {ground_truth}")
        self.log_and_print(f"  Prediction: {pred_answer}")
        
        if level == 1:  # Level 1: 单选题
            return self._eval_single_choice(pred_answer, ground_truth)
        
        elif level == 2:  # Level 2: 多选题
            return self._eval_multi_choice(pred_answer, ground_truth)
        
        elif level in [3, 4]:
            if item.get('Std') is not None:  # 数值型预测
                return self._eval_numerical(pred_answer, ground_truth, item['Std'])
            else:  
                return self._eval_ranking(pred_answer, ground_truth)
        
        return 0.0
    
    def evaluate_all(self, data: list) -> dict:
        """
        评估所有预测
        Args:
            data: JSON数据列表，每个item包含 ground_truth 和 answer
                [
                    {
                        "id": "xxx",
                        "ground_truth": "...",
                        "answer": "\\boxed{...}",
                        "level": 1,
                        "Std": ...
                    },
                    ...
                ]
        Returns:
            {
                "overall_score": float,
                "level_scores": {1: float, 2: float, 3: float, 4: float},
                "level_counts": {1: int, 2: int, 3: int, 4: int}
            }
        """
        level_scores = {1: [], 2: [], 3: [], 4: []}
        
        for item in data:
            if 'answer' not in item:
                self.log_and_print(f"Warning: No answer found for ID {item['id']}")
                continue
            score = self.evaluate_prediction(item['answer'], item)
            level_scores[item['level']].append(score)
        
        weights = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
        overall = sum(
            np.mean(scores) * weights[level] 
            for level, scores in level_scores.items() 
            if scores
        )
        
        return {
            'overall_score': overall,
            'level_scores': {k: np.mean(v) if v else 0.0 for k, v in level_scores.items()},
            'level_counts': {k: len(v) for k, v in level_scores.items()}
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FutureX 评估工具')
    parser.add_argument('-f', '--file_name', default='20250915-random.json')
    
    args = parser.parse_args()
    
    evaluator = FutureXEvaluator(
        api_key="df7a9bcb-f50e-486d-9018-28eaf88aedfd",
        base_url="https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        file_name=args.file_name
    )
    
    try:
        if not args.file_name.endswith('.json'):
            args.file_name += '.json'
        with open(os.path.join('prediction',args.file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        evaluator.log_and_print(f"正在评估文件: {args.file_name}")
        results = evaluator.evaluate_all(data)
        
        evaluator.log_and_print("\n" + "="*50)
        evaluator.log_and_print("评估结果:")
        evaluator.log_and_print(f"总分: {results['overall_score']:.4f}")
        evaluator.log_and_print("\n各级别得分:")
        for level in [1, 2, 3, 4]:
            evaluator.log_and_print(f"  Level {level}: {results['level_scores'][level]:.4f} (共 {results['level_counts'][level]} 题)")
    
    except FileNotFoundError:
        evaluator.log_and_print(f"错误: 找不到文件 '{args.file_name}'")
    except json.JSONDecodeError:
        evaluator.log_and_print(f"错误: 文件 '{args.file_name}' 不是有效的JSON格式")
    except Exception as e:
        evaluator.log_and_print(f"错误: {e}")