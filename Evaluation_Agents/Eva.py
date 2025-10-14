import re
import json
import numpy as np
from smolagents import OpenAIServerModel

class FutureXEvaluator:
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Args:
            api_key: OpenAI API key (可选，如果不提供则不使用LLM纠正)
            base_url: API base URL
        """
        self.use_llm = api_key is not None
        if self.use_llm:
            self.model = OpenAIServerModel(
                model_id="gpt-4.1", 
                api_key=api_key, 
                api_base=base_url
            )
    
    def _normalize_numerical_answer(self, pred: str, truth: str) -> tuple:
        """
        使用LLM将预测值和真实值标准化到同一单位（以ground truth为准）
        Returns:
            (pred_value: float, truth_value: float)
        """
        if not self.use_llm:
            return (self._extract_number(pred), self._extract_number(truth))
        
        prompt = f"""
You are a precise number normalization assistant. 

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
1. Keep the SAME unit as ground truth (if GT is "millions yuan", convert prediction to millions yuan)
2. Extract only the numeric value, ignore unit text
3. Return ONLY two numbers in this exact format:
   GROUND_TRUTH_VALUE: [number]
   PREDICTION_VALUE: [number]
4. No explanation, just the two numbers.

Now normalize:
"""
        
        try:
            response = self.model(prompt)
            
    
            truth_match = re.search(r'GROUND_TRUTH_VALUE:\s*([\d.]+)', response)
            pred_match = re.search(r'PREDICTION_VALUE:\s*([\d.]+)', response)
            
            if truth_match and pred_match:
                truth_value = float(truth_match.group(1))
                pred_value = float(pred_match.group(1))
                return (pred_value, truth_value)
            else:
                # LLM提取失败，回退到简单提取
                print(f"Warning: LLM normalization failed, falling back. Response: {response}")
                return (self._extract_number(pred), self._extract_number(truth))
        except Exception as e:
            print(f"Error in LLM normalization: {e}")
            return (self._extract_number(pred), self._extract_number(truth))
    
    def _eval_numerical(self, pred: str, truth: str, std: float) -> float:
        """
        Level 3/4 数值型:
        score(Y, Ŷ) = max(0, 1 - ((Y - Ŷ) / σ(Y))²)
        """
        try:
   
            pred_value, true_value = self._normalize_numerical_answer(pred, truth)
            
            if std == 0:
                return 1.0 if abs(pred_value - true_value) < 1e-6 else 0.0
            
            normalized_error = ((true_value - pred_value) / std) ** 2
            score = max(0.0, 1.0 - normalized_error)
            
            print(f"  Truth: {true_value}, Pred: {pred_value}, Std: {std}, Score: {score:.4f}")
            return score
        except Exception as e:
            print(f"Error in numerical evaluation: {e}")
            return 0.0
    
    def _eval_single_choice(self, pred: str, truth: str) -> float:
        """Level 1: score(Y, Ŷ) = 𝕀(Y = Ŷ)"""
        pred_option = self._extract_option(pred)
        truth_option = truth.strip()
        score = 1.0 if pred_option == truth_option else 0.0
        print(f"  Truth: {truth_option}, Pred: {pred_option}, Score: {score:.4f}")
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
        
        print(f"  Truth: {truth_set}, Pred: {pred_set}, F1: {score:.4f}")
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
        
        print(f"  Truth: {truth_list}")
        print(f"  Pred: {pred_list}")
        print(f"  Score: {score:.4f}")
        return score
    
    
    def _extract_option(self, text: str) -> str:
        """提取单个选项字母"""

        boxed = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            content = boxed.group(1)
            options = re.findall(r'\b([A-Z])\b', content.upper())
            return options[0] if options else ""
        return ""
    
    def _extract_options(self, text: str) -> set:
        """提取多个选项字母"""
   
        boxed = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            content = boxed.group(1)
        else:
            content = text
        

        options = re.findall(r'\b([A-Z])\b', content.upper())
        return set(options)
    
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
        
        print(f"\n[Level {level}] ID: {item['id']}")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Prediction: {pred_answer}")
        
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
                print(f"Warning: No answer found for ID {item['id']}")
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
    evaluator = FutureXEvaluator(
        api_key="your-api-key",
        base_url="your-base-url"
    )
    
    with open('data_with_predictions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = evaluator.evaluate_all(data)
    
    print("\n" + "="*50)
    print("评估结果:")
    print(f"总分: {results['overall_score']:.4f}")
    print("\n各级别得分:")
    for level in [1, 2, 3, 4]:
        print(f"  Level {level}: {results['level_scores'][level]:.4f} (共 {results['level_counts'][level]} 题)")