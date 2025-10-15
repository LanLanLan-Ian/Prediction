import json
import uuid
from datetime import datetime
from typing import Any, Dict, List


INPUT_PATH = "/Users/june/Ecode/projects/Prediction/dataset/data_filterd_50.json"
OUTPUT_PATH = "/Users/june/Ecode/projects/Prediction/dataset/25sp500.json"


def parse_float(value: Any) -> float:
    try:
        return float(str(value))
    except Exception:
        return 0.0


def format_date_ymd_gmt8(date_str: str) -> str:
    """Format input date (YYYY-MM-DD) to 'YYYY-MM-DD (GMT+8)'.

    If parsing fails, return the original string with ' (GMT+8)' appended.
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{dt.strftime('%Y-%m-%d')} (GMT+8)"
    except Exception:
        return f"{date_str} (GMT+8)"


def latest_date_key(d: Dict[str, Any]) -> str:
    # Choose the lexicographically max date in YYYY-MM-DD format
    # which is equivalent to the latest chronological date
    return max(d.keys())


def build_prompt(symbol_name: str, end_time_str: str) -> str:
    # Keep prompt style consistent with existing dataset prompts
    return (
        "You are an agent that can predict future events. "
        f"The event to be predicted: \"{end_time_str}, What will be {symbol_name}'s revenue?\"\n"
        "        IMPORTANT: Your final answer MUST end with this exact format:\n"
        "        \\boxed{YOUR_PREDICTION}\n"
        "        Do not use any other format. Do not refuse to make a prediction. "
        "Do not say \"I cannot predict the future.\" You must make a clear prediction "
        "based on the best data currently available, using the box format specified above."
    )


def convert_source_to_target(src: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ticker, info in src.items():
        symbol_name = info.get("symbolName", ticker)

        actual_rev: Dict[str, Any] = info.get("actualRev", {})
        forecast_rev: Dict[str, Any] = info.get("forecastRev", {})

        if not actual_rev:
            # Skip entries without actual revenue
            continue

        # Determine last date and corresponding values
        last_actual_date = latest_date_key(actual_rev)
        ground_truth_str = actual_rev.get(last_actual_date)

        # Additional value is the forecast at the same date, if available
        additional_val_str = None
        if last_actual_date in forecast_rev:
            additional_val_str = forecast_rev.get(last_actual_date)
        elif forecast_rev:
            # Fallback to latest forecast if matching date not found
            additional_val_str = forecast_rev.get(latest_date_key(forecast_rev))

        # Compute Std: abs(groundtruth - forecast_last_value) + groundtruth/100
        gt = parse_float(ground_truth_str)
        fv = parse_float(additional_val_str)
        std_val = abs(gt - fv) + (gt / 100.0)

        # Format to 'YYYY-MM-DD (GMT+8)'
        end_time_formatted = format_date_ymd_gmt8(last_actual_date)

        obj = {
            "id": uuid.uuid4().hex[:24],
            "prompt": build_prompt(symbol_name, end_time_formatted),
            "end_time": end_time_formatted,
            "level": 3,
            "ground_truth": str(ground_truth_str) if ground_truth_str is not None else "",
            "Std": std_val,
            "additional values": str(additional_val_str) if additional_val_str is not None else None,
            "Description": None,
        }
        out.append(obj)

    return out


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        src = json.load(f)

    converted = convert_source_to_target(src)

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} entries -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()