# LLM API Cost Tracking and Reporting

This document explains the cost tracking system for LLM API calls in the SGO training pipeline.

## Overview

The cost tracking system automatically tracks all LLM API calls made during SGO training and calculates costs based on model pricing. It provides detailed breakdowns by:
- Model (o3, gpt-5.2, gpt-4o, etc.)
- Call type (Part A batch, Part B batch, review generation, theme classification, etc.)
- Token usage (input, output, cached)
- Individual call details

## Model Pricing

Current pricing (per million tokens):

| Model | Input | Cached Input | Output |
|-------|-------|--------------|--------|
| o3 | $2.00 | $0.50 | $8.00 |
| gpt-5.2 | $1.75 | $0.175 | $14.00 |
| gpt-4o | $2.50 | $1.25 | $10.00 |
| gpt-4o-mini | $0.15 | $0.075 | $0.60 |
| gpt-5-mini | $0.15 | $0.075 | $0.60 |

## Automatic Cost Tracking

Cost tracking is automatically enabled during SGO training. All API calls are tracked in:
- `llm_client.py` - Part A (review generation/correction) and Part B (feedback loop)
- `llm_client_logprobs.py` - Review generation and theme classification with logprobs

## Cost Report Generation

### Automatic Report (During Pipeline)

The cost report is automatically generated at the end of the SGO training pipeline and saved to:
```
{OUTPUT_BASE_DIR}/cost_report.json
```

The report includes:
- Summary by model
- Summary by call type
- Detailed breakdown by model and call type
- Individual call records with timestamps and metadata

### Manual Report Generation

You can also generate a cost report manually:

```bash
# From the 07_sgo_training directory
cd scripts
python generate_cost_report.py --print-detailed

# Or save to a specific file
python generate_cost_report.py --output /path/to/cost_report.json --print-detailed
```

### Report Structure

The cost report JSON contains:

```json
{
  "report_metadata": {
    "total_calls": 1234,
    "grand_total_cost": 45.67,
    "report_timestamp": 1234567890.0,
    "pricing_used": {...}
  },
  "summary_by_model": {
    "gpt-5.2": {
      "total_calls": 100,
      "total_input_tokens": 50000,
      "total_output_tokens": 10000,
      "total_cost": 12.34,
      "avg_cost_per_call": 0.1234,
      "call_types": {...}
    }
  },
  "summary_by_call_type": {
    "part_b_batch": {
      "total_calls": 50,
      "total_cost": 5.67,
      "models_used": ["gpt-5.2"]
    }
  },
  "cost_breakdown": {
    "by_model_and_call_type": {...}
  },
  "all_calls": [...]
}
```

## Call Types Tracked

- `part_a_batch` - Part A batch processing (review generation/correction in confidence mode)
- `part_b_batch` - Part B batch processing (feedback loop analysis)
- `review_generation_logprobs` - Review generation in logprobs mode
- `theme_logprobs_with_context` - Theme classification with full persona context
- `theme_logprobs_without_context` - Theme classification without persona context
- `theme_logprobs_with_context_fallback` - Fallback theme classification (when logprobs not available)
- `theme_logprobs_without_context_fallback` - Fallback theme classification (when logprobs not available)

## Usage Examples

### View Cost Summary During Execution

The cost summary is automatically printed at the end of the pipeline. You can also call it programmatically:

```python
from sgo_training.utils.cost_tracker import print_cost_summary
print_cost_summary()
```

### Get Cost Data Programmatically

```python
from sgo_training.utils.cost_tracker import get_cost_summary, get_detailed_report

# Get summary
summary = get_cost_summary()
print(f"Total cost: ${summary['grand_total_cost']:.4f}")

# Get detailed report
report = get_detailed_report()
print(f"Total calls: {report['total_calls']}")
```

### Reset Cost Tracker

To reset cost tracking (e.g., for a new run):

```python
from sgo_training.utils.cost_tracker import reset_cost_tracker
reset_cost_tracker()
```

## Cost Calculation Details

Costs are calculated as:
```
Total Cost = (Input Tokens / 1,000,000) × Input Price
           + (Cached Input Tokens / 1,000,000) × Cached Input Price
           + (Output Tokens / 1,000,000) × Output Price
```

All token counts are extracted from the OpenAI API response's `usage` field.

## Notes

- Cost tracking is thread-safe and can be used in parallel processing
- All costs are in USD
- Token counts are tracked per API call
- Cached tokens are tracked separately when available
- The system automatically normalizes model names (e.g., "gpt-5.2" → "gpt-5.2")
















