# Nano

Nano: Nested Human-in-the-Loop Reward Learning for Controlling Distribution of Generated Text.

## Instructions

If at stage 0, run `python scripts/gen.py --stage 0 --prompts <prompts>`.

Otherwise, run `python scripts/train_classifier.py && python scripts/train_lm.py <prompts>`. Generate text with `python scripts/gen.py --stage <stage> --dev <device> --start-idx <index> --prompts <prompts>` for each prompt, incrementing `<index>` each time.

`<prompts>` is in the format `[(<number0>, '<prompt0>'), (<number1>, '<prompt1>')]` where `<number>` is the amount to generate for each prompt.

To label the results locally, run `python scripts/label.py --stage <stage>`.
