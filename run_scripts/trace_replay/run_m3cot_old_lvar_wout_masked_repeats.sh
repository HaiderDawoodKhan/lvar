# python3 lvar_scripts/infer_lvar_m3cot.py \
#   --config "configs/qwen2vl_m3cot.yaml" \
#   --vlm-path "D:/Haider/IVTLR-Baseline/qwen_vl/output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth" \
#   --controller-path "D:/Haider/lvar/outputs/controller_sft_m3cot/controller_sft.pt" \
#   --output "D:/Haider/lvar/outputs/inference/m3cot/current_lvar_model_validation_wout_masked_repeats/m3cot_lvar_predictions.jsonl" \
#   --use-validation-set

python3 lvar_scripts/infer_lvar_m3cot.py \
  --config "configs/qwen2vl_m3cot.yaml" \
  --vlm-path "D:/Haider/IVTLR-Baseline/qwen_vl/output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth" \
  --controller-path "D:/Haider/lvar/outputs/controller_sft_m3cot/controller_sft.pt" \
  --output "D:/Haider/lvar/outputs/inference/m3cot/current_lvar_model_test_wout_masked_repeats/m3cot_lvar_predictions.jsonl" \