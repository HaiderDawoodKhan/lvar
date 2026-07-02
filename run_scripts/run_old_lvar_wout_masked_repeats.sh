# python3 lvar_scripts/infer_lvar_m3cot.py \
#   --config "configs/qwen2vl_m3cot.yaml" \
#   --vlm-path "/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth" \
#   --controller-path "/home/csalt/Haider/DVLM/lvar/outputs/controller_sft_m3cot/controller_sft.pt" \
#   --output "/home/csalt/Haider/DVLM/lvar/outputs/inference/current_lvar_model_validation_wout_masked_repeats/m3cot_lvar_predictions.jsonl" \
#   --use-validation-set

python3 lvar_scripts/infer_lvar_m3cot.py \
  --config "configs/qwen2vl_m3cot.yaml" \
  --vlm-path "/home/csalt/Haider/DVLM/IVT-LR/qwen_vl/output/qwen_IVTLR_m3cot/epoch_16_full_model_fp32.pth" \
  --controller-path "/home/csalt/Haider/DVLM/lvar/outputs/controller_sft_m3cot/controller_sft.pt" \
  --output "/home/csalt/Haider/DVLM/lvar/outputs/inference/current_lvar_model_test_wout_masked_repeats/m3cot_lvar_predictions.jsonl" \