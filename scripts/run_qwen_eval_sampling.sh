python qwen/qwen_evaluate.py --output-json predictions_sampling/qwen_jailbreakFalse_1.json --do-sample --top-p 0.9
python qwen/qwen_evaluate.py --output-json predictions_sampling/qwen_jailbreakFalse_2.json --do-sample --top-p 0.9
python qwen/qwen_evaluate.py --output-json predictions_sampling/qwen_jailbreakFalse_3.json --do-sample --top-p 0.9
python qwen/qwen_evaluate.py --output-json predictions_sampling/qwen_jailbreakFalse_4.json --do-sample --top-p 0.9
python qwen/qwen_evaluate.py --output-json predictions_sampling/qwen_jailbreakFalse_5.json --do-sample --top-p 0.9

python qwen/qwen_vl_evaluate.py --output-json predictions_sampling/qwen_vl_jailbreakFalse_1.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --output-json predictions_sampling/qwen_vl_jailbreakFalse_2.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --output-json predictions_sampling/qwen_vl_jailbreakFalse_3.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --output-json predictions_sampling/qwen_vl_jailbreakFalse_4.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --output-json predictions_sampling/qwen_vl_jailbreakFalse_5.json --do-sample --top-p 0.9

python qwen/qwen_vl_evaluate.py --use-blank-image --output-json predictions_sampling/qwen_vl_jailbreakFalse_1.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --use-blank-image --output-json predictions_sampling/qwen_vl_jailbreakFalse_2.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --use-blank-image --output-json predictions_sampling/qwen_vl_jailbreakFalse_3.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --use-blank-image --output-json predictions_sampling/qwen_vl_jailbreakFalse_4.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --use-blank-image --output-json predictions_sampling/qwen_vl_jailbreakFalse_5.json --do-sample --top-p 0.9

python qwen/qwen_evaluate.py --use-jailbreak-prompt --output-json predictions_sampling/qwen_jailbreakTrue_1.json --do-sample --top-p 0.9
python qwen/qwen_evaluate.py --use-jailbreak-prompt --output-json predictions_sampling/qwen_jailbreakTrue_2.json --do-sample --top-p 0.9
python qwen/qwen_evaluate.py --use-jailbreak-prompt --output-json predictions_sampling/qwen_jailbreakTrue_3.json --do-sample --top-p 0.9
python qwen/qwen_evaluate.py --use-jailbreak-prompt --output-json predictions_sampling/qwen_jailbreakTrue_4.json --do-sample --top-p 0.9
python qwen/qwen_evaluate.py --use-jailbreak-prompt --output-json predictions_sampling/qwen_jailbreakTrue_5.json --do-sample --top-p 0.9

python qwen/qwen_vl_evaluate.py --use-jailbreak-prompt --output-json predictions_sampling/qwen_vl_jailbreakTrue_1.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --use-jailbreak-prompt --output-json predictions_sampling/qwen_vl_jailbreakTrue_2.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --use-jailbreak-prompt --output-json predictions_sampling/qwen_vl_jailbreakTrue_3.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --use-jailbreak-prompt --output-json predictions_sampling/qwen_vl_jailbreakTrue_4.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --use-jailbreak-prompt --output-json predictions_sampling/qwen_vl_jailbreakTrue_5.json --do-sample --top-p 0.9

python qwen/qwen_vl_evaluate.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/qwen_vl_jailbreakTrue_1.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/qwen_vl_jailbreakTrue_2.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/qwen_vl_jailbreakTrue_3.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/qwen_vl_jailbreakTrue_4.json --do-sample --top-p 0.9
python qwen/qwen_vl_evaluate.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/qwen_vl_jailbreakTrue_5.json --do-sample --top-p 0.9
