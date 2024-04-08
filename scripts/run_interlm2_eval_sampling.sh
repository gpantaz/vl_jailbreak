python internlm2/internlm2_eval.py --output-json predictions_sampling/internlm2_jailbreakFalse_1.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm2_eval.py --output-json predictions_sampling/internlm2_jailbreakFalse_2.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm2_eval.py --output-json predictions_sampling/internlm2_jailbreakFalse_3.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm2_eval.py --output-json predictions_sampling/internlm2_jailbreakFalse_4.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm2_eval.py --output-json predictions_sampling/internlm2_jailbreakFalse_5.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv

python internlm2/internlm_xcomposer.py --output-json predictions_sampling/internlm_xcomposer_jailbreakFalse_1.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --output-json predictions_sampling/internlm_xcomposer_jailbreakFalse_2.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --output-json predictions_sampling/internlm_xcomposer_jailbreakFalse_3.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --output-json predictions_sampling/internlm_xcomposer_jailbreakFalse_4.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --output-json predictions_sampling/internlm_xcomposer_jailbreakFalse_5.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv

python internlm2/internlm_xcomposer.py --use-blank-image --output-json predictions_sampling/blank_image_internlm_xcomposer_jailbreakFalse_1.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --use-blank-image --output-json predictions_sampling/blank_image_internlm_xcomposer_jailbreakFalse_2.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --use-blank-image --output-json predictions_sampling/blank_image_internlm_xcomposer_jailbreakFalse_3.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --use-blank-image --output-json predictions_sampling/blank_image_internlm_xcomposer_jailbreakFalse_4.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --use-blank-image --output-json predictions_sampling/blank_image_internlm_xcomposer_jailbreakFalse_5.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv

python internlm2/internlm2_eval.py --use-jailbreak-prompt --output-json predictions_sampling/internlm2_jailbreakTrue_1.json ---do-sample -top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm2_eval.py --use-jailbreak-prompt --output-json predictions_sampling/internlm2_jailbreakTrue_2.json ---do-sample -top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm2_eval.py --use-jailbreak-prompt --output-json predictions_sampling/internlm2_jailbreakTrue_3.json ---do-sample -top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm2_eval.py --use-jailbreak-prompt --output-json predictions_sampling/internlm2_jailbreakTrue_4.json ---do-sample -top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm2_eval.py --use-jailbreak-prompt --output-json predictions_sampling/internlm2_jailbreakTrue_5.json ---do-sample -top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv

python internlm2/internlm_xcomposer.py --use-jailbreak-prompt --output-json predictions_sampling/internlm_xcomposer_jailbreakTrue_1.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --use-jailbreak-prompt --output-json predictions_sampling/internlm_xcomposer_jailbreakTrue_2.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --use-jailbreak-prompt --output-json predictions_sampling/internlm_xcomposer_jailbreakTrue_3.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --use-jailbreak-prompt --output-json predictions_sampling/internlm_xcomposer_jailbreakTrue_4.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --use-jailbreak-prompt --output-json predictions_sampling/internlm_xcomposer_jailbreakTrue_5.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv

python internlm2/internlm_xcomposer.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/blank_image_internlm_xcomposer_jailbreakTrue_1.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/blank_image_internlm_xcomposer_jailbreakTrue_2.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/blank_image_internlm_xcomposer_jailbreakTrue_3.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/blank_image_internlm_xcomposer_jailbreakTrue_4.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python internlm2/internlm_xcomposer.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/blank_image_internlm_xcomposer_jailbreakTrue_5.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
