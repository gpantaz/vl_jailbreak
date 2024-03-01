python vicua/vicuna --output-json predictions_sampling/vicuna_jailbreakFalse_1.json --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python vicua/vicuna --output-json predictions_sampling/vicuna_jailbreakFalse_2.json --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python vicua/vicuna --output-json predictions_sampling/vicuna_jailbreakFalse_3.json --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python vicua/vicuna --output-json predictions_sampling/vicuna_jailbreakFalse_4.json --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python vicua/vicuna --output-json predictions_sampling/vicuna_jailbreakFalse_5.json --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv

python llava/llava_eval.py --output-json predictions_sampling/llava_jailbreakFalse_1.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --output-json predictions_sampling/llava_jailbreakFalse_2.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --output-json predictions_sampling/llava_jailbreakFalse_3.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --output-json predictions_sampling/llava_jailbreakFalse_4.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --output-json predictions_sampling/llava_jailbreakFalse_5.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv

python llava/llava_eval.py --use-blank-image --output-json predictions_sampling/blank_image_llava_jailbreakFalse_1.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --use-blank-image --output-json predictions_sampling/blank_image_llava_jailbreakFalse_2.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --use-blank-image --output-json predictions_sampling/blank_image_llava_jailbreakFalse_3.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --use-blank-image --output-json predictions_sampling/blank_image_llava_jailbreakFalse_4.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --use-blank-image --output-json predictions_sampling/blank_image_llava_jailbreakFalse_5.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv

python vicua/vicuna --use-jailbreak-prompt --output-json predictions_sampling/vicuna_jailbreakTrue_1.json --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python vicua/vicuna --use-jailbreak-prompt --output-json predictions_sampling/vicuna_jailbreakTrue_2.json --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python vicua/vicuna --use-jailbreak-prompt --output-json predictions_sampling/vicuna_jailbreakTrue_3.json --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python vicua/vicuna --use-jailbreak-prompt --output-json predictions_sampling/vicuna_jailbreakTrue_4.json --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python vicua/vicuna --use-jailbreak-prompt --output-json predictions_sampling/vicuna_jailbreakTrue_5.json --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv

python llava/llava_eval.py --use-jailbreak-prompt --output-json predictions_sampling/llava_jailbreakTrue_1.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --use-jailbreak-prompt --output-json predictions_sampling/llava_jailbreakTrue_2.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --use-jailbreak-prompt --output-json predictions_sampling/llava_jailbreakTrue_3.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --use-jailbreak-prompt --output-json predictions_sampling/llava_jailbreakTrue_4.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --use-jailbreak-prompt --output-json predictions_sampling/llava_jailbreakTrue_5.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv

python llava/llava_eval.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/blank_image_llava_jailbreakTrue_1.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/blank_image_llava_jailbreakTrue_2.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/blank_image_llava_jailbreakTrue_3.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/blank_image_llava_jailbreakTrue_4.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
python llava/llava_eval.py --use-jailbreak-prompt --use-blank-image --output-json predictions_sampling/blank_image_llava_jailbreakTrue_5.json --do-sample --top-p 0.9 --jailbreak-csv data/jailbreak-prompts2.csv
