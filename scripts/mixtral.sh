
#!/bin/sh
MODEL="mixtral-7B-chat"
CONFIG="./config.json"
EXP=1000
SHOTS=0
QUERYPOS="end"
METHOD="cot-wm-chat"
for DATASET in "adv-csfb" "mindgames" "tomi" "socialiqa" "fantom"
do
    for SC in 1 2 3 4 5
    do
        python3 -m main --wandb --dataset $DATASET --config $CONFIG --model $MODEL --query-method $METHOD --query-position $QUERYPOS --kshots $SHOTS --splitted-context $SC --num-experiments $EXP
    done
done

MODEL="mixtral-7B"
for DATASET in "adv-csfb" "mindgames" "tomi" "socialiqa" "fantom"
do
    for METHOD in "cot" "struct" "struct-yaml" "tot"
    do
        python3 -m \
            main \
            --wandb \
            --dataset $DATASET \
            --config $CONFIG \
            --model $MODEL \
            --query-method $METHOD \
            --query-position $QUERYPOS \
            --kshots $SHOTS \
            --splitted-context $SC \
            --num-experiments $EXP \
            --n_generate_sample 7 \
            --n_evaluate_sample 3 \
            --n_select_sample 3
    done
done
