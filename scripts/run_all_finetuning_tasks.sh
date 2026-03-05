#!/bin/bash

FRAGBERTA_PATH="/data/kaushal/FragBERTa"
VENV_PATH="$FRAGBERTA_PATH/.venv/bin/activate"
SCRIPT_DIR="$FRAGBERTA_PATH/scripts"
LOG_PATH=$FRAGBERTA_PATH/logs
mkdir -p $LOG_PATH

NUM_TRIALS=200
reg_targets=("esol" "freesolv" "lipo" "pdbbind")
slclass_targets=("bace" "bbbp" "hiv")
mlclass_targets=("tox21" "sider")

run_jobs () {
    task_type=$1
    shift
    targets=("$@")

    for target in "${targets[@]}"; do
        for scaffold in 0 1; do

            if [ "$scaffold" -eq 0 ]; then
                suffix="r"
            else
                suffix="s"
            fi

            session_name="${target}_${suffix}"
            log_file="${session_name}.log"

            tmux new -d -s "$session_name" "
                source $VENV_PATH
                cd $SCRIPT_DIR
                python finetuning_with_hpopt.py \
                    --target $target \
                    --task_type $task_type \
                    --use_scaffold $scaffold \
                    --root_dir $FRAGBERTA_PATH \
                    --pretrained_model_path $FRAGBERTA_PATH/models/pretrained \
                    --optuna_num_trials $NUM_TRIALS \
                    &> $LOG_PATH/$log_file
            "

            echo "Launched $session_name"
        done
    done
}

run_jobs "reg" "${reg_targets[@]}"
run_jobs "slclass" "${slclass_targets[@]}"
run_jobs "mlclass" "${mlclass_targets[@]}"