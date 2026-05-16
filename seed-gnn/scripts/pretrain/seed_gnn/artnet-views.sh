method="seed_gnn"
dataset="artnet-views"
output_dir_root=$1
dataset_dir=$2
shift 2

models=("gcn" "sage" "gin" "gat")

forwarded_args=()
while [[ $# -gt 0 ]]; do
        case "$1" in
                --models|--architectures)
                        shift
                        models=()
                        while [[ $# -gt 0 && "$1" != --* ]]; do
                                if [[ "$1" == *,* ]]; then
                                        IFS=',' read -r -a parsed_models <<< "$1"
                                        for parsed_model in "${parsed_models[@]}"; do
                                                if [[ -n "$parsed_model" ]]; then
                                                        models+=("$parsed_model")
                                                fi
                                        done
                                else
                                        models+=("$1")
                                fi
                                shift
                        done
                        ;;
                *)
                        forwarded_args+=("$1")
                        shift
                        ;;
        esac
done

for model in "${models[@]}"; do
        python main.py \
                --exp_desc="pretrain_${model}_${method}" \
                --pipeline_config_dir="config/pipeline_config/${method}/${model}/${dataset}.json" \
                --eval_config_dir="config/eval_config/edit_gnn/${dataset}.json" \
                --task="pretrain" \
                --output_folder_dir="${output_dir_root}/results/${method}/${model}/${dataset}/" \
                --pretrain_output_dir="${output_dir_root}/edit_ckpts" \
                --dataset_dir="${dataset_dir}" \
                "${forwarded_args[@]}"
done
