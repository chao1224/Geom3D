cd ../../examples_3D

export dataset=QM9
export task_list=(mu alpha homo lumo gap r2 zpve u0 u298 h298 g298 cv)

export split=customized_01
export seed=42
export batch_size=128

export model=N_Gram_Graph
export N_Gram_Graph_predictor_type=XGB
export emb_dim_list=(100 300)
export bond_list=(use_bond no_bond)
export N_Gram_Graph_normalize_list=(N_Gram_Graph_normalize no_N_Gram_Graph_normalize)

export XGB_n_estimators_list=(10 100)
export time=2



for task in "${task_list[@]}"; do

for emb_dim in "${emb_dim_list[@]}"; do
for XGB_n_estimators in "${XGB_n_estimators_list[@]}"; do
for bond in "${bond_list[@]}"; do
for N_Gram_Graph_normalize in "${N_Gram_Graph_normalize_list[@]}"; do

    export output_model_dir=../output/random/"$model"_"$N_Gram_Graph_predictor_type"/"$dataset"/"$task"_"$split"_"$seed"/"$emb_dim"_"$XGB_n_estimators"_"$bond"_"$N_Gram_Graph_normalize"
    export output_file="$output_model_dir"/result.out
    mkdir -p "$output_model_dir"

    python finetune_qm9_NGram.py \
    --model="$model" --N_Gram_Graph_predictor_type="$N_Gram_Graph_predictor_type" \
    --emb_dim="$emb_dim" \
    --XGB_n_estimators="$XGB_n_estimators" \
    --dataset="$dataset" \
    --task="$task" \
    --split="$split" --seed="$seed" \
    --batch_size="$batch_size" \
    --"$bond" \
    --"$N_Gram_Graph_normalize" \
    --output_model_dir="$output_model_dir" \
    > "$output_file"
    

done
done
done
done

done
