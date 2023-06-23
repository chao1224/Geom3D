cd ../../examples_3D

export dataset=QM9
export task_list=(mu alpha homo lumo gap r2 zpve u0 u298 h298 g298 cv)

export split=customized_01
export seed=42
export batch_size=128

export model=AWARE
export emb_dim_list=(100 300)
export r_prime_list=(300 500)
export lr_list=(5e-4 1e-4)
export decay_list=(0)
export max_walk_len_list=(3 6 9)
export bond_list=(use_bond)

export max_walk_len_list=(3 6)
export epochs_list=(1000)
export time=23


for task in "${task_list[@]}"; do
for emb_dim in "${emb_dim_list[@]}"; do
for r_prime in "${r_prime_list[@]}"; do
for lr in "${lr_list[@]}"; do
for decay in "${decay_list[@]}"; do
for max_walk_len in "${max_walk_len_list[@]}"; do
for bond in "${bond_list[@]}"; do
for epochs in "${epochs_list[@]}"; do

    export output_model_dir=../output/random/"$model"/"$dataset"/"$task"_"$split"_"$seed"/"$emb_dim"_"$r_prime"_"$lr"_"$decay"_"$max_walk_len"_"$bond"_"$epochs"
    export output_file="$output_model_dir"/result.out
    echo "$output_model_dir"
    mkdir -p "$output_model_dir"

    python finetune_QM9_NGram.py \
    --model="$model" \
    --emb_dim="$emb_dim" \
    --r_prime="$r_prime" \
    --lr="$lr" --decay="$decay" \
    --max_walk_len="$max_walk_len" \
    --epochs="$epochs" \
    --dataset="$dataset" \
    --task="$task" \
    --split="$split" --seed="$seed" \
    --batch_size="$batch_size" \
    --no_verbose \
    --"$bond" \
    --output_model_dir="$output_model_dir" \
    > "$output_file"
    
done
done
done
done
done
done
done
done
