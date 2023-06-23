cd ../../examples_3D

export model_3d=PaiNN
export dataset=QM9
export task_list=(mu alpha homo lumo gap r2 zpve u0 u298 h298 g298 cv)

export lr_scheduler_list=(CosineAnnealingLR)
export split=customized_01
export seed=42
export emb_dim_list=(128 300)
export batch_size_list=(128)

export epochs=1000
export time=19

export lr_list=(3e-4 5e-4)



for task in "${task_list[@]}"; do

for lr in "${lr_list[@]}"; do
for lr_scheduler in "${lr_scheduler_list[@]}"; do

for emb_dim in "${emb_dim_list[@]}"; do
for batch_size in "${batch_size_list[@]}"; do

    export output_model_dir=../output/random/"$model_3d"/"$dataset"/"$task"_"$split"_"$seed"/"$lr"_"$lr_scheduler"_"$emb_dim"_"$batch_size"_"$epochs"
    export output_file="$output_model_dir"/result.out
    mkdir -p "$output_model_dir"

    python finetune_QM9.py \
    --model_3d="$model_3d" --dataset="$dataset" --epochs="$epochs" \
    --task="$task" \
    --split="$split" --seed="$seed" \
    --batch_size="$batch_size" \
    --emb_dim="$emb_dim" \
    --lr="$lr" --lr_scheduler="$lr_scheduler" --no_eval_train --print_every_epoch=1 --num_workers=8 \
    --output_model_dir="$output_model_dir" \
    > "$output_file"
    
done
done
done

done
done