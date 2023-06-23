cd ../../examples_3D

export model_3d=SE3_Transformer
export dataset=QM9
export task_list=(mu alpha homo lumo gap r2 zpve u0 u298 h298 g298 cv)
export emb_dim_list=(128 300)

export SE3_Transformer_div=2
export SE3_Transformer_n_heads=8
export SE3_Transformer_num_degrees=4

export lr_list=(1e-3)
export lr_scheduler_list=(CosineAnnealingWarmRestarts)
export use_rotation_transform_list=(use_rotation_transform)
export SE3_Transformer_num_layers=7
export SE3_Transformer_num_channels=32
export batch_size=96
export split=customized_01
export seed=42


export epochs=100
export time=71



for task in "${task_list[@]}"; do

for lr in "${lr_list[@]}"; do
for lr_scheduler in "${lr_scheduler_list[@]}"; do
for use_rotation_transform in "${use_rotation_transform_list[@]}"; do
for emb_dim in "${emb_dim_list[@]}"; do

    export output_model_dir=../output/random/"$model_3d"/"$dataset"/"$task"_"$split"_"$seed"/"$lr"_"$lr_scheduler"_"$emb_dim"_"$epochs"_"$SE3_Transformer_num_layers"_"$SE3_Transformer_num_channels"_"$SE3_Transformer_num_degrees"_"$SE3_Transformer_div"_"$SE3_Transformer_n_heads"_"$batch_size"_"$use_rotation_transform"
    export output_file="$output_model_dir"/result.out
    mkdir -p "$output_model_dir"

    python finetune_QM9.py \
    --model_3d="$model_3d" --dataset="$dataset" --epochs="$epochs" --num_workers=8 \
    --task="$task" \
    --SE3_Transformer_num_layers="$SE3_Transformer_num_layers" --SE3_Transformer_num_channels="$SE3_Transformer_num_channels" --batch_size="$batch_size" --"$use_rotation_transform" \
    --SE3_Transformer_num_degrees="$SE3_Transformer_num_degrees" --SE3_Transformer_div="$SE3_Transformer_div" --SE3_Transformer_n_heads="$SE3_Transformer_n_heads" \
    --split="$split" --seed="$seed" \
    --batch_size=96 \
    --lr="$lr" --lr_scheduler="$lr_scheduler" --no_eval_train --print_every_epoch=1 --num_workers=8 \
    --emb_dim="$emb_dim" \
    --output_model_dir="$output_model_dir" \
    > "$output_file"

done
done
done
done
done