# Pretraining
python main_contrast.py \
--num_classes 200 \
--num_cpt 50 \
--base_model resnet18 \
--lr 0.0001 \
--epoch 60 \
--lr_drop 40 \
--pre_train True \
--dataset CUB200 \
--dataset_dir "datasets"

python main_contrast.py \
--num_classes 200 \
--num_cpt 50 \
--base_model resnet18 \
--lr 0.0001 \
--epoch 60 \
--lr_drop 40 \
--dataset CUB200 \
--dataset_dir "datasets" \
--weak_supervision_bias 0.1 \
--quantity_bias 0.1 \
--distinctiveness_bias 0.05 \
--consistence_bias 0.01

python process.py \
--num_classes 200 \
--num_cpt 50 \
--base_model resnet18 \
--dataset CUB200 \
--dataset_dir "datasets"
