# maif
### Dataset
Original links of datasets are:
- https://tianchi.aliyun.com/dataset/649
- https://grouplens.org/datasets/movielens/1m/
### Matching model

- For dataset, you can use `python matching_model/preprocess/data.py {dataset_name}` to generate a specific dataset for the matching model ComiRec training.
- For training, you can use `python matching_model/train.py --dataset {dataset_name} --model_type {model__name}` to train the ComiRec model.

### Ranking model
- For dataset, you can use `python ranking_model/preprocess/{dataset_name}_data_gen.py` to generate a specific dataset for the Ranking Model training.
- For training, you can use `python ranking_model/train.py {dataset_name} {model_name} ...parameters` to train any ranking model you want.

### Re-ranking model
- For re-ranking model, you can use `python ranking_model/test_rerank.py` to run the re-ranking model for testing.
