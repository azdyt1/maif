# maif
### Dataset
Original links of datasets are:
- https://www.yelp.com/dataset
- https://grouplens.org/datasets/movielens/1m/

### train movie

- For dataset, you can use `python preprocess/movie_data_gen.py` and `python preprocess/movie_sku_info_gen.py` to generate the movie dataset for the maif model training.
- For training, you can use `python train_movie/train_movie.py {datatype} {train dataset path} {valid dataset path} {test dataset path} ... {mode} {model_name} ...` to train the any model on movie dataset.

### train yelp
- For dataset, you can use `python preprocess/yelp_data_gen.py` and `python preprocess/yelp_sku_info_gen.py` to generate the yelp dataset for the maif model training.
- For training, you can use `python train_movie/train_movie.py {datatype} {train dataset path} {valid dataset path} {test dataset path} ... {mode} {model_name} ...` to train the maif on yelp dataset.

