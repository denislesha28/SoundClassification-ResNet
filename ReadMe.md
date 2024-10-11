# Sound classification using ResNet

### Configuration

The configuration is located in the config.py file. The following can be configured
- esc50_path : the directory where the dataset is initially downloaded into
- runs_path : the directory where training results get saved into
- model_constructor: define ResNet version / ResidualBlock layers
- test_experiments : the directory of the specific run which is to be used for testing / accuracy evaluation
- batch_size
- num_workers
- persistent_workers (bool)
- epochs
- patience
- lr (learning rate)
- weight_decay
- warm_epochs
- gamma
- step_size

### Dataset Preparation
Inside the `dataset` directory 2 python files handle data preparation for training. 
`dataset_ESC50.py` handles data acquisition (if not already downloaded) and preparation whereas `transforms.py` 
handles data augmentation.

### Model Definition
Inside the `models` directory the resnet model definition can be found in the `model_classifier.py` file.

### Training and Testing
The training process is defined in `train_crossval.py`, where the model is trained for each fold.
`train_crossval.py` generates a timestamped folder inside the results directory where the .pt training files are saved.
This folder can then be referenced in `test_experiments` to be used for testing / cross evaluating.
In similar fashion `test_crossval.py` handles the testing for each model fold. `test_crossval.py` references the directory 
`test_experiments` from `config.py` to locate the trained model .pt files to be used in testing.
`test_crossval.py` finally generates 2 files containing test scores and song classification probabilities.
