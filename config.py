esc50_path = 'data/esc50'
# esc50_path = 'D:/sound_datasets/esc50'
runs_path = 'results'
# sub-epoch (batch-level) progress bar display
disable_bat_pbar = False  # True

# do not change this block
n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5]
# ratio to split off from training data
val_size = .2  # could be changed

model_constructor = "ResNet(block=ResidualBlock, layers=[2, 2, 2, 2], num_classes=config.n_classes)"
# model_constructor = "ResNet(block=ResidualBlock, layers=[3, 4, 6, 3], num_classes=config.n_classes)"

# model checkpoints loaded for testing
test_checkpoints = ['terminal.pt']  # ['terminal.pt', 'best_val_loss.pt']
# experiment folder used for testing (result from cross validation training)
# test_experiment = 'results/2024-04-01-00-00'
test_experiment = 'results/2024-06-10-02-03_79_final'

# sampling rate for waves
sr = 44100

device_id = 0
batch_size = 64
num_workers = 6  # 16
persistent_workers = True

epochs = 150
lr = 1e-3
weight_decay = 1e-4  # Adjusted for better regularization
gamma = 0.8
step_size = 10  # Adjusted for less frequent LR reductions
