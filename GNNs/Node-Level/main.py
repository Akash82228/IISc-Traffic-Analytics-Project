import torch
from torch_geometric.loader import DataLoader
from data_loader.dataloader import SumoTrafficDataset
from train import train_script
from eval import eval
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model_list = ["ST_GAT", "ST_GCN", "ST_GIN", "ST_GraphSAGE", "ST_A3TGCN"]

model_name = "ST_GAT"

if(model_name not in model_list):
    print("Wrong Name of the Model, Check Again !")
    exit()

config = {
    "DEVICE": device,
    'BATCH_SIZE': 128,
    'EPOCHS': 1000,
    'WEIGHT_DECAY': 5e-5,
    'INITIAL_LR': 3e-5,
    'LOG_DIR': './runs',  
    'CHECKPOINT_DIR': './Checkpoints',
    'ERRORS_DIR': "./Errors",
    'N_PRED': 12,      
    'N_HIST': 12,  
    'DROPOUT': 0.2,
    'N_NODE': 20,
    "MODEL": model_name,
    "N_LOGGING_STEPS": 1,
    "DEBUG": True # If True than there will be no Tensorboard logging and weights will not be saved!
}

writer = SummaryWriter(log_dir=config["LOG_DIR"])

torch.cuda.empty_cache()

# Dataloader
dataset = SumoTrafficDataset(root='./sumo_dataset', hist=config["N_HIST"], fut=config["N_PRED"])

#NOTE: Debug for the custom Py-G dataloader
print(f"This is the type of dataset: {type(dataset)}")
print(f"This is the length of the dataset {len(dataset)}")
print(f"This is the first index of the dataset {dataset[0]}")


number_of_training_sample = int(0.80 * len(dataset))  # 70% are training samples
number_of_testing_sample = int(0.10 * len(dataset))  # 15% are testing samples
number_of_validation_sample = len(dataset) - number_of_training_sample - number_of_testing_sample  # remaining are validation samples

# Splitting of the dataset
train = dataset[:number_of_training_sample]
val = dataset[number_of_training_sample:number_of_training_sample + number_of_validation_sample]
test = dataset[number_of_training_sample + number_of_validation_sample:]

print(f'Training data: {len(train)} samples')
print(f'Validation data: {len(val)} samples')
print(f'Testing data: {len(test)} samples')

# Train Dataloader function
train_dataloader = DataLoader(train, batch_size=config["BATCH_SIZE"], shuffle=False)
val_dataloader = DataLoader(val, batch_size=config["BATCH_SIZE"], shuffle=False)
test_dataloader = DataLoader(test, batch_size=config["BATCH_SIZE"], shuffle=False)

# NOTE: DEBUG about the dataloader from pytorch
_, temp_train = next(enumerate(train_dataloader))
_, temp_test = next(enumerate(test_dataloader))
_, temp_val = next(enumerate(val_dataloader))

temp_train.x = temp_train.x.view(config["BATCH_SIZE"], 20, 12)
temp_test.x = temp_test.x.view(config["BATCH_SIZE"], 20, 12)
temp_val.x = temp_val.x.view(config["BATCH_SIZE"], 20, 12)

print(f"Iterating the first batch from train_loader {temp_train.x.shape}")
print(f"Iterating the first batch from val_loader {temp_val.x.shape}")
print(f"Iterating the first batch from test_loader {temp_test.x.shape}")

print(f"Iterating the first batch from train_loader {temp_train}")
print(f"Iterating the first batch from val_loader {temp_val}")
print(f"Iterating the first batch from test_loader {temp_test}")

torch.cuda.empty_cache()

device  = config["DEVICE"]
print(f"Using {device}")
print(f"N_NODE from Config: {dataset.n_node}, N_NODE from dataloader: {dataset.n_node}")

# Just to make sure:
config['N_NODE'] = dataset.n_node

trained_model = train_script(train_dataloader=train_dataloader, val_dataloader=val_dataloader, config=config, writer=writer)

# Checking the results on the training data:
mae_train, rmse_train, mape_train, y_pred_train, y_truth_train = eval(model=trained_model, dataloader=train_dataloader, type="Train", config=config, logging=True)

# Checking the results on the training data:
mae_val, rmse_val, mape_val, y_pred_val, y_truth_val = eval(model=trained_model, dataloader=val_dataloader, type="Validation", config=config, logging=True)

# Check the results on the testing data:
mae_test, rmse_test, mape_test, y_pred_test, y_truth_test = eval(model=trained_model, dataloader=test_dataloader, type="Test", config=config, logging=True)

writer.flush()
torch.cuda.empty_cache()
