from models.st_gat import ST_GAT, ST_GraphSAGE, ST_GCN, ST_GIN, ST_A3TGCN
import tqdm
import torch
import torch.optim as optim
import os
import time
from eval import eval

def train_logger(writer, description, x, y):

    writer.add_scalar(description, x, y)

def train_script(train_dataloader, val_dataloader, config, writer):

    device = config["DEVICE"]

    if config["MODEL"] == "ST_GAT":
        model = ST_GAT(
            in_channels=config['N_HIST'],
            out_channels=config['N_PRED'],
            n_nodes=config['N_NODE'],
            dropout=config['DROPOUT'])
    
    elif config["MODEL"] == "ST_GCN":
        model = ST_GCN(
            in_channels=config['N_HIST'],
            out_channels=config['N_PRED'],
            n_nodes=config['N_NODE'],
            dropout=config['DROPOUT'])
    
    elif config["MODEL"] == "ST_GIN":
        model = ST_GIN(
            in_channels=config['N_HIST'],
            out_channels=config['N_PRED'],
            n_nodes=config['N_NODE'],
            dropout=config['DROPOUT'])
        
    elif config["MODEL"] == "ST_GraphSAGE":
        model = ST_GraphSAGE(
            in_channels=config['N_HIST'],
            out_channels=config['N_PRED'],
            n_nodes=config['N_NODE'],
            dropout=config['DROPOUT'])
        
    elif config["MODEL"] == "ST_A3TGCN":
        model = ST_A3TGCN(
            in_channels=config['N_HIST'],
            out_channels=config['N_PRED'],
            n_nodes=config['N_NODE'],
            dropout=config['DROPOUT'])
        
    else: 
        print("Model Name Not Found !")
        exit()

    print(model)

    # Send model to device
    model.to(device=device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['INITIAL_LR'], weight_decay=config['WEIGHT_DECAY'])
    loss_fn = torch.nn.MSELoss()  # Correct initialization of the loss function

    for epoch in range(config['EPOCHS']):
        
        # Set model to train mode
        model.train()

        total_loss = 0
        for _, batch in enumerate(tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}")):

            batch = batch.to(device)
            optimizer.zero_grad()

            y_pred = model(batch, device)

            batch.y = batch.y.view(-1, config["N_NODE"], config["N_PRED"])

            # Shape Check
            if y_pred.shape != batch.y.shape:
                print(f"Shape mismatch: y_pred {y_pred.shape}, batch.y {batch.y.shape}")
                exit()

            loss = loss_fn(y_pred, batch.y.float())
            total_loss += loss.item()

            train_logger(writer=writer, description = "Loss/train", x=loss.item(), y=epoch)

            loss.backward()
            optimizer.step()

        if(epoch % config["N_LOGGING_STEPS"] == 0):
            # After the enumerate loop calculate the overall loss for 1 epoch and print it if necassary:
            avg_loss = total_loss / len(train_dataloader)
            print(f"Average Loss: {avg_loss:.3f} for Epoch: {epoch}, now logging the values for train and val on tensorboard ...")
            
            if(config["DEBUG"] == False):
            
                # Log the values for the training dataset:
                mae_train, rmse_train, mape_train, _, _ = eval(model=model, dataloader=train_dataloader, type="Train", config=config, logging=False)
                train_logger(writer, description="Train/MAE", x=mae_train, y=epoch)
                train_logger(writer, description="Train/RMSE", x=rmse_train, y=epoch)
                train_logger(writer, description="Train/MAPE", x=mape_train, y=epoch)

                # Log the values for validation dataset:
                mae_val, rmse_val, mape_val, _, _ = eval(model=model, dataloader=val_dataloader, type="Validation", config=config, logging=False)
                train_logger(writer, description="Validation/MAE", x=mae_val, y=epoch)
                train_logger(writer, description="Validation/RMSE", x=rmse_val, y=epoch)
                train_logger(writer, description="Validation/MAPE", x=mape_val, y=epoch)

    if(config["DEBUG"] == False):
        # Save the model at the end of training
        timestr = time.strftime("%Y%m%d-%H%M%S")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, os.path.join(config["CHECKPOINT_DIR"], f"model_{timestr}_{epoch}.pt"))

        checkpoint_dir = config["CHECKPOINT_DIR"]
        print(f"Model Weights saved as: {timestr} at {checkpoint_dir}")

    return model