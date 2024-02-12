import torch
import pandas as pd

def save_to_csv(type, mae_per_node, rmse_per_node, mape_per_node, mae, rmse, mape, config):

        # Convert torch tensors to pandas DataFrame
        errors_df = pd.DataFrame({
            'Node': range(1, (config['N_NODE']+1) ),
            'N_PRED': config["N_PRED"],
            'MAE': mae_per_node.numpy(),
            'RMSE': rmse_per_node.numpy(),
            'MAPE': mape_per_node.numpy()
        })

        # Creating a DataFrame from the overall metrics dictionary
        overall_metrics_df = pd.DataFrame({
            'Node': ["Overall"],
            "N_PRED": [config["N_PRED"]],
            'MAE': [mae.numpy()],
            'RMSE': [rmse.numpy()],
            'MAPE': [mape.numpy()]
        })

        # Append overall metrics to the existing DataFrame
        errors_df = pd.concat([errors_df, overall_metrics_df], ignore_index=True)
        epochs = config["EPOCHS"]

        name_str = str(config["ERRORS_DIR"]) + "/" + type + "_" + str(epochs) + "_METRICS.csv"

        # Save to CSV
        errors_df.to_csv(name_str, mode="a", index=False)

        print(f"Results of the {type} are saved at: {name_str}")

@torch.no_grad()
def eval(model, dataloader, type, config, logging):

    print(f"The eval function is called for {type}") #NOTE: Debug

    device = config["DEVICE"]

    # Set the model to eval
    model.eval()
    model.to(device)

    all_preds, all_truths = [], []

    for batch in dataloader:

        batch = batch.to(device)
        pred = model(batch, device)  # Should be [batch_size, n_nodes, n_preds]

        truth = batch.y.view(-1, config["N_NODE"], config["N_PRED"])  # Assuming each batch.y initially [batch_size * n_nodes, n_preds]

        # NOTE: Debug
        # print(f"Shape of batch: {batch}, Shape of pred: {pred.shape}, shape of truth: {truth.shape}")

        all_preds.append(pred.detach())
        all_truths.append(truth.detach())

    y_pred = torch.cat(all_preds, dim=0)
    y_truth = torch.cat(all_truths, dim=0)

    # Saving results to .CSV File:
    y_pred = y_pred.to("cpu")
    y_truth = y_truth.to("cpu")

    # Checking the shape
    print(f"Concatenated y_pred shape: {y_pred.shape}, Concatenated y_truth shape: {y_truth.shape}")

    mae = MAE(y_truth, y_pred)
    rmse = RMSE(y_truth, y_pred)
    mape = MAPE(y_truth, y_pred)

    if(logging==True):
        
        mae_per_node = MAE_per_node(y_truth, y_pred)
        rmse_per_node = RMSE_per_node(y_truth, y_pred)
        mape_per_node = MAPE_per_node(y_truth, y_pred)
        save_to_csv(type, mae_per_node, rmse_per_node, mape_per_node, mae, rmse, mape, config)

    print(f'Overall results of {type} --> MAE: {mae}, RMSE: {rmse}, MAPE: {mape}')

    return mae, rmse, mape, y_pred, y_truth


def MAPE(v, v_):
    return torch.mean(torch.abs((v_ - v)) / (torch.abs(v)) * 100)


def RMSE(v, v_):
    """
    Mean squared error.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, RMSE averages on all elements of input.
    """
    return torch.sqrt(torch.mean((v_ - v) ** 2))

def MAE(v, v_):
    """
    Mean absolute error.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, MAE averages on all elements of input.
    """
    return torch.mean(torch.abs(v_ - v))


def MAE_per_node(v, v_):
    """
    Mean absolute error.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, MAE averages on all nodes.
    """
    return torch.mean(torch.abs(v_ - v), dim=[0, 2])  # Average across datapoints and predictions

def RMSE_per_node(v, v_):
    """
    Mean squared error.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, RMSE averages on all nodes.
    """
    return torch.sqrt(torch.mean((v_ - v) ** 2, dim=[0, 2]))  # Average across datapoints and predictions

def MAPE_per_node(v, v_):
    """
    Mean absolute percentage error, given as a % (e.g. 99 -> 99%).
    This modification includes a small constant, epsilon, in the denominator to avoid division by zero.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch tensor, MAPE averaged on all nodes.
    """
    return torch.mean(torch.abs((v_ - v) / (torch.abs(v))), dim=[0, 2]) * 100  # Average across datapoints and predictions
