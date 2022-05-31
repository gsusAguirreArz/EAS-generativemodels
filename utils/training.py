""" Training Loops Information Printer

This script allows the user to print to the console usefull information about their corresponding training loop.

This script requires....

This file can also be imported as a module and contains the following functions:

    * print_batches_progress_bar - creates and prints a dynamic loading bar of the batches completed for each epoch.
    * print_epoch_info - prints current epoch number and some other epoch related information
"""

from sys import stdout
import torch
from torch.nn import Module


def print_batches_progress_bar( step: int, num_batches: int ) -> None:
    """ Prints a loading bar for each epoch of the training.

    Parameters
    ----------
    step : int
        Current step of the epoch training loop
    num_batches : int
        Total number of batches
    
    Returns
    -------
    None
    """
    write = stdout.write
    flush = stdout.flush

    write( '\r' )
    write( "[%-70s] %d%%, batch %d \t" % (
        '=' * int( 70 * (step + 1) / num_batches) + '>',
        int( 100 * (step + 1) / num_batches),
        int(step+1)
    ) )
    flush()
    return


def print_epoch_info( epoch: int, num_epochs: int, n_print_frecuency: int, info_dict: dict ) -> None:
    """ Prints the information related to the current epoch and feeded on info dict

    Parameters
    ----------
    epoch : int
        Integer representing the current epoch
    num_epochs : int
        Integer representing the total number of epochs
    n_print_frecuency : int
        Integer that controls the frecuency of the information printing
    info_dict : dict<str,int>
        Dictionary containing data
    
    Returns
    -------
    None
    """

    text = f"| Epoch: [{epoch+1:03}/{num_epochs:03}] | "
    for key in info_dict:
        if key == "time":
            text += f"{key}: {info_dict[key]:.2f}s | "
        else:
            text += f"{key}: {info_dict[key]:.4f} | "
    
    line_break = '-' * (len(text) + 10)

    if epoch % n_print_frecuency == 0 or epoch == num_epochs-1:
        print( f"\n{line_break}\n{text}\n{line_break}\n" )
    return


def print_model_summary( model : Module ) -> None:
    """ Prints the model summary - similar to summary methos in tensorflow

    Parameters
    ----------
    model : torch.nn.Module
        The model we want to see the summary
    
    Returns
    -------
    None
    """
    print(f"Block name: {model.layer_name}")
    print("-"*40)
    print(" | LayerName | \t | Size | \t | Nparams | ")
    print("="*70)
    for param_tensor in model.state_dict():
        print(f"{param_tensor}: \t {tuple(model.state_dict()[param_tensor].size())}, \t {model.state_dict()[param_tensor].numel()}")
    # for name, param in model.named_parameters(): # more info of each sublayer: for idx, m in enumerate(model.named_modules()): print(f"{idx} -> {m}")
    #     print(f"{name}: \t {param.numel()}")
    print("="*70)

    params = sum(params.numel() for params in model.parameters())
    params_grad = sum( params.numel() for params in model.parameters() if params.requires_grad )
    print( f"Total parameters: {params}")
    print( f"Trainable parameters: {params_grad}")
    print( f"Non-trianable parameters: {params - params_grad}\n")
    return


def gradient_penalty(critic_model: Module, real: torch.Tensor, fake: torch.Tensor, device: str="cpu", labels: torch.Tensor=None ):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand( (BATCH_SIZE, 1, 1, 1) ).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic_model(interpolated_images, labels) if labels != None else critic_model(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean( (gradient_norm - 1)**2 )
    return gradient_penalty


def save_checkpoint( epoch: int, models: list, optimizers: list, losses: list, filename: str="model_checkpoint.pth.tar" ) -> None:
    # print("====> Saving checkpoint")

    checkpoint ={'epoch': epoch}
    for i, model in enumerate(models):
        checkpoint[f"{model.layer_name}_{i}_state_dict"] = model.state_dict()
    for i, optimizer in enumerate(optimizers):
        checkpoint[f"optimizer_{i}_state_dict"] = optimizer.state_dict()
    for i, loss in enumerate(losses):
        checkpoint[f"loss_{i}"] = loss

    torch.save(checkpoint, filename)
    # print("Success")
    return


def get_checkpoint(PATH: str, show_info: bool=False):
    checkpoint = torch.load(PATH)
    if show_info:
        for key in checkpoint:
            print(f"Key: '{key}'")
    return checkpoint


def load_state_dicts(checkpoint: dict, models: list) -> None:
    # print("====> Loading states dicts")
    for i, model in enumerate(models):
        model.load_state_dict(checkpoint[f"{model.layer_name}_{i}_state_dict"])
    # for i, optimizer in enumerate(optimizers):
    #     optimizer.load_state_dict(checkpoint[f"optimizer_{i}_state_dict"])
    print("Success")
    return


# if __name__ == "__main__":
#     test_info_dict = {"time":2, "d_loss": 23, "g_loss": 34}
#     print_epoch_info(1,5,1,test_info_dict)