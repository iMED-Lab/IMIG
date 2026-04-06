import os
import torch


def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    """Save the model checkpoint if accuracy exceeds the target threshold."""
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        torch.save(obj=model, f=os.path.join(
            model_dir, (model_name + '{0:.4f}.pth').format(accu)))
