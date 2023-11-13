import os
import yaml
import torch
from audioldm_eval import EvaluationHelper

device = torch.device(f"cuda:{0}")
evaluator = EvaluationHelper(None, device)


def locate_yaml_file(path):
    for file in os.listdir(path):
        if ".yaml" in file:
            return os.path.join(path, file)
    return None


def is_evaluated(path):
    candidates = []
    for file in os.listdir(
        os.path.dirname(path)
    ):  # all the file inside a experiment folder
        if ".json" in file:
            candidates.append(file)
    folder_name = os.path.basename(path)
    for candidate in candidates:
        if folder_name in candidate:
            return True
    return False


def locate_validation_output(path):
    folders = []
    for file in os.listdir(path):
        dirname = os.path.join(path, file)
        if "val_" in file and os.path.isdir(dirname):
            if not is_evaluated(dirname):
                folders.append(dirname)
    return folders


def evaluate_exp_performance(exp_name):
    abs_path_exp = os.path.join(latent_diffusion_model_log_path, exp_name)
    config_yaml_path = locate_yaml_file(abs_path_exp)

    if config_yaml_path is None:
        print("%s does not contain a yaml configuration file" % exp_name)
        return

    folders_todo = locate_validation_output(abs_path_exp)

    for folder in folders_todo:
        print(folder)

        if len(os.listdir(folder)) == 964:
            test_dataset = "audiocaps_16k"
        elif len(os.listdir(folder)) > 5000:
            test_dataset = "musiccaps"
        else:
            continue

        test_audio_data_folder = os.path.join(test_audio_path, test_dataset)

        evaluator.main(folder, test_audio_data_folder)


def eval(exps):
    for exp in exps:
        try:
            evaluate_exp_performance(exp)
        except Exception as e:
            print(exp, e)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="AudioLDM model evaluation")

    parser.add_argument(
        "-l", "--log_path", type=str, help="the log path", required=True
    )
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        help="the experiment name",
        required=False,
        default=None,
    )

    args = parser.parse_args()

    test_audio_path = "log/testset_data"
    latent_diffusion_model_log_path = args.log_path

    if latent_diffusion_model_log_path != "all":
        exp_name = args.exp_name
        if exp_name is None:
            exps = os.listdir(latent_diffusion_model_log_path)
            eval(exps)
        else:
            eval([exp_name])
    else:
        todo_list = [os.path.abspath("log/latent_diffusion")]
        for todo in todo_list:
            for latent_diffusion_model_log_path in os.listdir(todo):
                latent_diffusion_model_log_path = os.path.join(
                    todo, latent_diffusion_model_log_path
                )
                if not os.path.isdir(latent_diffusion_model_log_path):
                    continue
                print(latent_diffusion_model_log_path)
                exps = os.listdir(latent_diffusion_model_log_path)
                eval(exps)
