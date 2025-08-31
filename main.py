import json

import mlflow
import tempfile
import os
import wandb
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

proj_root = os.path.dirname(os.path.abspath(__file__))
local_components = os.path.join(proj_root, "components")

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps
    repo_root = get_original_cwd()

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )


        if "basic_cleaning" in active_steps:
            ##################
            # Implement here #
            ##################

            _ = mlflow.run(
                os.path.join(repo_root, "src", "basic_cleaning"),
                "main",
                env_manager="conda",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_data",
                    "output_description": "Data after basic cleaning",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )
            

        if "data_check" in active_steps:
            ##################
            # Implement here #
            ##################

            _ = mlflow.run(
                os.path.join(repo_root, "src", "data_check"),  
                "main",
                env_manager="conda",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": str(config["data_check"]["kl_threshold"]),
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )


        if "data_split" in active_steps:
            ##################
            # Implement here #
            ##################

            _ = mlflow.run(
                os.path.join(local_components, "train_val_test_split"),
                "main",
                parameters={
                    "input": "clean_sample.csv:latest",                   # <- required name
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################

            step_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "train_random_forest"))

            _ = mlflow.run(
                step_path,
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "output_artifact": "random_forest_export",
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                },
            )

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            _ = mlflow.run(
                os.path.join(os.path.dirname(__file__), "components", "test_regression_model"),
                "main",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest",
                },
            )


if __name__ == "__main__":
    go()
