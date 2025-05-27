import os
import crested
import pandas as pd
import keras

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_PATH, "..", "model")
MODEL_NAME = "deepbiccn2"
saved_models_path = os.path.join(MODEL_PATH, f"{MODEL_NAME}.keras")
targets_file = os.path.join(MODEL_PATH, f"{MODEL_NAME}_output_classes.tsv")


def get_cell_type_index():
    """
    Returns a dictionary mapping cell type names to their corresponding indices.
    """
    targets_df = pd.read_csv(targets_file, sep="\t", names=["target"])
    return {target: i for i, target in enumerate(targets_df["target"])}


def predict_crested(sequences: dict) -> dict | str:
    predictions = {}
    try:
        # extract sequences from dict
        seqs = list(sequences.values())
        seqs_ids = list(sequences.keys())
        model = keras.models.load_model(saved_models_path, compile=False)
        crested_predictions = crested.tl.predict(
            input=seqs,
            model=model,
            genome=None,
        )  # (N, C)
        for i, seq_id in enumerate(seqs_ids):
            predictions[seq_id] = crested_predictions[i]
    except Exception as e:
        predictions = str(e)
    return predictions
