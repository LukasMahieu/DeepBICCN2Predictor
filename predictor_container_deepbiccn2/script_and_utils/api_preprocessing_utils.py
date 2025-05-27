import crested

import numpy as np
import random
import base64
import tqdm


## model specific checks that cause a "prediction_request_failed" error
def check_seqs_specifications(sequences, json_return_error_model):
    required_length = 2114
    for sequence in sequences:
        value = sequences[sequence]
        key = sequence

        if len(value) != required_length:
            json_return_error_model["prediction_request_failed"].append(
                f"length of a sequence in {key} is not equal to {required_length}"
            )
    return json_return_error_model


def fake_model_point(sequences, json_dict):
    predictions = {}
    # Use tqdm to show progress as we process each sequence.
    for sequence in tqdm.tqdm(
        sequences, desc="Processing sequences (point prediction)", unit="seq"
    ):
        predictions[sequence] = random.randint(0, 1)
    json_dict["predictions"] = predictions
    return json_dict


