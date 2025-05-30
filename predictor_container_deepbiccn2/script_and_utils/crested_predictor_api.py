# crested_predictor_api.py
import os
import sys
import json
import tqdm
import struct
import socket
import numpy as np

from error_message_functions_updated import *
from api_preprocessing_utils import *
from crested_utils import predict_crested, get_cell_type_index

# Get the absolute path of the script's directory
MODEL_NAME = "DeepBICCN2"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Determine if running inside a container or not
if os.path.exists("/.singularity.d"):
    # Running inside the container
    print("Running inside the container...🥡")
    HELP_FILE = (
        "/predictor_container_deepbiccn2/script_and_utils/predictor_help_message.json"
    )
else:
    # Running outside the container
    print("Running outside the container...📋")
    PREDICTOR_CONTAINER_DIR = os.path.dirname(SCRIPT_DIR)
    HELP_FILE = os.path.join(SCRIPT_DIR, "predictor_help_message.json")

# Set buffer size for TCP
BUFFER_SIZE = 65536


def recv_message_loop(client_socket):
    # Step 1: Receive total bytes (length) of the Evaluator's request
    # Step 2: Receive file from Evaluator

    # ---------------------- Receive Evaluator JSON ----------------------
    while True:
        # Initialize data to store a new message on each iteration
        json_data_recv = b""
        # Before receiving JSON from Evaluator
        # Receive length of the incoming JSON message (4-byte integer)
        # Can change to 8-byte integer by changing .recv(4) to .recv(8)
        # and replacing format string '>I' to '>Q'
        # Step 1
        try:
            msg_length = client_socket.recv(4)
            if not msg_length:
                print("Failed to receive message length. Closing connection.")
                client_socket.close()
                break  # Exit the loop if no message length is received

            # Unpack message length from 4 bytes
            msglen = struct.unpack(">I", msg_length)[0]
            print(f"Expecting {msglen} bytes of data from the Evaluator.")

            # Initialize the progress bar
            progress = tqdm.tqdm(
                range(msglen),
                unit="B",
                desc="Receiving Evaluator Request(s)",
                unit_scale=True,
                unit_divisor=1024,
            )

            # Step 2
            # Now we want to receive the actual JSON in packets
            while len(json_data_recv) < msglen:
                packet = client_socket.recv(BUFFER_SIZE)
                if not packet:
                    print("Connection closed unexpectedly.")
                    break
                json_data_recv += packet
                progress.update(len(packet))
                # print(f"Received packet of {len(packet)} bytes, total received: {len(data)} bytes")

            # Close the progress bar when done
            progress.close()

            # Decode and display the received data if all of it is received
            if len(json_data_recv) == msglen:
                print("Evaluator request received completely")
                pass
            else:
                print("Data received was incomplete or corrupted.")
                break
        except Exception as e:
            print(f"Error while receiving data: {e}")
            client_socket.close()
            break  # Break the loop on exception

        # ---------------------- Process Received JSON ----------------------
        evaluator_request_full = json_data_recv
        evaluator_json = evaluator_request_full.decode("utf-8")
        evaluator_json = json.loads(evaluator_json)

        # group these functions
        json_return_error = {"bad_prediction_request": []}

        # if only a "help" was requested return the predictor information file
        if evaluator_json["request"] == "help":
            # model builder should place help file in predictor folder
            help_file = HELP_FILE
            print(f"Help requested! Sending {HELP_FILE}...")
            jsonResult_help = json.load(open(help_file))

            jsonResult_help = json.dumps(jsonResult_help)
            try:
                jsonResult_help_bytes = jsonResult_help.encode("utf-8")
                jsonResult_help_total_bytes = len(jsonResult_help_bytes)
                client_socket.sendall(struct.pack(">I", jsonResult_help_total_bytes))
                client_socket.sendall(jsonResult_help_bytes)
                continue
            except socket.error as e:
                print("server_error: Error sending help response: %s" % e)
                # finally:
                client_socket.close()
                # server.close()
                print("Connection to client closed")
                # sys.exit(0)

        # --- MODEL-SPECIFIC: Determine readout type ---
        readout_type = evaluator_json.get("readout", "point")
        is_point_readout = readout_type == "point"

        # re-usable error checking functions
        json_return_error = check_mandatory_keys(
            evaluator_json.keys(), json_return_error
        )
        json_return_error = check_request(evaluator_json["request"], json_return_error)
        json_return_error = check_prediction_task_mandatory_keys(
            evaluator_json["prediction_tasks"], json_return_error
        )
        # if any of the mandatory keys are missing immediately return an error to the evaluator
        if any(json_return_error.values()) == True:
            json_string = json.dumps(json_return_error)
            try:
                jsonResult_error_bytes = json_string.encode("utf-8")
                jsonResult_error_total_bytes = len(jsonResult_error_bytes)
                client_socket.sendall(struct.pack(">I", jsonResult_error_total_bytes))
                client_socket.sendall(jsonResult_error_bytes)
                continue
            except socket.error as e:
                print("server_error: Error sending error response: %s" % e)
                # finally:
                client_socket.close()
                # server.close()
                print("Connection to client closed")
                # sys.exit(1)
                break
        else:
            json_return_error = check_key_values_readout(
                evaluator_json["readout"], json_return_error
            )
            json_return_error = check_prediction_task_name(
                evaluator_json["prediction_tasks"], json_return_error
            )
            json_return_error = check_prediction_task_type(
                evaluator_json["prediction_tasks"], json_return_error
            )
            json_return_error = check_prediction_task_cell_type(
                evaluator_json["prediction_tasks"], json_return_error
            )
            json_return_error = check_prediction_task_species(
                evaluator_json["prediction_tasks"], json_return_error
            )
            if "prediction_ranges" in evaluator_json.keys():
                json_return_error = check_seq_ids(
                    evaluator_json["prediction_ranges"],
                    evaluator_json["sequences"],
                    json_return_error,
                )
                json_return_error = check_prediction_ranges(
                    evaluator_json["prediction_ranges"], json_return_error
                )

            if (
                "upstream_seq" in evaluator_json.keys()
                or "downstream_seq" in evaluator_json.keys()
            ):
                json_return_error = check_key_values_upstream_flank(
                    evaluator_json["upstream_seq"], json_return_error
                )
            if "downstream_seq" in evaluator_json.keys():
                json_return_error = check_key_values_downstream_flank(
                    evaluator_json["downstream_seq"], json_return_error
                )

            # --- MODEL SPECIFIC: Ensure this CREsted Predictor only supports mus_musculus ---
            for task in evaluator_json["prediction_tasks"]:
                if task.get("species", "").lower() != "mus_musculus":
                    json_return_error["bad_prediction_request"].append(
                        f"This predictor only supports species: mus_musculus. Received '{task.get('species')}' for task '{task.get('name')}'."
                    )
                    break

            # if any errors were caught return them all to evaluator
            if any(json_return_error.values()) == True:
                json_string = json.dumps(json_return_error)
                try:
                    jsonResult_error_bytes = json_string.encode("utf-8")
                    jsonResult_error_total_bytes = len(jsonResult_error_bytes)
                    client_socket.sendall(
                        struct.pack(">I", jsonResult_error_total_bytes)
                    )
                    client_socket.sendall(jsonResult_error_bytes)
                    continue
                except socket.error as e:
                    print("server_error: Error sending error response: %s" % e)
                    # finally:
                    client_socket.close()
                    # server.close()
                    print("Connection to client closed")
                    # sys.exit(1)
                    break

        # ---------------------- Process Sequences and Prediction Ranges ----------------------
        # Extract sequences to predict
        # Check that the sequences meet model specifications
        # Otherwise do any other formatting required for the model
        sequences = evaluator_json["sequences"]

        # --- Add upstream and downstream flanking sequences, if provided by the evaluator ---
        # Default to empty string if not provided
        upstream_seq = evaluator_json.get("upstream_seq", "")
        downstream_seq = evaluator_json.get("downstream_seq", "")
        if upstream_seq or downstream_seq:
            print(
                f"Applying flanking:\
                    \n+{len(upstream_seq)} bases upstream,\
                    \n+{len(downstream_seq)} bases downstream"
            )
            for seq_id, sequence in tqdm.tqdm(
                sequences.items(),
                desc="Flanking sequences",
                unit="sequence",
                total=len(sequences),
                dynamic_ncols=True,
            ):
                flanked = f"{upstream_seq}{sequence}{downstream_seq}"
                sequences[seq_id] = flanked

        # Can add any additional error checking functions here
        json_return_error_model = {"prediction_request_failed": []}
        json_return_error_model = check_seqs_specifications(
            sequences, json_return_error_model
        )

        # --- Process prediction_ranges if provided ---
        if "prediction_ranges" in evaluator_json:
            prediction_ranges = evaluator_json["prediction_ranges"]
            for seq_id, pr in prediction_ranges.items():
                # Only process non-empty ranges
                if pr:
                    # Unpack start and end indices
                    start, end = pr
                    # Check that the end index does not exceed sequence length
                    if end >= len(sequences[seq_id]):
                        json_return_error_model["prediction_request_failed"].append(
                            f"Prediction range for '{seq_id}' exceeds the sequence length!"
                        )
                    else:
                        # Slice the sequence. `prediction_range` is start, end inclusive
                        sequences[seq_id] = sequences[seq_id][start : end + 1]
                        print(
                            f"Sequence '{seq_id}' trimmed to prediction range [{start}, {end}]."
                        )

        # if anything is caught don't run the model and return to evaluator to fix
        if any(json_return_error_model.values()) == True:
            json_string = json.dumps(json_return_error_model)
            try:
                jsonResult_bytes = json_string.encode("utf-8")
                jsonResults_total_bytes = len(jsonResult_bytes)
                client_socket.sendall(struct.pack(">I", jsonResults_total_bytes))
                client_socket.sendall(jsonResult_bytes)
                continue
            except socket.error as e:
                print("server_error: Error sending error response: %s" % e)
                # finally:
                client_socket.close()
                # server.close()
                print("Connection to client closed")
                # sys.exit(1)
                break

            # ---------------------- Extract Prediction Tasks and Run the Model ----------------------
            # Start big loop here for all the prediction_tasks
            # Connect to cell type matching container in cases of multi-task models
            # cell_type_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # cell_type_socket.connect((cell_type_matcher_ip, cell_type_matcher_port))

        task_predictions = predict_crested(
            sequences
        )  # return predictions over all cell types {seq_id: [[preds]]}
        cell_type_mapping = get_cell_type_index()

        # --- ADDITION: Early bail-out if model returns error or cell type is not found---
        # Send the error to client and close this client
        def _send_error_and_continue(client_socket, errors):
            """
            Helper to package up a list of error messages and send them back to the Evaluator.
            Returns True if it sent (so caller should `continue`), False on socket error.
            """
            payload = {"prediction_request_failed": errors}
            js = json.dumps(payload).encode("utf-8")
            try:
                client_socket.sendall(struct.pack(">I", len(js)))
                client_socket.sendall(js)
                print(
                    "Sent prediction error back; closing connection with this Evaluator"
                )
                return True
            except socket.error as e:
                print(f"server_error: Error sending error response: {e}")
                client_socket.close()
                print("Connection to client closed")
                return False

        cell_type_errors = [
            f"Cell type '{t['cell_type']}' not recognized."
            for t in evaluator_json["prediction_tasks"]
            if t["cell_type"] not in cell_type_mapping
        ]
        if isinstance(task_predictions, str):
            cell_type_errors.append(task_predictions)
        if cell_type_errors:
            if not _send_error_and_continue(client_socket, cell_type_errors):
                break
            continue

        # Now format predictions to API JSON structure
        # Create JSON to return
        json_return = {
            "request": evaluator_json["request"],
            # Prediction task is an array of objects for all requested tasks
            "prediction_tasks": [],
        }

        # Loop through all the prediction tasks
        for prediction_task in evaluator_json["prediction_tasks"]:
            request_type = prediction_task["type"]
            cell_type = prediction_task["cell_type"]
            idx = cell_type_mapping.get(cell_type, None)
            if idx is None:
                # If the cell type is not found in the mapping, return an error
                json_return_error["bad_prediction_request"].append(
                    f"Cell type '{cell_type}' not recognized."
                )
                continue
            formatted_preds = {}
            for seq_id, preds in task_predictions.items():
                raw = preds[idx]
                if isinstance(raw, (np.generic, float, int)):
                    formatted_preds[seq_id] = float(raw)
                else:
                    formatted_preds[seq_id] = raw.tolist()

            # Cell type predictor container is running, send the predictor's cell type and evaluator cell type to it
            # If you want to override the cell type container you can remove the following code
            # Send the predictor and evaluator cell type
            # cell_type_socket.sendall(b'Hello, cell type matcher dude!')
            # cell_type_matcher_return = cell_type_socket.recv(1024)

            # The following code will be model specific
            # Sample point prediction model
            # Model builders need to add the appropriate returns here

            # Create structured response for the evaluator
            current_prediction_task = {
                "name": prediction_task["name"],
                "type_requested": request_type,
                "type_actual": request_type,  # If remapped, update this
                "cell_type_requested": cell_type,
                "cell_type_actual": cell_type,  # If remapped, update this
                "species_requested": prediction_task["species"],
                "species_actual": prediction_task["species"],
                "predictions": formatted_preds,
            }

            # Append results for current prediction task to the main JSON object
            json_return["prediction_tasks"].append(current_prediction_task)

        # Convert dictionary to JSON object and send back to evaluator
        json_string = json.dumps(json_return)
        try:
            jsonResult_bytes = json_string.encode("utf-8")
            jsonResults_total_bytes = len(jsonResult_bytes)
            client_socket.sendall(struct.pack(">I", jsonResults_total_bytes))
            client_socket.sendall(jsonResult_bytes)
            continue
        except socket.error as e:
            print("server_error: Error sending prediction response: %s" % e)
            # finally:
            client_socket.close()
            # server.close()
            print("Connection to client closed")
            # sys.exit(0)
            break

        # # ---------------------- Close Connection Sockets ----------------------
        # client_socket.close()
        # print("Connection to client closed")
        # # close server socket
        # server.close()


def run_predictor():
    predictor_ip = sys.argv[1]
    predictor_port = int(sys.argv[2])
    # cell_type_matcher_ip = sys.argv[3]
    # cell_type_matcher_port = sys.argv[4]

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # bind the socket to a specific address and port
    server.bind((predictor_ip, predictor_port))
    # listen for incoming connections
    server.listen(0)
    print(f"Listening on {predictor_ip}:{predictor_port}")

    # We want to have multiple evaluators to connect so predictor
    # can take multiple requests (and not just multiple tasks per evaluator)

    # This loop allows the Predictor server to stay running so that different Evaluators can connect
    while True:
        try:
            print("Waiting for an Evaluator to connect")
            # accept incoming connections
            client_socket, client_address = server.accept()
            print(f"Accepted connection from {client_address[0]}:{client_address[1]}")
            # Once connected, receive request
            recv_message_loop(client_socket)
        except Exception as e:
            print(f"Error accepting client: {e}")


run_predictor()
