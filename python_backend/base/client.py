import numpy as np
import tritonclient.grpc as grpcclient


class Cfg:
    url = "Localhost:8000"
    model_name = "base"
    model_version = ""
    verbose = False


FLAGS = Cfg()


def base_request(batched_data):
    triton_client = grpcclient.InferenceServerClient(
        url=FLAGS.url, verbose=FLAGS.verbose
    )
    model_metadata = triton_client.get_model_metadata(
        model_name=FLAGS.model_name, model_version=FLAGS.model_version
    )
    model_config = triton_client.get_model_config(
        model_name=FLAGS.model_name, model_version=FLAGS.model_version
    )
    input_name = model_metadata.inputs[0].name
    input_type = model_metadata.inputs[0].datatype
    output_name = model_metadata.outputs[0].name

    inputs = [triton_client.InferInput(input_name, batched_data.shape, input_type)]
    inputs[0].set_data_from_numpy(batched_data)

    outputs = [triton_client.InferRequestedOutput(output_name)]

    response = triton_client.infer(
        FLAGS.model_name,
        inputs,
        request_id=str(sent_count),
        model_version=FLAGS.model_version,
        outputs=outputs,
    )
    output_array = response.as_numpy(output_name)
    return output_array


if __name__ == "__main__":
    batched_data = np.array([[3.0], [3.6], [2.7], [8.6]], dtype=np.float32)
    reponse = base_request(batched_data)
    print(response)
