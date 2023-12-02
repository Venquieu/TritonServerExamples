import numpy as np
from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

model_name = "ctc_decode_model"
real_input = ["---", "---c", "cc", "cccc", "aaaa", "aa", "---tt", "tt"]
expect_output = ["", "c", "c", "c", "ca", "ca", "cat", "cat"]
sequence_id = 10086

with httpclient.InferenceServerClient("Localhost:8000") as client:
    for idx, data in enumerate(real_input):
        input0_data = np.array([[data]], dtype=np.dtype(object))

        inputs = [
            httpclient.InferInput(
                "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
            )
        ]
        inputs[0].set_data_from_numpy(input0_data)

        outputs = [httpclient.InferRequestedOutput("OUTPUT0")]
        response = client.infer(
            model_name,
            inputs,
            outputs=outputs,
            request_id=str(idx),
            sequence_id=sequence_id,
            sequence_start=(idx == 0),
            sequence_end=(idx == len(real_input) - 1),
        )

        result = response.get_response()
        print(
            "Get response from {}th chunk: {}".format(idx, response.as_numpy("OUTPUT0"))
        )
        assert expect_output[idx] == response.as_numpy("OUTPUT0")[0][0].decode("utf-8")
