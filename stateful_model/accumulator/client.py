import numpy as np
from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc as grpcclient

model_name = "accumulator"
real_input = [
    ("+", 3),
    ("-", 2),
    ("+", 9),
    ("+", -5),
    ("+", 0),
    ("-", 21),
    ("+", 7),
    ("+", 3),
]
expect_output = [3, 1, 10, 5, 5, -16, -9, -6]
sequence_id = 10087

with grpcclient.InferenceServerClient("Localhost:8000") as client:
    for idx, data in enumerate(real_input):
        input0_data = np.array([[data[0]]], dtype=np.dtype(object))
        input1_data = np.array([[data[1]]], dtype=np.int64)

        inputs = [
            grpcclient.InferInput(
                "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
            ),
            grpcclient.InferInput(
                "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
            ),
        ]
        inputs[0].set_data_from_numpy(input0_data)
        inputs[1].set_data_from_numpy(input1_data)

        outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]
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
        assert expect_output[idx] == response.as_numpy("OUTPUT0")[0][0]
