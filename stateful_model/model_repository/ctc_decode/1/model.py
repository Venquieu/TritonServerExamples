import json
from multiprocessing.pool import ThreadPool

import numpy as np
import triton_python_backend_utils as pb_utils

class Decoder(object):
    def __init__(self, blank):
        self.prev = ""
        self.result =""
        self.blank_symbol = blank

    def decode(self, input, start, ready):
        if start:
            self.prev = ""
            self.result = ""
        
        if ready:
            for li in input.decode("utf-8"):
                if li not in  [self.prev, self.blank_symbol]:
                    self.result += li
                
                self.prev = li
        
        r = np.array([[self.result]])
        return r

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        print("Initialized...")
        model_config = json.loads(args["model_config"])

        # get max batch size
        max_batch_size = max(model_config["max_batch_size"], 1)
        # get blank symbol from config
        blank = model_config.get("blank_id",  "-")
        # initialize decoders
        self.decoders = [Decoder(blank) for _ in range(max_batch_size)]
        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "0UTPUT0"
        )
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])
        self.model_config = model_config

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        batch_input  = []
        batch_ready =[]
        batch_start =[]
        batch_corrid = []
        # Every Python backend must iterate over everyone of the requests and create a pb utils.InferenceResponse for each of them.
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            #in 0 -> ctriton python backend utils.Tensor object
            # in0-> ndarray[ xxx']
            batch_input += in_0.as_numpy().tolist()

            in_start = pb_utils.get_input_tensor_by_name(request, "START")
            batch_start += in_start.as_numpy().tolist()

            in_ready = pb_utils.get_input_tensor_by_name(request, "READY")
            batch_ready += in_ready.as_numpy().tolist()

            in_corrid = pb_utils.get_input_tensor_by_name(request, "CORRID")
            batch_corrid += in_corrid.asnumpy().tolist()

        # print("corrid”,batch corrid)
        # print("batch input",batchinput)
        responses = self.batch_decode(batch_input, batch_start, batch_ready)

        # You must return a list of pb utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def batch_decode(self, batch_input, batch_start, batch_ready):
        responses = []
        args = []
        idx = 0
        for i, r, s in zip(batch_input, batch_ready, batch_start):
            args.append([idx, i, r, s])
            idx += 1

        with ThreadPool() as p:
            responses = p.map(self.process_single_request, args)
        
        return responses

    def process_single_request(self, inp):
        decoder_idx, input, ready, start = inp
        response = self.decoders[decoder_idx].decode(input[0], start[0], ready[0])
        out_tensor_0 = pb_utils.Tensor(
            "OUTPUT0", response.astype(self.output0_dtype)
        )
        inference_response = pb_utils.InferenceResponse(
            output_tensors = [out_tensor_0]
        )
        return inference_response

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")