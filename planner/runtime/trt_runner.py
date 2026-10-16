"""Original TensorRT component execution on the current Torch CUDA stream."""
from threading import RLock
import torch


class Engine:
    """Execute a named component on the current Torch CUDA stream."""

    def __init__(self, engine_path, input_names, output_names, device="cuda"):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("TensorRT inference requires CUDA.")
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError("TensorRT Python package is required to load .engine files.") from exc
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with torch.cuda.device(self.device):
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
            self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {engine_path}")
        actual_inputs, actual_outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            target = actual_inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else actual_outputs
            target.append(name)
        if set(actual_inputs) != set(input_names) or set(actual_outputs) != set(output_names):
            raise RuntimeError(
                f"Unexpected TensorRT bindings in {engine_path}: "
                f"inputs={actual_inputs}, outputs={actual_outputs}; "
                f"expected inputs={list(input_names)}, outputs={list(output_names)}"
            )
        self.input_names, self.output_names = tuple(input_names), tuple(output_names)
        self._lock = RLock()
        self._last_execution = None

    @torch.inference_mode()
    def run(self, **inputs):
        with self._lock, torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            # A context may be used from successive ROS callback streams, but
            # must not execute overlapping requests on different streams.
            if self._last_execution is not None:
                stream.wait_event(self._last_execution)
            bound_inputs = {}
            for name in self.input_names:
                source = inputs[name]
                if not source.is_cuda or source.device != stream.device:
                    raise ValueError(f"TensorRT input {name} must be on {stream.device}")
                tensor = source.to(dtype=self._torch_dtype(self.engine.get_tensor_dtype(name))).contiguous()
                if not self.context.set_input_shape(name, tuple(tensor.shape)):
                    raise ValueError(f"TensorRT input {name} has incompatible shape {tuple(tensor.shape)}")
                self.context.set_tensor_address(name, tensor.data_ptr())
                tensor.record_stream(stream)
                bound_inputs[name] = tensor
            outputs = []
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dim < 0 for dim in shape):
                    raise RuntimeError(f"Unresolved TensorRT output shape for {name}: {shape}")
                # Fresh storage matters: cached encoder outputs must survive
                # subsequent calls without being overwritten by a shared buffer.
                tensor = torch.empty(shape, dtype=self._torch_dtype(self.engine.get_tensor_dtype(name)), device=stream.device)
                self.context.set_tensor_address(name, tensor.data_ptr())
                outputs.append(tensor)
            if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed.")
            self._last_execution = torch.cuda.Event()
            self._last_execution.record(stream)
            return tuple(outputs)

    def _torch_dtype(self, trt_dtype):
        for name, dtype in (("float32", torch.float32), ("float16", torch.float16),
                            ("int32", torch.int32), ("int8", torch.int8), ("bool", torch.bool)):
            if trt_dtype == getattr(self.trt, name, None):
                return dtype
        raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")
