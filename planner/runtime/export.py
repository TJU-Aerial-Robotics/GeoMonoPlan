"""ONNX export, local TensorRT build and PT/engine verification."""
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import tempfile
import time
import torch
from torch import nn

from planner.models.network import load_model
from planner.runtime.policy import ENGINE_SCALES, Policy


class Encoder(nn.Module):
    def __init__(self, model, fp16=False):
        super().__init__()
        self.policy = model
        self.fp16 = fp16

    def forward(self, rgb):
        result = self.policy.encode_frame(rgb.half() if self.fp16 else rgb)
        return result.float() if self.fp16 else result


class Planner(nn.Module):
    def __init__(self, model, fp16=False):
        super().__init__()
        self.policy = model
        self.fp16 = fp16

    def forward(self, features, obs, pose):
        if self.fp16:
            features, obs, pose = features.half(), obs.half(), pose.half()
        endstate, score = self.policy.plan_from_features(features, obs, pose)
        return (endstate.float(), score.float()) if self.fp16 else (endstate, score)


@contextmanager
def export_attention(model):
    # Avoid PyTorch's native fused MHA path in the legacy ONNX exporter.
    backend = getattr(torch.backends, "mha", None)
    original = backend.get_fastpath_enabled() if backend is not None else None
    attention = [(module, module.training) for module in model.modules() if isinstance(module, nn.MultiheadAttention)]
    try:
        if backend is not None:
            backend.set_fastpath_enabled(False)
        for module, _ in attention:
            module.train()
        yield
    finally:
        for module, training in attention:
            module.train(training)
        if backend is not None:
            backend.set_fastpath_enabled(original)


def export_onnx(model, directory, inputs, opset, fp16=False):
    import onnx
    if fp16:
        # Export actual half-precision operations, including explicit casts for
        # generated angular constants; graph-wide dtype rewriting loses these.
        model = copy.deepcopy(model).half().eval()
    rgb, obs, pose = inputs
    with torch.no_grad():
        frame_features = Encoder(model, fp16)(rgb[:, 0].contiguous())
        features = torch.stack((frame_features, frame_features), dim=1)
    for name, wrapper, arguments, input_names, output_names in (
        ("encoder", Encoder(model, fp16), (rgb[:, 0].contiguous(),), ["rgb"], ["features"]),
        ("planner", Planner(model, fp16), (features, obs, pose), ["features", "obs", "pose"], ["endstate", "score"]),
    ):
        path = directory / (name + ("_fp16.onnx" if fp16 else ".onnx"))
        print("Exporting {}".format(path), flush=True)
        wrapper.eval()
        with export_attention(model):
            torch.onnx.export(wrapper, arguments, str(path), input_names=input_names,
                              output_names=output_names, opset_version=opset, do_constant_folding=True,
                              training=torch.onnx.TrainingMode.PRESERVE)
        onnx.checker.check_model(str(path))


def build_engine(onnx_path, engine_path, precision, workspace_gib):
    import tensorrt as trt
    build_path = onnx_path
    fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    if precision == "fp16" and fp16_flag is None:
        build_path = onnx_path.with_name(onnx_path.stem + "_fp16.onnx")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    explicit = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network = builder.create_network(0 if explicit is None else 1 << int(explicit))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(build_path.read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(index)) for index in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 1024 ** 3))
    if precision == "fp16" and fp16_flag is not None:
        config.set_flag(fp16_flag)
    print("Building {} ({})".format(engine_path.name, precision), flush=True)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT could not build {}".format(engine_path))
    engine_path.write_bytes(bytes(engine))


def make_inputs(config, device):
    # Original example_inputs: zero RGB/observation and identity relative pose.
    rgb = torch.zeros((1, 2, 3, config["image_height"], config["image_width"]), dtype=torch.float32, device=device)
    obs = torch.zeros((1, 9), dtype=torch.float32, device=device)
    pose = torch.zeros((1, 7), dtype=torch.float32, device=device)
    pose[:, 3] = 1.0
    return rgb, obs, pose


def compare_outputs(expected, observed, atol, rtol, require_candidate_match=False):
    result = {}
    for name, reference, actual in zip(("endstate", "score"), expected, observed):
        if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
            raise RuntimeError("Nonfinite {} during verification".format(name))
        difference = (reference - actual).abs()
        result[name + "_max_abs_error"] = difference.max().item()
        result[name + "_mean_abs_error"] = difference.mean().item()
        if not torch.allclose(reference, actual, atol=atol, rtol=rtol):
            raise RuntimeError("{} verification failed: max absolute error {}".format(name, difference.max().item()))
    reference_action, actual_action = int(expected[1].argmax()), int(observed[1].argmax())
    result["pt_candidate"] = reference_action
    result["candidate"] = actual_action
    result["candidate_match"] = reference_action == actual_action
    if reference_action != actual_action and require_candidate_match:
        top_scores = torch.topk(expected[1].flatten(), 2).values
        raise RuntimeError(
            "PT and engine select different candidates: {} vs {}; "
            "PT score margin={}, score max absolute error={}".format(
                reference_action, actual_action, (top_scores[0] - top_scores[1]).item(),
                result["score_max_abs_error"],
            )
        )
    return result


@torch.inference_mode()
def benchmark_split(model, policy, iterations, atol, rtol, require_candidate_match=False):
    """Original six moving pairs and one-new-frame cached-stream benchmark.

    Candidate differences are recorded, matching the original acceptance of
    FP16 numerical variation. Strict candidate equality is an optional check.
    """
    device = policy.device
    rgb, obs, pose = make_inputs(model.config, device)
    generator = torch.Generator(device=device).manual_seed(42)
    frames = torch.rand((3, *rgb.shape[2:]), generator=generator, device=device)
    pairs = [torch.stack((frames[i], frames[(i + 1) % 3]), dim=0)[None] for i in range(3)]
    obs = torch.randn(obs.shape, generator=generator, device=device) * 0.2
    samples = []
    for i in range(6):
        current_pose = pose.clone()
        current_pose[:, 4] = 0.1 * (i + 1)
        current_pose[:, 5] = -0.05 * i
        expected = model(pairs[i % 3], obs, current_pose)
        actual = policy(pairs[i % 3], obs, current_pose, frame_ids=(i, i + 1), return_tensors=True)
        torch.cuda.synchronize(device)
        samples.append(compare_outputs(expected, actual, atol, rtol, require_candidate_match))
    policy.clear()
    for i in range(10):
        model(pairs[i % 3], obs, pose)
        policy(pairs[i % 3], obs, pose, frame_ids=(i, i + 1), return_tensors=True)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for i in range(iterations):
        model(pairs[(i + 10) % 3], obs, pose)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    misses_before = policy.feature_cache.misses
    t2 = time.perf_counter()
    for i in range(10, 10 + iterations):
        policy(pairs[i % 3], obs, pose, frame_ids=(i, i + 1), return_tensors=True)
    torch.cuda.synchronize(device)
    t3 = time.perf_counter()
    fresh_frames = policy.feature_cache.misses - misses_before
    if fresh_frames != iterations:
        raise RuntimeError("Streaming benchmark encoded {} frames, expected {}".format(fresh_frames, iterations))
    return {
        "status": "passed", "moving_pairs": samples,
        "candidate_mismatches": sum(not value["candidate_match"] for value in samples),
        "require_candidate_match": require_candidate_match,
        "endstate_max_abs_error": max(value["endstate_max_abs_error"] for value in samples),
        "score_max_abs_error": max(value["score_max_abs_error"] for value in samples),
        "torch_full_two_frame_ms": 1000 * (t1 - t0) / iterations,
        "cached_pipeline_ms": 1000 * (t3 - t2) / iterations,
        "fresh_encodings": fresh_frames, "planning_calls": iterations,
    }


def convert(settings, args):
    from planner.settings import resolve_path
    weights = resolve_path(settings["weights"])
    destination = resolve_path(settings["engine_dir"])
    device = torch.device(settings["device"])
    if not args.onnx_only and device.type != "cuda":
        raise ValueError("Building and verifying TensorRT engines requires a CUDA device")
    if args.opset < 16:
        raise ValueError("Use ONNX opset 16 or newer for the planning model")
    model = load_model(weights, device)
    if not args.onnx_only:
        for name, value in ENGINE_SCALES.items():
            if not math.isclose(float(model.config[name]), value, rel_tol=1e-6):
                raise ValueError("Engine runtime expects released model normalization: {}={}".format(name, value))
    inputs = make_inputs(model.config, device)
    destination.mkdir(parents=True, exist_ok=True)
    # Verify in a temporary directory before replacing any successful build.
    with tempfile.TemporaryDirectory(prefix=".build-", dir=str(destination)) as temporary:
        directory = Path(temporary)
        export_onnx(model, directory, inputs, args.opset)
        if args.onnx_only:
            for name in ("encoder.onnx", "planner.onnx"):
                (directory / name).replace(destination / name)
            print("Exported both ONNX models to {}".format(destination))
            return
        import tensorrt as trt
        if args.precision == "fp16" and getattr(trt.BuilderFlag, "FP16", None) is None:
            export_onnx(model, directory, inputs, args.opset, fp16=True)
        for name in ("encoder", "planner"):
            build_engine(directory / (name + ".onnx"), directory / (name + ".engine"), args.precision, args.workspace_gib)
        # Verify the graph at its embedded reference speed, independently of
        # the user's runtime planning speed in inference.yaml.
        engine_settings = dict(settings, engine_dir=str(directory), velocity=model.config["velocity"])
        policy = Policy(engine_settings, "engine")
        verification = benchmark_split(model, policy, args.benchmark, args.atol, args.rtol, args.require_candidate_match)
        if args.verify_input:
            from planner.processing.inputs import load_inputs
            verify_inputs = tuple(torch.as_tensor(value, device=device) for value in load_inputs(args.verify_input))
            with torch.inference_mode():
                verification["provided_input"] = compare_outputs(
                    model(*verify_inputs), policy(*verify_inputs, return_tensors=True), args.atol, args.rtol,
                    args.require_candidate_match,
                )
        for name in ("encoder.engine", "planner.engine"):
            (directory / name).replace(destination / name)
        print(json.dumps(verification, indent=2))
        print("Verified engines saved to {}".format(destination))
