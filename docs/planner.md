# Planner inference release

Only two executable entry points are provided in `scripts/`:

- `run_planner.py`: ROS planning with PT or TensorRT, or one offline NPZ inference.
- `export_engine.py`: local PT → encoder/planner ONNX → encoder/planner engine conversion.

Model definitions, backends, preprocessing, trajectory generation and ROS callbacks
are implemented under `planner/`. Training, privileged depth expert, auxiliary
supervision, datasets and 3DGS code are not part of this release. The controller
and quadrotor simulator source is provided as a separate catkin workspace under
`Controller/`. Unity remains an external component.

## Environment

Use Python 3.8+ and a CUDA-enabled PyTorch/torchvision pair suitable for your GPU
or JetPack image. Install the pair first, then install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

For ROS mode, ROS1 Noetic, `rospy`, ROS messages and `cv_bridge` must be available
to that interpreter. Build and source the bundled [controller workspace](../Controller/README.md)
that provides `quadrotor_msgs/PositionCommand` before starting planning. Offline NPZ inference
does not require ROS or the controller.

The validation environment uses PyTorch 2.4.1, Ultralytics 8.4.174 and TensorRT
11.1. Install TensorRT separately for your platform. Do not replace JetPack's
TensorRT with an unrelated desktop wheel. The engine runner uses named tensor
APIs available in TensorRT 8.5 and newer; additional versions require testing.

## Weights

The release checkpoint is `weights/geomonoplan.pt`. All required YOLO26-Depth
parameters are included. Its architecture configuration is also embedded, so
no additional YOLO checkpoint, pretraining download or research directory is
required. Loading checks every model tensor with `strict=True`.

The release format is a dictionary with `format_version`, `model_config`, and
`state_dict`, loaded with `torch.load(..., weights_only=True)`. It includes
inference parameters and persistent buffers, not optimizer state or training
configuration. Use a release checkpoint matching this implementation; an arbitrary
research `.pth` file is not a supported drop-in replacement.

## Run with PT

With Unity and the controller running and their workspace sourced:

```bash
python scripts/run_planner.py --backend pt
```

Topic names, camera pitch, goal, arrival threshold and timing are in
`config/inference.yaml`. Each processed image uses the latest received odometry,
as in the original RGB planning callback. The planner does not match or reject
image/odometry pairs based on header timestamps or arrival-time skew.
Odometry supplies world-frame linear velocity, body-to-world xyzw orientation
and world position. The planner retains the original camera/body
conventions, including the configurable camera pitch.

## Convert and run with TensorRT

```bash
python -m pip install -r requirements.txt
python scripts/export_engine.py
python scripts/run_planner.py --backend engine
```

Conversion runs on the target device. It exports fixed batch-one shapes using
ONNX opset 16 and defaults to FP16. TensorRT versions without `BuilderFlag.FP16`
use a mixed-precision ONNX with FP32 external interfaces. Two engines, their ONNX
intermediates and a manifest are stored in `engines/`.

Conversion follows the original split-encoder export: zero RGB/observation and
identity relative pose for tracing, opset 16, and a 2 GiB builder workspace.
On TensorRT with `BuilderFlag.FP16`, the builder receives the FP32 ONNX with the
FP16 flag. Otherwise it receives a fresh export of the actual half-precision
model with FP32 interfaces. No profiling verbosity override is applied.

Verification follows the original six moving frame pairs (seed 42) and changing
poses, followed by ten warmup calls and a cached stream benchmark (100 calls by
default). Each measured engine planning call must encode exactly one new frame.
The manifest records errors, candidate agreement, PT full-two-frame latency and
cached engine pipeline latency. Finite-value and numerical-tolerance checks gate
replacement of the old engine files. Candidate differences are
recorded without rejecting FP16; `--require-candidate-match` enables the optional
strict check. This sample check is not a flight evaluation.
Use a representative input for more relevant numerical validation:

```bash
python scripts/export_engine.py --verify-input /path/to/inputs.npz
```

An optional verification input is checked after the standard moving-pair test;
it does not replace the original zero-valued tracing inputs.

If FP16 numerical verification fails, inspect the error and try FP32 rather than silently
loosening tolerances:

```bash
python scripts/export_engine.py --precision fp32
```

`--workspace-gib` limits builder scratch memory; the default is 2 GiB. Rebuild
when changing weights, GPU/platform or an incompatible TensorRT installation.
The manifest hashes detect accidentally mixed encoder/planner files. Engine mode
does not import the network or Ultralytics, and does not require the PT file.
Switching backends is explicit; load failures do not trigger a silent fallback.

## Offline input/output contract

```bash
python scripts/run_planner.py --backend pt --device cpu \
    --input /path/to/inputs.npz --output /tmp/prediction.npz
```

The NPZ contains float32 arrays:

| Key | Shape | Convention |
| --- | --- | --- |
| `rgb` | `[1,2,3,270,480]` | Previous/current **BGR**, CHW, divided by 255 |
| `obs` | `[1,9]` | Metric velocity, acceleration, goal displacement in planner coordinates |
| `pose` | `[1,7]` | Previous body pose expressed in current body frame: xyzw, then xyz translation |

The model applies observation normalization and terminal-state decoding.
Do not apply them a second time. Outputs are `endstate` `[1,9,3,5]` (position
offset xyz, velocity xyz, acceleration xyz) and `score` `[1,3,5]`; higher scores
are preferred. Online trajectory generation rotates these into world coordinates
and adds the chosen start position to the position offset.

Relative artifact paths in configuration and `--weights`/`--engine-dir` resolve
against the repository root, independently of the shell's current directory.
Explicit input/output NPZ paths resolve against the shell's current directory.

## Online behavior

Set `ros.verbose: true` in `config/inference.yaml` to print stage latency once
per second: image processing, input preparation, inference, trajectory generation,
visualization and total. The original post-inference CUDA synchronization is used
regardless of verbose mode. The switch only controls logging.

The RGB callback, history/relative pose, observation construction, world-frame
selection, quintic interpolation, yaw limiting and arrival handling follow
`YOPO/test_yopo_ros.py`. Planning runs in the image callback using the latest
odometry. `ros.plan_hz` sets the planner output publication rate (default 50 Hz).
The publication timer advances the polynomial by the node's fixed `ctrl_dt`
(currently 0.02 s) before emitting each reference command. Trajectory sampling
and publication period are independent. Trajectory start position and velocity
are read after inference.
There is no added worker, timestamp matching, input timeout hold, automatic
reset on backward timestamps, or wall-clock trajectory sampling.

The selected trajectory is published on `/trajectory`
(`nav_msgs/Path`), sampled from the same selected polynomials. The original
PointCloud2 lattice/all-candidate displays are not published by this adapter.
The goal remains configurable in YAML. As in the original callback, a new goal
from `/move_base_simple/goal` uses the message x/y and fixes z to **1.5 m**; it only
clears the arrival flag and does not reset history or the current trajectory.
Arrival immediately publishes a zero-velocity/acceleration hold. An expired
trajectory stops publishing fresh commands, following the original behavior.
Normal mode starts immediately; `validation_mode: true` enables the original
`/yopo_net/flight_enable` start/stop topic and clears frame history/cache on toggles.
Messages on `/sim/reset_pose` clear the trajectory and frame cache and rebase
the reference state while preserving the stopped/arrived status.

Set the planning speed using `velocity` (m/s) in `config/inference.yaml`:

```yaml
velocity: 3.0
```

Both PT and engine backends read this value at startup. Planning speed is
configured in YAML rather than through a command-line argument.

The checkpoint/engine manifest retains the model's reference normalization. For
requested speed `v`, with reference speed `v0`, the original planner uses
`r = v / v0`: acceleration scale becomes `a0 * r**2`, and trajectory duration
becomes `T0 / r`. The common runtime adapter divides observed velocity by `r`
and observed acceleration by `r**2` before the fixed graph, then multiplies decoded
terminal velocity by `r` and acceleration by `r**2`. Position offsets, goal input,
relative pose and image features keep their original geometric units.

Thus speed changes do not require new weights or rebuilt engines. For this release,
3 m/s gives a 1.5 m/s^2 acceleration scale and a 3.333 s segment; the default 6 m/s
uses a 6 m/s^2 scale and a 1.667 s segment. Speed must be finite and positive.
Conversion verification always runs at the model's reference speed.

## Verification

Release weights and PT inference were compared with the original RGB model.
Fresh FP32/FP16 encoder and planner ONNX computation graphs, including constants
and parameter values, match the original exporter after ignoring automatic
tensor symbol numbering. TensorRT builder settings follow the same split export.

Each local engine build records its precision, artifact hashes, numerical errors,
candidate differences and cached inference latency in `engines/manifest.json`.
The included local engines use FP16. Engine artifacts must be rebuilt on the
target platform. These checks are not a closed-loop flight equivalence claim.

This is an inference release. It does not expose the training pipeline or claim
that the released model reproduces every preprint experiment. See the main README
for the relationship to the paper and demonstrations.
