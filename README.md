<h1 align="left">Learning Geometry-Aware Planning Representations for Monocular Flight with Cross-Modal Supervision</h1>

<p align="center">
  <img src="assets/arxiv.svg" alt="arXiv" height="28">
  <a href="https://mioulo.github.io/"><img src="assets/homepage.svg" alt="Personal homepage" height="28"></a>
</p>

## 🌏 Overview

![Overview of simulation and real-world flight experiments](assets/overview.png)

🎬 Watch the paper introduction video on [🌐 Wencan Feng](https://mioulo.github.io/)'s homepage.

## 🎮 Simulation Flights

### 🕹️ Unity

<p align="center"><img src="assets/simulation/unity/random1.gif" alt="unity flight 1" width="33.33%"><img src="assets/simulation/unity/random2.gif" alt="unity flight 2" width="33.33%"><img src="assets/simulation/unity/chemistryfactory.gif" alt="unity flight 3" width="33.33%"><br><img src="assets/simulation/unity/factory.gif" alt="unity flight 1" width="33.33%"><img src="assets/simulation/unity/forest.gif" alt="unity flight 2" width="33.33%"><img src="assets/simulation/unity/office.gif" alt="unity flight 3" width="33.33%"><br><img src="assets/simulation/unity/park.gif" alt="unity flight 1" width="33.33%"><img src="assets/simulation/unity/village1.gif" alt="unity flight 2" width="33.33%"><img src="assets/simulation/unity/village2.gif" alt="unity flight 3" width="33.33%"></p>

<p align="left"><sub>🟢 Training environments　 🔵 Validation environments</sub></p>

### ✨ 3D Gaussian Splatting (3DGS)

<p align="center"><img src="assets/simulation/3dgs/abd.gif" alt="3DGS flight demonstration" width="50%"><img src="assets/simulation/3dgs/eastgate.gif" alt="3DGS flight demonstration" width="50%"><br><img src="assets/simulation/3dgs/changfang.gif" alt="3DGS flight demonstration" width="33.33%"><img src="assets/simulation/3dgs/jinnan.gif" alt="3DGS flight demonstration" width="33.33%"><img src="assets/simulation/3dgs/wdd.gif" alt="3DGS flight demonstration" width="33.33%"></p>

## 🚁 Real-World Flights

<p align="center"><img src="assets/real-world/1st.gif" alt="Real-world flight view 1" width="57.1428%"><img src="assets/real-world/3rd.gif" alt="Real-world flight view 2" width="42.8571%"></p>

## 🧠 Method

![Network architecture and three-stage training](assets/method.png)

Our monocular planner generates smooth trajectories from **two consecutive RGB images, their metric relative pose, and the vehicle state**. During training, a privileged depth expert transfers planning-relevant geometry to the RGB branch through feature alignment, aided by an auxiliary coarse-depth objective. Pose-conditioned temporal fusion brings together geometric cues across frames.

1. **Learn from depth:** pretrain the depth expert and shared planner with trajectory objectives.
2. **Transfer geometry:** freeze the expert and shared planner; train the RGB branch with feature alignment and auxiliary depth supervision.
3. **Adapt for RGB:** freeze RGB perception and adapt the shared planner to its learned representation.

At deployment, the depth expert and auxiliary depth decoder are removed. The RGB planner selects a smooth trajectory from predicted motion primitives, without explicitly reconstructing depth.

## 🛠️ Real-World Deployment

<table>
  <tr>
    <td rowspan="6" width="36%" align="center" valign="middle"><img src="assets/drone.png" alt="Onboard quadrotor hardware" width="340"></td>
    <th>Component</th><th>Configuration</th>
  </tr>
  <tr><td>Onboard computer</td><td>NVIDIA Jetson Orin NX</td></tr>
  <tr><td>Visual sensor</td><td>Monocular fisheye camera</td></tr>
  <tr><td>State estimation</td><td>VINS-Mono with calibrated fisheye images and IMU</td></tr>
  <tr><td>Onboard pipeline</td><td>State estimation, planning, and control</td></tr>
  <tr><td>Sim-to-real transfer</td><td>No real-world fine-tuning</td></tr>
</table>

## 🤗 Citation

```bibtex
% BibTeX entry coming soon.
```

## 📌 Note

The paper has not yet been accepted for publication. PT and TensorRT inference code is available; see the [deployment guide](docs/planner.md) for setup, engine export, and ROS usage.
