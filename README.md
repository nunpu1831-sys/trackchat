# TrackChat — Real-time Markerless Full-Body AR

Attach 3D items to **anyone in any video or live stream**, in real time — no mocap suit, no markers, no special camera.

Point the tool at any video (screen capture) → it reconstructs **everyone's full-body 3D mesh** live → **click a person** to put 3D items on them (a crown on the head, a club in the hand) that **follow the body, joints, and orientation** frame by frame. Multi-person, whole-body (incl. hands), single ordinary RGB input.

> Research / portfolio prototype. Non-commercial.

![demo](docs/demo.gif)
<!-- 데모: docs/demo.gif 넣기. 전체 영상은 YouTube 링크로 -->
**Demo video:** _(YouTube 링크 넣기)_

---

## What it does
- **Real-time 3D people** from any 2D video/stream via on-screen capture.
- **Click-to-select** a person and attach 3D items to them.
- Items are rigidly bound to body joints:
  - **Crown** → top of the head, follows head tilt & facing direction.
  - **Club** → right hand, oriented by the actual **wrist rotation** (bends when the wrist bends).
- **Multi-person** tracking with stable per-person identity/color.
- **Live metrics HUD** (received fps, RTT = inference + network, bandwidth).

## How it works (pipeline)
1. **Capture** — browser `getDisplayMedia` grabs the screen → JPEG frames → WebSocket to the server.
2. **Perception** — [Multi-HMR](https://github.com/naver/multi-hmr) (ECCV'24, NAVER): single-shot, multi-person, whole-body **SMPL-X** mesh + 3D joints + per-joint rotations, using the **Anny** body model (Apache-2.0). ~30–70 ms/frame on GPU.
3. **Transport** — vertices sent as **int16-quantized binary** (~5× smaller than JSON) + head/wrist joint transforms per person.
4. **Client (three.js)** — per-person tracker (greedy centroid matching + short "coasting" so brief misses don't drop identity), a **delayed-interpolation buffer** that renders smooth 60 fps between sparse model frames, and **joint-anchored item attachment** (joint position + offset + axis from head/wrist rotation). Free-fly camera (WASD + mouse).

## Tech stack
Python · PyTorch · Multi-HMR · SMPL-X / Anny · FastAPI + WebSocket · three.js · cloudflared

## Run it
Requires a CUDA GPU (tested on RTX 4090).
1. Clone [Multi-HMR](https://github.com/naver/multi-hmr) and set up its `multihmr` environment.
2. Download a checkpoint (e.g. `multiHMR_672_L_anny`) from the Multi-HMR repo, and the **SMPL-X** neutral model from [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) (register first — **not** redistributed here).
3. Drop `live_mhmr11.py` into `multi-hmr/`, then:
   ```bash
   pip install -q fastapi uvicorn websockets
   xvfb-run -a python live_mhmr11.py      # serves on :8000
   ```
4. Open `http://localhost:8000` (or a tunnel URL) → **Start capture** → pick the video window → click a person.

Controls: **click a person** = attach items · **click empty space** = fly (WASD / mouse / Space / Shift / wheel) · **C / V** = tweak club axis.

## Honest limitations
- Real-time throughput is bounded by **server→client bandwidth**, not model speed (the model is ~30–70 ms; the bottleneck is streaming the mesh). On a distant cloud GPU, delivered frame-rate drops and motion looks choppy.
- Tracking IDs can swap on **hard scene cuts** or when frame-rate is very low.
- Single fixed viewpoint; occluded / out-of-frame geometry is not recovered.

## License & attribution
- **My code in this repo: [MIT](LICENSE).**
- **Multi-HMR** (NAVER) — code & checkpoints are **non-commercial** (research/education/artistic). Checkpoints are **not** included here; get them from the official repo.
- **SMPL-X** — **non-commercial**, **not** redistributed; download yourself after registering.
- **Anny** (Apache-2.0), **Depth Anything V2** (Apache-2.0), **three.js** (MIT).
- Demo footage: [Pexels](https://www.pexels.com/) (free to use).

**This project is a non-commercial research/portfolio demo.** If you use Multi-HMR, please cite:
```bibtex
@inproceedings{multi-hmr2024,
  title={Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot},
  author={Baradel, Fabien and Armando, Matthieu and Galaaoui, Salma and Br{\'e}gier, Romain and Weinzaepfel, Philippe and Rogez, Gr{\'e}gory and Lucas, Thomas},
  booktitle={ECCV}, year={2024}
}
```
