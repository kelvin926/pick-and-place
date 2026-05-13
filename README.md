# pick-and-place

Isaac Sim / IsaacLab pick-and-place demos run from `E:\pick-and-place`.

The workspace expects:

- Isaac Sim 5.1.0 at `E:\isaac-sim-5.1.0`
- IsaacLab at `E:\IsaacLab`
- Conda environment `env_isaaclab_2.3.1`

Run the Agibot A2D cube demo:

```powershell
.\run_isaaclab.ps1 -OmniRootName .omni_agibot_a2d_cube_qhd `
  .\scripts\play_agibot_a2d_cube_pickplace.py `
  --headless --device cuda:0 --num_envs 1 `
  --max_steps 900 --drive_steps 180 `
  --record_frames --capture_every 4 `
  --frames_dir artifacts\frames\agibot_a2d_cube_pickplace_qhd `
  --summary artifacts\eval\agibot_a2d_cube_pickplace_qhd.json
```

Final demo outputs are under `artifacts/videos` and validation summaries are under `artifacts/eval`.

