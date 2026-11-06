# heaicare-capture-sim

Virtual capture rig for the heAICare project. It renders synthetic images of a neonatal head from a configurable set of fixed cameras, all triggered at once, so that camera count and placement can be studied for 3D reconstruction of the cranial vault. The images go to heAICare's reconstruction pipeline (outside this repo), and the result is compared against the ground-truth mesh saved with every capture.

> **Status:** v0.1 implemented.

## Scope

In scope now:

- One trigger captures every camera simultaneously, from a GUI button or a CLI command. Both use the same engine, so the same rig gives identical output.
- A synthetic neonatal head with procedural skin texture (real meshes load through the same path later).
- Ideal pinhole cameras defined by datasheet values (resolution, focal length in mm, pixel size in µm).
- Head alone on a neutral background, fixed neutral lighting.
- Output per capture: images, camera parameters (OpenCV convention), ground-truth mesh.

Out of scope: running the reconstruction pipeline, and scoring results.

Planned later: per-camera depth maps and vault masks; lens distortion, noise and blur; configurable lights; a mattress under the head; real head meshes.

## Install

```bash
uv sync
```

## Usage

```bash
# Generate a rig from a preset
heaicare-capture-sim preset ring --n 8 --radius-mm 300 --height-mm 100 -o rig.json
heaicare-capture-sim preset dome --n 12 --radius-mm 300 --min-elevation-deg 20 -o rig.json

# Capture every camera in the rig
heaicare-capture-sim capture --rig rig.json --out captures/run01 [--mesh head.ply]

# Interactive GUI (opens in the browser at http://localhost:8080)
heaicare-capture-sim gui [--rig rig.json] [--mesh head.ply] [--port 8080]
```

Without `--mesh`, the synthetic head is used.

### Preset options

| Preset | Options |
|---|---|
| `ring` | `--n`, `--radius-mm`, `--height-mm` (z of the ring plane), `--target-mm X Y Z` (default origin). Cameras are evenly spaced around the z axis, all aimed at the target. |
| `dome` | `--n`, `--radius-mm`, `--min-elevation-deg`, `--target-mm X Y Z` (default origin). Cameras are spread evenly (Fibonacci pattern) over the hemisphere above the crown, all aimed at the target. |

Both presets accept the camera options `--width`, `--height`, `--focal-length-mm` and `--pixel-size-um`. Defaults: 1920×1080, 8 mm, 3.45 µm.

### GUI

- **3D view:** the head plus one frustum per camera. The selected camera is highlighted; clicking a frustum selects it.
- **Camera panel:** a camera dropdown, then fields for the selected camera (eye, target, up, focal length, pixel size, resolution), with Add, Duplicate and Remove.
- **Presets:** ring and dome, with the same parameters as the CLI. Applying a preset replaces the camera list.
- **Preview:** an Open3D render of the selected camera (scaled down), refreshed after edits. It uses the same renderer as capture.
- **Buttons:** Load rig, Save rig, and Capture. Capture writes `captures/<timestamp>/` and shows the path.

## Coordinate frame

Millimetres, right-handed, **z up**. The head centre is at the origin, the **crown points to +z**, and the **face points to +y** (ears along ±x).

## Rig file

A rig is always an explicit list of cameras; presets just generate this list.

```json
{
  "schema_version": 1,
  "cameras": [
    {
      "name": "cam00",
      "width": 1920, "height": 1080,
      "focal_length_mm": 8.0, "pixel_size_um": 3.45,
      "eye_mm": [300.0, 0.0, 150.0],
      "target_mm": [0.0, 0.0, 0.0],
      "up": [0.0, 0.0, 1.0]
    }
  ]
}
```

- Square pixels. The focal length in pixels is `focal_length_mm * 1000 / pixel_size_um`.
- The principal point is the image centre, `((width-1)/2, (height-1)/2)`.
- Rejected: duplicate names, `eye_mm == target_mm`, and `up` parallel to the viewing direction.

## Output folder

```
<out>/
  images/<camera name>.png   8-bit RGB, neutral grey background
  cameras.json               camera parameters (below)
  rig.json                   copy of the input rig
  ground_truth.ply           head geometry in the world frame, mm (geometry only)
```

`cameras.json`:

```json
{
  "created": "2026-11-06T15:30:00Z",
  "version": "0.1.0",
  "mesh_source": "synthetic",
  "cameras": [
    {
      "name": "cam00",
      "image": "images/cam00.png",
      "width": 1920, "height": 1080,
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "R": [[...], [...], [...]],
      "t_mm": [tx, ty, tz]
    }
  ]
}
```

`K`, `R` and `t_mm` use the OpenCV convention. `R` and `t_mm` map world to camera (`x_cam = R @ x_world + t`). The camera axes are x right, y down, z forward.

## Errors

| Situation | Behaviour |
|---|---|
| Invalid rig (schema or geometry) | Every bad field is listed. The CLI exits with code 2; the GUI shows the message and keeps running. |
| `--out` exists and isn't empty | The CLI refuses and exits with code 2. |
| Unreadable mesh | Exit code 2 with the path and the reason. |
| Capture fails partway | No partial folder is left. Output is written to a temporary folder next to the target and renamed only when complete. |
| A camera sees none of the head | A warning; the image is still captured. That's a valid (bad) rig. |
