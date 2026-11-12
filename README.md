# heaicare-capture-sim

Virtual capture rig for the heAICare project. It renders synthetic images of a neonatal head from a configurable set of fixed cameras, all triggered at once, so that camera count and placement can be studied for 3D reconstruction of the cranial vault. The images go to heAICare's reconstruction pipeline (outside this repo), and the result is compared against the ground-truth mesh saved with every capture.

> **Status:** v0.1 implemented.

## Scope

In scope now:

- One trigger captures every camera simultaneously, from a GUI button or a CLI command. Both use the same engine, so the same rig gives identical output.
- A synthetic neonatal head with procedural skin texture, or a head mesh from a file (any OBJ via `import-mesh`, e.g. from MakeHuman, see below).
- Ideal pinhole cameras defined by datasheet values (resolution, focal length in mm, pixel size in µm).
- Head on a neutral background, or inside a scenario (a room model such as `scenarios/greybox/`, see below), optionally with a full rendered body. Fixed lighting: the head is unlit, the scenario shaded by one fixed light.
- Output per capture: images, camera parameters (OpenCV convention), ground-truth mesh.

Out of scope: running the reconstruction pipeline, and scoring results.

Planned later: per-camera depth maps and vault masks; lens distortion, noise and blur; configurable lights; scenario textures.

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
heaicare-capture-sim capture --rig rig.json --out captures/run01 [--mesh baby_body] [--scenario greybox]

# Interactive GUI (opens in the browser at http://localhost:8080)
heaicare-capture-sim gui [--rig rig.json] [--port 8080]
```

Heads live in the project's `meshes/` folder, **one folder per head** holding its `.obj`, `.mtl` and textures:

```
meshes/
├── baby_head/
│   ├── baby_head.obj
│   ├── baby_head.mtl
│   └── baby_head_0.png …
└── scan_01/
    └── model.obj          (any file name)
```

`capture --mesh` takes `synthetic` (the default), a head's folder name (`--mesh baby_head` loads the one `.obj` in `meshes/baby_head/`; zero or several is an error), or any mesh path (anything with a `/` or an extension). The GUI picks the head in its **Head** panel and starts on the first head in `meshes/`, or the synthetic head if there is none. A mesh must already be in the world frame (mm, see below); an OBJ keeps its textures, and its quads and other polygons are triangulated on load. In a multi-material mesh, the **first material is the head**: it is the ground-truth surface. Every other material (eyes, eyebrows, eyelashes, tongue) is rendered but left out of `ground_truth.ply`. A texture with transparent texels renders see-through; triangles that only touch fully transparent texels are not drawn.

### Importing any mesh

`import-mesh` puts an OBJ from any source (scan, Blender, MakeHuman...) in the world frame and writes a ready-to-use OBJ, typically into its own folder under `meshes/` (created if missing), where the GUI's **Head** panel lists it:

```bash
heaicare-capture-sim import-mesh scan.obj -o meshes/scan_01/scan_01.obj --units mm --up +z --front +y
heaicare-capture-sim import-mesh baby.obj -o meshes/baby/baby.obj --preset makehuman
```

| Option | Meaning | Default |
|---|---|---|
| `--units` | Source units: `m`, `dm`, `cm`, `mm`, `inch` | `mm` |
| `--up` | Source axis pointing to the crown: `+x`, `-x`, `+y`, `-y`, `+z`, `-z` | `+z` |
| `--front` | Source axis the face points along (perpendicular to `--up`) | `+y` |
| `--neck-cut` | Drop every triangle with a vertex more than this fraction of the height below the crown (for whole-body models; the neck is left open) | no cut |
| `--head-length-mm` | Rescale uniformly so the head's front-to-back extent, nose included, is this | no rescale |
| `--keep-body` | With `--neck-cut`: keep the skin below the cut as a rendered-only part (a body that occludes like a real one; never ground truth) | dropped |
| `--preset makehuman` | `--units dm --up +y --front +z --neck-cut 0.25 --head-length-mm 120`; explicit options override it | — |

The defaults change nothing but centring, so a mesh already in the world frame comes out as it went in, and a real scan keeps its true size. The head (ground truth) is the material with the most triangles; it is written as the first material, the other materials as extras. The head's bounding box is centred on the origin. Vertex colours are kept. The axes are never guessed: a wrong `--up` or `--front` puts the face in the wrong place, so check the result in the GUI.

### Heads from MakeHuman

The repository ships one such head, `meshes/baby_head/`: a MakeHuman 1.3.0 export (youngest age, *young asian male* skin, high-poly brown eyes, eyebrow011, eyelashes01, tongue01; all MakeHuman assets are CC0) converted with `--preset makehuman`.

MakeHuman cannot make a newborn (its youngest age is 1 year), so the preset takes a 1-year-old and scales the head down to neonatal size (120 mm long, the synthetic head's length).

1. In MakeHuman, set **Age** to the minimum and pick a textured skin (**Materials → Skin**, e.g. *Young caucasian female*); the default skin has no texture. Choose eyes, eyebrows, eyelashes and tongue under **Geometries** as wanted; set teeth and hair to **None**. The eyebrows, eyelashes, tongue and textured skins come from MakeHuman's system assets pack (CC0).
2. Export as **OBJ** in decimeters (the default).
3. `heaicare-capture-sim import-mesh baby.obj -o meshes/baby/baby.obj --preset makehuman`, then pick it in the GUI's **Head** panel or pass `--mesh baby` to `capture`. Check the neck cut on your first export and adjust `--neck-cut` if it keeps neck or loses chin.

`meshes/baby_body/` is the same export with `--keep-body`: the ground truth is still the head above the neck cut, and the rest of the body (scaled with the head, about 54 cm tall, feet at z = -470 mm) is rendered only. Its pose is whatever the MakeHuman export has (standing); for another pose, such as sitting, pick it in MakeHuman's **Pose/Animate → Pose** tab before exporting, then re-import.

### Scenarios

A scenario is a room model rendered around the head: it occludes and fills the background, but is never part of `ground_truth.ply`. Each scenario is a folder under `scenarios/` with the model file and a `scenario.json`:

```
scenarios/
└── greybox/
    ├── Greybox.fbx
    └── scenario.json
```

```json
{
  "file": "Greybox.fbx",
  "units": "cm",
  "up": "+y",
  "front": "+z",
  "head": [-80.0, 137.04, -50.0]
}
```

| Field | Meaning |
|---|---|
| `file` | Model file next to `scenario.json`; anything Open3D reads (FBX, OBJ, glTF, PLY...) |
| `units` | Model units as Open3D reads them: `m`, `dm`, `cm`, `mm`, `inch` |
| `up`, `front` | Model axes that become world +z (crown) and +y (face), as in `import-mesh` |
| `head` | Model point, in model units, where the head centre (world origin) goes |
| `baby_offset_mm`, `baby_yaw_deg` | Optional, default 0: moves the baby, with its cameras, away from `head` inside the room: a turn about the vertical (counter-clockwise seen from above), then a shift in the room's x, y, z (mm, world axes). Outputs stay head-centred, so the room is rendered moved the other way. The GUI writes these |

The head stays at the world origin, so rigs and ground truth are unchanged; the room is moved around it. Only material base colours are used (textures are ignored). `greybox` is a 4 × 4 m room with six floor-to-ceiling poles around a 90 cm high box; `head` puts `meshes/baby_body` standing on the box, facing the room's +z. With another head or pose, adjust `head[1]` (height) so the body rests on the box; other heads float above it.

`capture --scenario` takes `none` (the default), a folder name (`greybox`), or a `scenario.json` path.

### GUI

- **Head:** a dropdown of `synthetic` and every `meshes/<head>/*.obj` (the folder is created on start if missing). Picking one swaps the head in the 3D view, preview and captures. **Reload** re-reads the head on screen from disk (so edits to its file show up) and rescans the folder for new files. Meshes there must already be in the world frame (convert other meshes with `import-mesh` first). A mesh that fails to load shows an error and the current head stays.
- **Scenario:** a dropdown of `none` and every `scenarios/<name>/scenario.json`. Picking one draws the room in the 3D view and adds it to the preview and captures. **Reload scenario** re-reads it from disk (e.g. after editing `scenario.json` or re-exporting the model) and rescans the folder. The room stays still in the 3D view. **Baby offset (mm)** and **Baby yaw (°)** move the baby inside it, and the cameras move with the baby (the rig is relative to the head); the baby can also be dragged by the gizmo at its feet. **Save scenario** writes the baby pose into the `scenario.json`, so `capture --scenario` reproduces it. Room objects (poles, box) can't be moved individually.
- **3D view:** the textured head with all its parts (eyes, eyebrows, eyelashes, tongue), plus one frustum per camera. The browser lights it for viewing only; captures and the preview use the Open3D renderer. The selected camera is highlighted; clicking a frustum selects it. The selected camera has a move gizmo: dragging it moves the camera's eye, the camera keeps looking at its target, and the fields and preview follow. A second gizmo at the centre of all eyes drags every camera together.
- **Camera panel:** a camera dropdown, then fields for the selected camera (eye, target, up, focal length, pixel size, resolution), with Add, Duplicate and Remove.
- **Move all cameras:** **Shift (mm)** and **Turn (°)**, then **Apply move**, move every camera rigidly: a turn about the vertical through the mean of the targets (a ring spins around its centre), then a shift. Each click applies the move again.
- **Presets:** ring and dome with the same layout parameters as the CLI; the generated cameras reuse the selected camera's resolution, lens and target. Applying a preset replaces the camera list.
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
  "scenario_source": null,
  "scenario": null,
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

`scenario_source` is the `scenario.json` path, or `null` without a scenario. `scenario` is the scenario spec as used for this capture (all `scenario.json` fields, baby pose included, even if moved in the GUI and not saved), or `null`. `K`, `R` and `t_mm` use the OpenCV convention. `R` and `t_mm` map world to camera (`x_cam = R @ x_world + t`). The camera axes are x right, y down, z forward.

## Errors

| Situation | Behaviour |
|---|---|
| Invalid rig (schema or geometry) | Every bad field is listed. The CLI exits with code 2; the GUI shows the message and keeps running. |
| `--out` exists and isn't empty | The CLI refuses and exits with code 2. |
| Unreadable mesh | Exit code 2 with the path and the reason. |
| Unreadable or invalid scenario (`scenario.json` field, missing model) | Exit code 2 with the reason; the GUI shows it and keeps the scenario on screen. |
| `import-mesh`: bad option (e.g. `--up` parallel to `--front`), output not `.obj`, no head above the neck cut, `--keep-body` without a cut | Exit code 2 with the reason. |
| Capture fails partway | No partial folder is left. Output is written to a temporary folder next to the target and renamed only when complete. |
| A camera sees none of the head | A warning (without a scenario); the image is still captured. That's a valid (bad) rig. |
