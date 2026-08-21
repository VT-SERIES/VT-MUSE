# HDF5 data format

VT-MUSE consumes one or more trajectories from each HDF5 file. The canonical
layout is:

```text
observation/camera/<camera_name>/rgb
tactile/left_tactile/rgb_marker
tactile/right_tactile/rgb_marker
tactile/left_tactile/depth
tactile/right_tactile/depth
actor/action
```

`rgb` and `rgb_marker` may contain raw image arrays or encoded image bytes.
The two `depth` streams are required when Stage 2 uses
`--tactile_flow_target_mode depth_delta`. `actor/action` is optional when
action conditioning is disabled.

Place training and validation files in separate directories. For multitask
training, use filenames of the form `<task>__episode_<index>.hdf5`; otherwise
the parent directory name is used as the task name.

The loader also accepts the historical `left_gsmini`/`right_gsmini` tactile
group names and files containing multiple top-level episode groups.
