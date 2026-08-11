# Exact GKE FedPer private heads

Run ID: `14339380272482304688`

Model digest: `42642e4cc839c09dfe8519511aa7cf7cdf5ca7350a8dd376e118ee31a6a74bbf`

The run directory contains six client directories. Each client has 30 private
classifier-head checkpoints and one metadata document:

```text
exact-gke-run-14339380272482304688/
├── 1-1/
├── 3-1/
├── 34-1/
├── 36-1/
├── 39-1/
└── 9-1/
    ├── head-0001.npz
    ├── ...
    ├── head-0030.npz
    └── metadata.json
```

For final inference, merge the matching client's `head-0030.npz` with:

`../../runs/fedper-20260805T162143974378Z-270b7ffe84/checkpoints/best_model.npz`

The earlier heads support round-by-round audit and rollback. Do not treat a
head from one client as the personalized model for another client.
