Place `geomonoplan.pt` here. The local release includes this file; distribute it
as a release asset rather than committing the large binary to Git.

The checkpoint contains the complete planning inference state, including the
YOLO26-Depth backbone, neck, calibrated depth head, residual adapters and feature
extractor. No separate YOLO weight download is needed. The runtime reconstructs
the encoder from the configuration embedded in this checkpoint and loads all
state strictly. It does not download pretrained weights.
