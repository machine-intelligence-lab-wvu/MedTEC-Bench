"""Tiny NIfTI-1 reader that bypasses nibabel.

Workaround for a silent crash in nibabel 5.4.0 on this Windows env when
loading certain AMOS .nii.gz volumes (uint16 labels at 768x768x90). The raw
gzip + numpy parse below handles the same file format and matches what
nib.load + np.asarray(dataobj) would have produced.

Only the slice of NIfTI-1 needed for AMOS image/label volumes is parsed
(scl_slope/scl_inter ignored, RAS orientation assumed). Sufficient for the
mask-area + slice-render pipeline in scripts/select_ct_probes.py.
"""
import gzip
import struct
from pathlib import Path

import numpy as np

DATATYPE_TO_NP = {
    2:   np.uint8,    4:   np.int16,    8:   np.int32,    16:  np.float32,
    64:  np.float64,  256: np.int8,     512: np.uint16,   768: np.uint32,
}


def load_volume(path) -> np.ndarray:
    """Return the volume array (axis order matches nibabel's get_fdata,
    i.e. (X, Y, Z) Fortran-order)."""
    p = Path(path)
    if str(p).endswith(".gz"):
        with gzip.open(p, "rb") as f:
            raw = f.read()
    else:
        raw = p.read_bytes()
    dim = struct.unpack("<8h", raw[40:56])
    datatype = struct.unpack("<h", raw[70:72])[0]
    vox_offset = int(struct.unpack("<f", raw[108:112])[0])
    n_dim = dim[0]
    shape = tuple(int(d) for d in dim[1:1 + n_dim])
    dtype = DATATYPE_TO_NP.get(datatype, np.uint8)
    n_voxels = int(np.prod(shape))
    buf = raw[vox_offset:vox_offset + n_voxels * np.dtype(dtype).itemsize]
    return np.frombuffer(buf, dtype=dtype).reshape(shape, order="F").copy()


if __name__ == "__main__":
    import sys
    arr = load_volume(sys.argv[1])
    print(f"shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()}")
