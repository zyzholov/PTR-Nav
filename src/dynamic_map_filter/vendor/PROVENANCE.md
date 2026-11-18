# Pinned upstream code

DUFOMap reference application: https://github.com/KTH-RPL/dufomap
commit `9e239ddd5995136e14f5212f33382a6ebc59e518`.

The unmodified UFOMap core headers are from its pinned submodule:
https://github.com/UnknownFreeOccupied/ufomap
commit `fcb20f445d257280c039c07ab756c9d5b37ebd78`.
The original BSD notices and licenses are retained. No datasets or build outputs
are vendored. `liblzf/lzf.h` is the unmodified declaration header from Ubuntu
22.04 liblzf-dev 3.6-3, with its copyright file. PCD I/O is not used by this adapter.

The ROS-independent adapter limits threads/range/memory and exposes the actual
upstream `seenFree` classification. It does not reimplement or rename a custom
filter as the published method. Static-map classification is never a free-space
authorization for navigation. New hits remain present in the live collision input.
