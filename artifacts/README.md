# Generated artifacts

Store run output under `artifacts/<experiment-id>/<run-id>/`. These files are ignored by Git. This directory currently contains no run output.

Every run should produce a manifest containing the code revision and dirty patch reference, full command/config, seed and generator version, input checksums, toolchain, hardware, timestamps, timing scope, and output checksums. Keep a small copy of the manifest and the result summary in `docs/experiments/`.

Before deleting local outputs, upload any irreplaceable raw results to a durable artifact store and record the URI and checksums in the experiment record. No artifact backend is configured yet. Local ignored files alone are not a permanent research record. Do not commit datasets, binaries, or W&B credentials.
