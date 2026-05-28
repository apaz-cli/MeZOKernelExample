# ZOKernelExample

An example implementation of a fused MeZO `layernorm(silu(inp @ W ± εz))` kernel.

The kernels are not meant to be used for production, but as examples of how to write ZO kernels.
They are not written for any particular hardware generation and do not use any hardware features,
but are written in CUTLASS terminology so they will be easy to port and optimize.

Compile with `./build.sh`, run with `./fused_example` and `./fused_zo_example`.

<br>
<p align="center">
  <img src="https://apaz.dev/blog/images/trolling.gif" alt="trolling">
</p>

