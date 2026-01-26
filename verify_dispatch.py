import torch
import os
import vllm._custom_ops as ops

def test_dispatch():
    num_tokens = 32
    hidden_size = 512
    group_size = 128
    device = "cuda"
    quant_dtype = torch.float8_e4m3fn
    
    input_tensor = torch.randn((num_tokens, hidden_size * 2), device=device, dtype=torch.bfloat16)

    print("--- Testing CUDA (default) ---")
    os.environ["VLLM_USE_TRITON_SILU_MUL_QUANT"] = "0"
    out_cuda, scales_cuda = ops.silu_and_mul_per_block_quant(input_tensor, group_size, quant_dtype)
    print(f"CUDA output shape: {out_cuda.shape}, scales shape: {scales_cuda.shape}")

    print("\n--- Testing Triton ---")
    os.environ["VLLM_USE_TRITON_SILU_MUL_QUANT"] = "1"
    out_triton, scales_triton = ops.silu_and_mul_per_block_quant(input_tensor, group_size, quant_dtype)
    print(f"Triton output shape: {out_triton.shape}, scales shape: {scales_triton.shape}")

    print("\n--- Testing Native (Eager) ---")
    os.environ["VLLM_USE_EAGER_SILU_QUANT"] = "1"
    out_native, scales_native = ops.silu_and_mul_per_block_quant(input_tensor, group_size, quant_dtype)
    print(f"Native output shape: {out_native.shape}, scales shape: {scales_native.shape}")

    # Compare results
    torch.testing.assert_close(scales_cuda, scales_native, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(scales_triton, scales_native, rtol=1e-5, atol=1e-5)
    print("\n✅ All scales match native implementation!")

    # Compare FP8 outputs (may have slight differences due to rounding, but should be close)
    diff_triton = (out_triton.to(torch.float32) - out_native.to(torch.float32)).abs().mean()
    diff_cuda = (out_cuda.to(torch.float32) - out_native.to(torch.float32)).abs().mean()
    print(f"Mean diff Triton vs Native: {diff_triton.item():.6f}")
    print(f"Mean diff CUDA vs Native: {diff_cuda.item():.6f}")

if __name__ == "__main__":
    test_dispatch()
