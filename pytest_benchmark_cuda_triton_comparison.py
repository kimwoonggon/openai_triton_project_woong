import math
import random
import pytest
import torch
import torch.nn.functional as F
from typing import Callable, Dict, Tuple, Union, Optional
from transformer_engine.pytorch.attention import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)
from einops import rearrange
from triton_rotary_main import RotaryEmb
from utils.triton_rotary_embedding import *
import triton.testing as testing
import os

CUDA_FUSED = True
prefix = "TritonVsCUDA_"


def get_tol(dtype: torch.dtype) -> Dict:
    if dtype == torch.bfloat16:
        return dict(atol=1e-2, rtol=1e-2)
    elif dtype == torch.float16:
        return dict(atol=1e-3, rtol=1e-3)
    return dict(atol=1e-5, rtol=1.3e-6)


def _overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    return output.sum() * 2

# Gradient is a full tensor
def _non_overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    t = torch.ones_like(output)
    return torch.sum(output * t)

    
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_length", [2048, 4096])
@pytest.mark.parametrize("hidden_size", [128, 256, 512])
@pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
@pytest.mark.parametrize("margin", [0, 10])
@pytest.mark.parametrize("tensor_format", ["sbhd", "bshd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_fused_rope(
    dtype: torch.dtype,
    seq_length: int,
    hidden_size: int,
    rotary_percent: float,
    margin: int,
    tensor_format: str,
    loss_func: Callable,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    t = torch.rand(
        (seq_length - margin, batch_size, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    
    if tensor_format == "bshd":
        t = t.transpose(0, 1).contiguous()
    # triton kernel input shape : (batch_size, seq_length, nheads, headdim)
    if tensor_format == "bshd":
        t2 = t.clone()
    elif tensor_format == "sbhd":
        t2 = t.clone()
        t2 = t2.transpose(0,1).contiguous()
    else:
        raise ValueError("Invalid tensor format")
    
    t.requires_grad = True
    t2.requires_grad=True
    
    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(seq_length)
    
    # Triton embedding
    
    rotary_pos_emb_triton = RotaryPositionEmbedding_Triton(hidden_size, rotary_percent)
    emb_triton = rotary_pos_emb_triton(seq_length)
    
    triton_cos_ = emb_triton.clone()
    triton_sin_ = emb_triton.clone()
    
    
    # Triton fused kernel output
    
    triton_out_fused = RotaryEmb(
        t2, triton_cos_, triton_sin_, seqlen_offsets=0,
    )
    if tensor_format == "sbhd":
        triton_out_fused = triton_out_fused.transpose(0,1).contiguous()
    loss_triton_output = loss_func(triton_out_fused)
    loss_triton_output.backward()
    grad_triton_fused = t2.grad.detach().clone()
    if tensor_format == "sbhd":
        grad_triton_fused = grad_triton_fused.transpose(0,1).contiguous()
    t2.grad = None
    
    # unfused
    output_unfused = apply_rotary_pos_emb(
        t, emb, tensor_format=tensor_format, fused=False
    )
    
    loss_unfused = loss_func(output_unfused)
    loss_unfused.backward()
    grad_unfused = t.grad.detach().clone()
    t.grad = None

    # fused
    output_fused = apply_rotary_pos_emb(
        t,
        emb,
        tensor_format=tensor_format,
        fused=True,
    )
    loss_fused = loss_func(output_fused)
    loss_fused.backward()
    grad_fused = t.grad.detach().clone()
    t.grad = None

    # CUDA FUSED vs TORCH UNFUSED
    torch.testing.assert_close(output_fused, output_unfused, **get_tol(dtype))
    print("stage 1 CUDA FUSED vs TORCH UNFUSED success")
    
    # CUDA FUSED vs TRITON FUSED
    torch.testing.assert_close(output_fused, triton_out_fused, **get_tol(dtype))
    print("stage 2 CUDA FUSED vs TRITON FUSED success")
    
    # TORCH UNFUSED VS TRITON FUSED
    torch.testing.assert_close(output_unfused, triton_out_fused, **get_tol(dtype))
    print('stage 3 TORCH UNFUSED VS TRITON FUSED success')
    
    # CUDA FUSED GRAD VS TORCH UNFUSED GRAD
    torch.testing.assert_close(grad_fused, grad_unfused, **get_tol(dtype))
    print('stage 4 CUDA FUSED GRAD VS TORCH UNFUSED GRAD success')
    
    # TRITON FUSED GRAD VS CUDA FUSED GRAD
    torch.testing.assert_close(grad_triton_fused, grad_fused, **get_tol(dtype))
    print('stage 5 TRITON FUSED GRAD VS CUDA FUSED GRAD success')
    
    # TRITON FUSED GRAD VS TORCH UNFUSED GRAD
    torch.testing.assert_close(grad_triton_fused, grad_unfused, **get_tol(dtype))
    print("stage 6 TRITON FUSED GRAD VS TORCH UNFUSED GRAD success")
    
    assert output_fused.is_contiguous()
    
    
@testing.perf_report(
    [
        testing.Benchmark(
            x_names=["seq_length"],
            x_vals=[256*i for i in range(1,24)],
            x_log=False,
            line_arg="backend",
            line_vals=["triton","cuda"],
            line_names=["triton","cuda"],
            ylabel="milliseconds",
            plot_name="RoPE performance",
            args={"batch_size":8}
        )
    ]
)
def benchmark(batch_size,seq_length, backend):
    head_num=64
    hidden_size=256
    device = torch.device("cuda:0")
    dtype = torch.float32
    tri_input = torch.rand(
        (batch_size, seq_length, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    #Triton embedding
    rotary_pos_emb_triton = RotaryPositionEmbedding_Triton(hidden_size)
    emb_triton = rotary_pos_emb_triton(seq_length)
    
    triton_cos_ = emb_triton.clone()
    triton_sin_ = emb_triton.clone()
    cuda_input = tri_input.clone()
    
    rotary_pos_emb = RotaryPositionEmbedding(hidden_size)
    emb = rotary_pos_emb(seq_length)
    
    if backend == "triton":
        return testing.do_bench(lambda: RotaryEmb(tri_input, triton_cos_, triton_sin_, 0))
    else:
        return testing.do_bench(lambda: apply_rotary_pos_emb(cuda_input, emb, tensor_format="bshd", fused=CUDA_FUSED))
        #return testing.do_bench(lambda : 1+1)

if CUDA_FUSED:
    os.makedirs(prefix+"rope_benchmark_seq_length_fused", exist_ok=True)
    benchmark.run(print_data=True, show_plots=True, save_path=prefix+"rope_benchmark_seq_length_fused")
else:
    os.makedirs(prefix+"rope_benchmark_seq_length_unfused", exist_ok=True)
    benchmark.run(print_data=True, show_plots=True, save_path=prefix+"rope_benchmark_seq_length_unfused")

@testing.perf_report(
    [
        testing.Benchmark(
            x_names=["hidden_size"],
            x_vals=[2**i for i in range(4,10)],
            x_log=False,
            line_arg="backend",
            line_vals=["triton","cuda"],
            line_names=["triton","cuda"],
            ylabel="milliseconds",
            plot_name="RoPE performance",
            args={"batch_size":8}
        )
    ]
)
def benchmark(batch_size,hidden_size, backend):
    head_num=64
    seq_length=1024
    device = torch.device("cuda:0")
    dtype = torch.float32
    tri_input = torch.rand(
        (batch_size, seq_length, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    #Triton embedding
    rotary_pos_emb_triton = RotaryPositionEmbedding_Triton(hidden_size)
    emb_triton = rotary_pos_emb_triton(seq_length)
    
    triton_cos_ = emb_triton.clone()
    triton_sin_ = emb_triton.clone()
    cuda_input = tri_input.clone()
    
    rotary_pos_emb = RotaryPositionEmbedding(hidden_size)
    emb = rotary_pos_emb(seq_length)
    
    if backend == "triton":
        return testing.do_bench(lambda: RotaryEmb(tri_input, triton_cos_, triton_sin_, 0))
    else:
        return testing.do_bench(lambda: apply_rotary_pos_emb(cuda_input, emb, tensor_format="bshd", fused=CUDA_FUSED))
if CUDA_FUSED:
    os.makedirs(prefix+"rope_benchmark_hidden_size_fused", exist_ok=True)
    benchmark.run(print_data=True, show_plots=True, save_path=prefix+"rope_benchmark_hidden_size_fused")
else:
    os.makedirs(prefix+"rope_benchmark_hidden_size_unfused", exist_ok=True)
    benchmark.run(print_data=True, show_plots=True, save_path=prefix+"rope_benchmark_hidden_size_unfused")

@testing.perf_report(
    [
        testing.Benchmark(
            x_names=["head_num"],
            x_vals=[2**i for i in range(3,9)],
            x_log=False,
            line_arg="backend",
            line_vals=["triton","cuda"],
            line_names=["triton","cuda"],
            ylabel="milliseconds",
            plot_name="headNum performance",
            args={"batch_size":8}
        )
    ]
)
def benchmark(batch_size,head_num, backend):
    seq_length=1024
    hidden_size=256
    device = torch.device("cuda:0")
    dtype = torch.float32
    tri_input = torch.rand(
        (batch_size, seq_length, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    #Triton embedding
    rotary_pos_emb_triton = RotaryPositionEmbedding_Triton(hidden_size)
    emb_triton = rotary_pos_emb_triton(seq_length)
    
    triton_cos_ = emb_triton.clone()
    triton_sin_ = emb_triton.clone()
    cuda_input = tri_input.clone()
    
    rotary_pos_emb = RotaryPositionEmbedding(hidden_size)
    emb = rotary_pos_emb(seq_length)
    
    if backend == "triton":
        return testing.do_bench(lambda: RotaryEmb(tri_input, triton_cos_, triton_sin_, 0))
    else:
        return testing.do_bench(lambda: apply_rotary_pos_emb(cuda_input, emb, tensor_format="bshd", fused=CUDA_FUSED))
        #return testing.do_bench(lambda : 1+1)
if CUDA_FUSED:
    os.makedirs(prefix+"rope_benchmark_head_num_fused", exist_ok=True)
    benchmark.run(print_data=True, show_plots=True, save_path=prefix+"rope_benchmark_head_num_fused")
else:
    os.makedirs(prefix+"rope_benchmark_head_num_unfused", exist_ok=True)
    benchmark.run(print_data=True, show_plots=True, save_path=prefix+"rope_benchmark_head_num_unfused")

@testing.perf_report(
    [
        testing.Benchmark(
            x_names=["batch_size"],
            x_vals=[i for i in range(1,2**7,2)],
            x_log=False,
            line_arg="backend",
            line_vals=["triton","cuda"],
            line_names=["triton","cuda"],
            ylabel="milliseconds",
            plot_name="batchSize performance",
            args={"seq_length":512}
        )
    ]
)
def benchmark(seq_length,batch_size, backend):
    head_num=64
    hidden_size=256
    device = torch.device("cuda:0")
    dtype = torch.float32
    tri_input = torch.rand(
        (batch_size, seq_length, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    #Triton embedding
    rotary_pos_emb_triton = RotaryPositionEmbedding_Triton(hidden_size)
    emb_triton = rotary_pos_emb_triton(seq_length)
    
    triton_cos_ = emb_triton.clone()
    triton_sin_ = emb_triton.clone()
    cuda_input = tri_input.clone()
    
    rotary_pos_emb = RotaryPositionEmbedding(hidden_size)
    emb = rotary_pos_emb(seq_length)
    
    if backend == "triton":
        return testing.do_bench(lambda: RotaryEmb(tri_input, triton_cos_, triton_sin_, 0))
    else:
        return testing.do_bench(lambda: apply_rotary_pos_emb(cuda_input, emb, tensor_format="bshd", fused=CUDA_FUSED))
        #return testing.do_bench(lambda : 1+1)
if CUDA_FUSED:
    os.makedirs(prefix+"rope_benchmark_batch_size_fused", exist_ok=True)
    benchmark.run(print_data=True, show_plots=True, save_path=prefix+"rope_benchmark_batch_size_fused")
else:
    os.makedirs(prefix+"rope_benchmark_batch_size_unfused", exist_ok=True)
    benchmark.run(print_data=True, show_plots=True, save_path=prefix+"rope_benchmark_batch_size_unfused")

