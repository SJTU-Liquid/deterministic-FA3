import torch
from functools import partial
from flash_attn_interface import flash_attn_func
from flash_attn.utils.benchmark import benchmark_backward

def check_bwd_deterministic(repeats=2, batch_size=2, seqlen=8192, nheads=16, headdim=128, causal=False, dtype=torch.float16, device='cuda'):
    """
    Checks if the backward pass of flash_attn_func is deterministic using benchmark_backward.

    Args:
        repeats (int): Number of times to run the backward pass.
        batch_size (int): Batch size.
        seqlen (int): Sequence length.
        nheads (int): Number of attention heads.
        headdim (int): Head dimension.
        causal (bool): Whether to use causal masking.
        dtype (torch.dtype): Data type for tensors.
        device (str): Device to run the test on.
    """
    print(f"Checking bwd determinism with: {repeats=}, {batch_size=}, {seqlen=}, {nheads=}, {headdim=}, {causal=}, {dtype=}")

    # Set seed for reproducibility of input data
    torch.manual_seed(0)

    # 在循环外创建一次张量，确保每次迭代的输入都相同
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    grad_o = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)

    q_grads, k_grads, v_grads = [], [], []

    for i in range(repeats):
        # 在每次迭代前重置梯度
        if q.grad is not None:
            q.grad.zero_()
        if k.grad is not None:
            k.grad.zero_()
        if v.grad is not None:
            v.grad.zero_()

        # benchmark_backward 会运行反向传播
        # 我们传入相同的 q, k, v, grad_o
        benchmark_backward(
            flash_attn_func, q, k, v, grad=grad_o, causal=causal, repeats=1, verbose=False, deterministic=True
        )

        # 存储梯度
        q_grads.append(q.grad.clone())
        k_grads.append(k.grad.clone())
        v_grads.append(v.grad.clone())
        print(f"Finished run {i+1}/{repeats}")

    # 比较梯度
    print("\nComparing gradients...")
    all_identical = True
    # 从第二次运行开始与第一次比较
    for i in range(1, repeats):
        q_diff = torch.max(torch.abs(q_grads[0] - q_grads[i])).item()
        k_diff = torch.max(torch.abs(k_grads[0] - k_grads[i])).item()
        v_diff = torch.max(torch.abs(v_grads[0] - v_grads[i])).item()

        print(f"Comparison between run 1 and run {i+1}:")
        print(f"  Max diff in q.grad: {q_diff}")
        print(f"  Max diff in k.grad: {k_diff}")
        print(f"  Max diff in v.grad: {v_diff}")

        # 使用一个足够小的阈值来处理浮点数精度问题
        if q_diff > 1e-7 or k_diff > 1e-7 or v_diff > 1e-7:
            all_identical = False

    if all_identical:
        print("\nConclusion: Backward pass appears to be deterministic.")
    else:
        print("\nConclusion: Backward pass is NOT deterministic.")

if __name__ == '__main__':
    # 为了在 CUDA 上获得确定性结果，还需要设置这些选项
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 在较新版本的 PyTorch 中，这个函数可以方便地设置确定性算法
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        print("torch.use_deterministic_algorithms not available in this PyTorch version.")

    check_bwd_deterministic()