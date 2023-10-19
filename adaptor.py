try:
    import torch_npu
    from torch_npu.npu import amp
except ImportError:
    amp = None