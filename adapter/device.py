try:
    import torch_npu
    from torch_npu.npu import amp
    from manas.aisample.dataset.mfile import copy_from_local
    device_label = 'npu'
    backend_label = 'hccl'
    from manas.model import repository
    from manas.model.parameter_builder import ParameterBuilder
    from manas.dataset import PathConvert
except ImportError:
    amp = None
    torch_npu = None
    copy_from_local = None
    device_label = 'cuda'
    backend_label = 'nccl'
    repository = None
    ParameterBuilder = None
    PathConvert = None
