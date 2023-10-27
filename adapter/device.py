try:
    import torch_npu
    from torch_npu.npu import amp
    from manas.aisample.dataset.mfile import copy_from_local
    device_label = 'npu'
    backend_label = 'hccl'
    from manas.model2 import repository
    from manas.model2.metadata.metadataBuilder import MetaDataBuilder
    from manas.model2.metadata.model_spec import ModelSpec
    from manas.model2.metadata.ParameterBuilder import ParameterBuilder
except ImportError:
    amp = None
    torch_npu = None
    copy_from_local = None
    device_label = 'cuda'
    backend_label = 'nccl'
    repository = None
    ParameterBuilder = None
    PathConvert = None
    repository = None
    MetaDataBuilder = None
    ModelSpec = None
    ParameterBuilder = None
