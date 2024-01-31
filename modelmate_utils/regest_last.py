from manas.model2 import repository
from manas.model2.metadata.metadataBuilder import MetaDataBuilder
from manas.model2.metadata.model_spec import ModelSpec
from manas.model2.metadata.ParameterBuilder import ParameterBuilder
import argparse


parser = argparse.ArgumentParser(description='transformer parameters')
parser.add_argument('--tags', type=str, default='init', help='tag')
args = parser.parse_args()

save_path = "/dls/app/model_reg/"
model_spec = ModelSpec.Builder().name('checkpoint_cpu_last').type("pytorch-pth").model_version("1.0.0").tags('%s'%args.tags).build()
metadata = MetaDataBuilder().model_spec(model_spec).build()
builder = ParameterBuilder().model(save_path).metadata(metadata).build()
repository.modelGenerate(builder, save_path)
result = repository.modelRegister(save_path)
