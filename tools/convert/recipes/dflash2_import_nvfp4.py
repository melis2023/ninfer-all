"""Import the DFlash2 module's NVFP4 weights verbatim from a ModelOpt source.

Unlike ``dflash2_nvfp4.py`` (which re-quantises a BF16/FP8 companion), this override
keeps the source's NVFP4 codes, block scales and global divisor. Use it when the
companion checkpoint is already ModelOpt NVFP4, e.g.
``maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal``.

The source's ``quantization_config.ignore`` list keeps the feature projection, the
selector hidden projection and the convolution kernel projections in BF16; those
parameters are left to the base recipe.
"""
from tools.convert.methods import import_encoded

# Parameters the ModelOpt checkpoint keeps outside NVFP4.
_KEPT_DIRECT = (
    'dflash2/feature_projection',
    'dflash2/candidate_selector/hidden_projection',
)

# The engine's native QKV input projection requires Q8_G32_FP16, so those three
# projections keep the base recipe's Q8 even though the source ships them as NVFP4.
_KEPT_Q8 = ('/attention/query', '/attention/key', '/attention/value')


def configure(model, recipe, sources):
    if 'dflash2' not in model.components:
        raise ValueError('DFlash2 NVFP4 import requires the DFlash2 component')
    for name, parameter in model.parameters.items():
        if not name.startswith('dflash2/') or not parameter.projection:
            continue
        if name in _KEPT_DIRECT or 'conv' in name or '/attention/context_' in name:
            continue
        if any(name.endswith(suffix) for suffix in _KEPT_Q8):
            continue
        recipe.assign(name, format='nvfp4', method=import_encoded)
