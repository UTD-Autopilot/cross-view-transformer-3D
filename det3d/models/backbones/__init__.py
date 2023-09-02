import importlib
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None
from .cvt.cvt_backbone import CrossViewTransformerBackbone

if found:
    from .scn import SpMiddleResNetFHD
else:
    print("No spconv, sparse convolution disabled!")

__all__ = [
    "CrossViewTransformerBackbone"
]

