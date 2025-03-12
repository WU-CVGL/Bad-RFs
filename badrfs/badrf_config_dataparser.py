"""
BAD-RF dataparser configs.
"""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from badrfs.deblur_nerf_dataparser import DeblurNerfDataParserConfig

DeblurNerfDataParser = DataParserSpecification(config=DeblurNerfDataParserConfig())
