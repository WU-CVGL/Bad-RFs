"""
BAD-NeRF dataparser config.
"""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from badnerf.deblur_nerf_dataparser import DeblurNerfDataParserConfig

DeblurNerfDataParser = DataParserSpecification(config=DeblurNerfDataParserConfig())
