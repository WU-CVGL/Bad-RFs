"""
BAD-NeRF dataparser config.
"""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from badnerf.image_restoration_dataparser import ImageRestorationDataParserConfig
from badnerf.deblur_nerf_dataparser import DeblurNerfDataParserConfig

BadNerfDataParser = DataParserSpecification(config=ImageRestorationDataParserConfig())
DeblurNerfDataParser = DataParserSpecification(config=DeblurNerfDataParserConfig())
