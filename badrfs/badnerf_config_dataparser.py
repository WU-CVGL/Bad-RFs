"""
BAD-NeRF dataparser config.
"""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from badrfs.image_restoration_dataparser import ImageRestorationDataParserConfig
from badrfs.deblur_nerf_dataparser import DeblurNerfDataParserConfig

ImageRestoreDataParser = DataParserSpecification(config=ImageRestorationDataParserConfig())
DeblurNerfDataParser = DataParserSpecification(config=DeblurNerfDataParserConfig())
