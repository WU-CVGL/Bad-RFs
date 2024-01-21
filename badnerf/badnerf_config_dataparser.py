"""
BAD-NeRF dataparser config.
"""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from badnerf.image_restoration_dataparser import ImageRestorationDataParserConfig


BadNerfDataparser = DataParserSpecification(config=ImageRestorationDataParserConfig())
