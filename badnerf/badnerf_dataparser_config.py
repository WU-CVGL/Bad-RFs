"""
BAD-NeRF dataparser config.
"""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from badnerf.data.badnerf_dataparser import BadNerfDataParserConfig


BadNerfDataparser = DataParserSpecification(config=BadNerfDataParserConfig())
