from os import getenv

# Utility / Build settings
LOCALITY = getenv('LOCALITY', "USEN")
CLEAN = getenv('CLEAN', False)
# Important directories
DATA_DIR = "./data"
TEMPLATES_DIR = "./templates"
RIPPED_DATA = "./fehdata"
ASSETS_DIR = "%s/assets" % RIPPED_DATA
JSON_DIR = "%s/json" % RIPPED_DATA
# JSON data directories
COMMON_DATA = "%s/files/assets/Common" % JSON_DIR
TRANSLATION_DATA = "%s/files/assets/%s/Message/Data" % (JSON_DIR, LOCALITY)
MAP_DATA = "%s/SRPGMap" % COMMON_DATA
SKILL_DATA = "%s/SRPG/Skill" % COMMON_DATA
# Databases
INDICES_CACHE_NAME = "%s/indices" % DATA_DIR
MAP_CACHE_NAME = "%s/maps" % DATA_DIR
SKILL_CACHE_NAME = "%s/skills" % DATA_DIR
TRANSLATION_CACHE_NAME = "%s/%s" % (DATA_DIR, LOCALITY)
# Miscellaneous locations/files
MAP_DIRECTORY = '%s/Maps' % ASSETS_DIR
FONT_FILE = "%s/Fire_Emblem_Heroes_Font.ttf" % ASSETS_DIR
# Constants
MAP_RATIO_L = 540/675
MAP_RATIO_N = 540/720
MAP_DIMENSIONS_L = (8, 10)
MAP_DIMENSIONS_N = (6, 8)
