"""Name mappings between dataset class names and the text-description corpus.

Most class names match the corpus keys directly; the exceptions are mapped here.
"""

# EuroSAT class names (from split_zhou_EuroSAT.json) that differ from the corpus keys
# after case/spacing normalization. The other six EuroSAT classes
# ("Annual Crop Land", "Forest", "Highway or Road", "Pasture Land", "Permanent Crop Land",
# "River") match a corpus key once normalized.
EUROSAT_CLASSNAME_MAP = {
    "Herbaceous Vegetation Land": "brushland or shrubland",
    "Industrial Buildings": "industrial buildings or commercial buildings",
    "Residential Buildings": "residential buildings or homes or apartments",
    "Sea or Lake": "lake or sea",
}

# The dataset is registered as "fgvc" but the corpus file/template uses "fgvc_aircraft".
DATASET_TO_CORPUS = {"fgvc": "fgvc_aircraft"}
