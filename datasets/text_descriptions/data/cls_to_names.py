# Prompt template for each dataset, used to turn a class name into a
# natural-language prompt for the CLIP text encoder.
CUSTOM_TEMPLATES = {
    "oxford_pets": "a photo of a {} from OxfordPets",
    "oxford_flowers": "a photo of a {}, a type of flower",
    "fgvc_aircraft": "a photo of a {} from Fine-Grained Visual Classification of Aircraft",
    "dtd": "a photo of a {}, which is a type of texture, or a type of pattern",
    "eurosat": "satellite photo of the {}",
    "stanford_cars": "a photo of a {}",
    "food101": "a photo of a {} from Food101",
    "sun397": "a photo of a {}",
    "caltech101": "a photo of a {}",
    "ucf101": "a photo of a person doing {}",
    "imagenet": "a photo of a {}",
}
