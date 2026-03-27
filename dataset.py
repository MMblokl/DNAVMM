"""
BIOSCAN-5M Dataset Loader

Author: Zahra Gharaee (https://github.com/zahrag)
License: MIT License
Description:
    This custom dataset loader provides structured access to the BIOSCAN-5M dataset,
    which includes millions of annotated insect images and associated metadata
    for machine learning and biodiversity research. It supports multiple image resolutions
    (e.g., cropped and original), and predefined splits for training, evaluation,
    and pretraining. The loader integrates with the Hugging Face `datasets` library
    to simplify data access and preparation.

Usage

To load the dataset from dataset.py:

    from datasets import load_dataset
    
    ds = load_dataset("dataset.py", name="cropped_256_eval", split="validation", trust_remote_code=True)

"""


import os
import csv
import datasets
import json


_CITATION = """\n----Citation:\n@inproceedings{gharaee2024bioscan5m,
  title={{BIOSCAN-5M}: A Multimodal Dataset for Insect Biodiversity},
  booktitle={Advances in Neural Information Processing Systems},
  author={Zahra Gharaee and Scott C. Lowe and ZeMing Gong and Pablo Millan Arias
          and Nicholas Pellegrino and Austin T. Wang and Joakim Bruslund Haurum
          and Iuliia Zarubiieva and Lila Kari and Dirk Steinke and Graham W. Taylor
          and Paul Fieguth and Angel X. Chang},
  editor={A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
  pages={36285--36313},
  publisher={Curran Associates, Inc.},
  year={2024},
  volume={37},
  url={https://proceedings.neurips.cc/paper_files/paper/2024/file/3fdbb472813041c9ecef04c20c2b1e5a-Paper-Datasets_and_Benchmarks_Track.pdf}
}\n"""

_DESCRIPTION = (
    "\n----Description:\n'BIOSCAN-5M' is a comprehensive multimodal dataset containing data for over 5 million insect specimens.\n"
    "Released in 2024, this dataset substantially enhances existing image-based biological resources by incorporating:\n"
    "- Taxonomic labels\n- Raw nucleotide barcode sequences \n- Assigned barcode index numbers\n- Geographical information\n"
    "- Specimen size information\n\n"
    "-------------- Dataset Feature Descriptions --------------\n"
    "1- processid: A unique number assigned by BOLD (International Barcode of Life Consortium).\n"
    "2- sampleid: A unique identifier given by the collector.\n"
    "3- taxon: Bio.info: Most specific taxonomy rank.\n"
    "4- phylum: Bio.info: Taxonomic classification label at phylum rank.\n"
    "5- class: Bio.info: Taxonomic classification label at class rank.\n"
    "6- order: Bio.info: Taxonomic classification label at order rank.\n"
    "7- family: Bio.info: Taxonomic classification label at family rank.\n"
    "8- subfamily: Bio.info: Taxonomic classification label at subfamily rank.\n"
    "9- genus: Bio.info: Taxonomic classification label at genus rank.\n"
    "10- species: Bio.info: Taxonomic classification label at species rank.\n"
    "11- dna_bin: Bio.info: Barcode Index Number (BIN).\n"
    "12- dna_barcode: Bio.info: Nucleotide barcode sequence.\n"
    "13- country: Geo.info: Country associated with the site of collection.\n"
    "14- province_state: Geo.info: Province/state associated with the site of collection.\n"
    "15- coord-lat: Geo.info: Latitude (WGS 84; decimal degrees) of the collection site.\n"
    "16- coord-lon: Geo.info: Longitude (WGS 84; decimal degrees) of the collection site.\n"
    "17- image_measurement_value: Size.info: Number of pixels occupied by the organism.\n"
    "18- area_fraction: Size.info: Fraction of the original image the cropped image comprises.\n"
    "19- scale_factor: Size.info: Ratio of the cropped image to the cropped_256 image.\n"
    "20- inferred_ranks: An integer indicating at which taxonomic ranks the label is inferred.\n"
    "21- split: Split set (partition) the sample belongs to.\n"
    "22- index_bioscan_1M_insect: An index to locate organism in BIOSCAN-1M Insect metadata.\n"
    "23- chunk: The packaging subdirectory name (or empty string) for this image.\n"
)

license = "\n----License:\nCC BY 3.0: Creative Commons Attribution 3.0 Unported (https://creativecommons.org/licenses/by/3.0/)\n"

SUPPORTED_FORMATS = {"csv": "csv", "jsonld": "jsonld"}

SUPPORTED_PACKAGES = {
        "original_256": "BIOSCAN_5M_original_256.zip",
        "original_256_pretrain": "BIOSCAN_5M_original_256_pretrain.zip",
        "original_256_train": "BIOSCAN_5M_original_256_train.zip",
        "original_256_eval": "BIOSCAN_5M_original_256_eval.zip",
        "cropped_256": "BIOSCAN_5M_cropped_256.zip",
        "cropped_256_pretrain": "BIOSCAN_5M_cropped_256_pretrain.zip",
        "cropped_256_train": "BIOSCAN_5M_cropped_256_train.zip",
        "cropped_256_eval": "BIOSCAN_5M_cropped_256_eval.zip",
    }


def safe_cast(value, cast_type):
    try:
        return cast_type(value) if value else None
    except ValueError:
        return None

def extract_info_from_filename(package_name):
    """
    Extract imgtype and split_name using string ops.
    Assumes package_name format: BIOSCAN_5M_<imgtype>[_<split_name>].zip

    """

    if package_name not in SUPPORTED_PACKAGES.values():
        raise ValueError(
            f"Unsupported package: {package_name}\n"
            f"Supported packages are:\n  - " + "\n  - ".join(sorted(SUPPORTED_PACKAGES.values()))
        )

    # Remove prefix and suffix
    core = package_name.replace("BIOSCAN_5M_", "").replace(".zip", "")

    parts = core.split("_")

    if len(parts) == 2:
        imgtype = "_".join(parts)
        data_split = "full"
    elif len(parts) == 3:
        imgtype = "_".join(parts[:2])
        data_split = parts[2]
    else:
        imgtype, data_split = None, None  # Unexpected format

    return imgtype, data_split


class BIOSCAN5MConfig(datasets.BuilderConfig):
    def __init__(self, metadata_format="csv", package_name="BIOSCAN_5M_cropped_256.zip", **kwargs):
        super().__init__(**kwargs)
        self.metadata_format = metadata_format
        self.package_name = package_name


class BIOSCAN5M(datasets.GeneratorBasedBuilder):
    """Custom dataset loader for BIOSCAN-5M (images + metadata)."""

    BUILDER_CONFIGS = [
        BIOSCAN5MConfig(
            name="cropped_256_eval",
            version=datasets.Version("0.0.0"),
            description="Cropped_256 images for evaluation splits.",
            metadata_format=SUPPORTED_FORMATS["csv"],
            package_name=SUPPORTED_PACKAGES["cropped_256_eval"],
        ),
        BIOSCAN5MConfig(
            name="cropped_256_train",
            version=datasets.Version("0.0.0"),
            description="Cropped_256 images for training split.",
            metadata_format=SUPPORTED_FORMATS["csv"],
            package_name=SUPPORTED_PACKAGES["cropped_256_train"],
        ),
        BIOSCAN5MConfig(
            name="cropped_256_pretrain",
            version=datasets.Version("0.0.0"),
            description="Cropped images for pretraining split.",
            metadata_format=SUPPORTED_FORMATS["csv"],
            package_name=SUPPORTED_PACKAGES["cropped_256_pretrain"],
        ),
        BIOSCAN5MConfig(
            name="cropped_256",
            version=datasets.Version("0.0.0"),
            description="Cropped_256 images for full splits.",
            metadata_format=SUPPORTED_FORMATS["csv"],
            package_name=SUPPORTED_PACKAGES["cropped_256"],
        ),
        BIOSCAN5MConfig(
            name="original_256_eval",
            version=datasets.Version("0.0.0"),
            description="Original_256 images for evaluation splits.",
            metadata_format=SUPPORTED_FORMATS["csv"],
            package_name=SUPPORTED_PACKAGES["original_256_eval"],
        ),
        BIOSCAN5MConfig(
            name="original_256_train",
            version=datasets.Version("0.0.0"),
            description="Original_256 images for training split.",
            metadata_format=SUPPORTED_FORMATS["csv"],
            package_name=SUPPORTED_PACKAGES["original_256_train"],
        ),
        BIOSCAN5MConfig(
            name="original_256_pretrain",
            version=datasets.Version("0.0.0"),
            description="Original images for pretraining split.",
            metadata_format=SUPPORTED_FORMATS["csv"],
            package_name=SUPPORTED_PACKAGES["original_256_pretrain"],
        ),
        BIOSCAN5MConfig(
            name="original_256",
            version=datasets.Version("0.0.0"),
            description="Original_256 images for full splits.",
            metadata_format=SUPPORTED_FORMATS["csv"],
            package_name=SUPPORTED_PACKAGES["original_256"],
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "image": datasets.Image(),
                "processid": datasets.Value("string"),
                "sampleid": datasets.Value("string"),
                "taxon": datasets.Value("string"),
                "phylum": datasets.Value("string"),
                "class": datasets.Value("string"),
                "order": datasets.Value("string"),
                "family": datasets.Value("string"),
                "subfamily": datasets.Value("string"),
                "genus": datasets.Value("string"),
                "species": datasets.Value("string"),
                "dna_bin": datasets.Value("string"),
                "dna_barcode": datasets.Value("string"),
                "country": datasets.Value("string"),
                "province_state": datasets.Value("string"),
                "coord-lat": datasets.Value("float"),
                "coord-lon": datasets.Value("float"),
                "image_measurement_value": datasets.Value("int64"),
                "area_fraction": datasets.Value("float"),
                "scale_factor": datasets.Value("float"),
                "inferred_ranks": datasets.Value("int32"),
                "split": datasets.Value("string"),
                "index_bioscan_1M_insect": datasets.Value("int32"),
                "chunk": datasets.Value("string"),
            }),
            supervised_keys=None,
            homepage="https://huggingface.co/datasets/bioscan-ml/BIOSCAN-5M",
            citation=_CITATION,
            license=license,
        )

    def _split_generators(self, dl_manager, **kwargs ):
        """Custom dataset split generator"""

        metadata_format = self.config.metadata_format
        package_name = self.config.package_name

        imgtype, data_split = extract_info_from_filename(package_name)

        # Download metadata
        metadata_url = "https://huggingface.co/datasets/bioscan-ml/BIOSCAN-5M/resolve/main/BIOSCAN_5M_Insect_Dataset_metadata_MultiTypes.zip"
        metadata_archive = dl_manager.download_and_extract(metadata_url)
        metadata_file = os.path.join(
            metadata_archive,
            f"bioscan5m/metadata/{metadata_format}/BIOSCAN_5M_Insect_Dataset_metadata.{metadata_format}"
        )

        # Download image archives
        image_url = f"https://huggingface.co/datasets/bioscan-ml/BIOSCAN-5M/resolve/main/{package_name}"
        image_archives = dl_manager.download_and_extract([image_url])
        image_dirs = [archive for archive in image_archives]

        # Define all available splits
        eval_splits = [
            "val", "test", "val_unseen", "test_unseen", "key_unseen", "other_heldout"
        ]
        splits = ["pretrain", "train"] + eval_splits

        hf_splits = {
            "train": datasets.Split.TRAIN,
            "val": datasets.Split.VALIDATION,
            "test": datasets.Split.TEST,
        }

        if data_split == "full":  # All partitions
            return [
                datasets.SplitGenerator(
                    name=hf_splits.get(split, split),
                    gen_kwargs={
                        "metadata_path": metadata_file,
                        "image_dirs": image_dirs,
                        "split": split,
                        "imgtype": imgtype,
                    },
                )
                for split in splits
            ]

        elif data_split == "eval":  # Evaluation partitions
            return [
                datasets.SplitGenerator(
                    name=hf_splits.get(split, split),
                    gen_kwargs={
                        "metadata_path": metadata_file,
                        "image_dirs": image_dirs,
                        "split": split,
                        "imgtype": imgtype,
                    },
                )
                for split in eval_splits
            ]

        else:   # train and pretrain partitions
            return [
                datasets.SplitGenerator(
                    name=hf_splits.get(data_split, data_split),
                    gen_kwargs={
                        "metadata_path": metadata_file,
                        "image_dirs": image_dirs,
                        "split": data_split,
                        "imgtype": imgtype,
                    },
                )
            ]

    def _generate_examples(self, metadata_path, image_dirs, split, imgtype):

        if metadata_path.endswith(".csv"):
            with open(metadata_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    if row["split"] != split:
                        continue  # Skip others and keep the chosen split samples

                    processid = row["processid"]
                    chunk = row.get("chunk", "").strip() if row.get("chunk") else ""

                    # Construct expected relative path
                    if chunk == "":
                        rel_path = f"bioscan5m/images/{imgtype}/{split}/{processid}.jpg"
                    else:
                        rel_path = f"bioscan5m/images/{imgtype}/{split}/{chunk}/{processid}.jpg"

                    # Search for the image file inside extracted image_dirs
                    image_path = None
                    for image_dir in image_dirs:
                        potential_path = os.path.join(image_dir, rel_path)
                        if os.path.exists(potential_path):
                            image_path = potential_path
                            break  # Image found; end search

                    if image_path is None:
                        print(f" ---- Image NOT Found! ---- \n{potential_path}")
                        continue

                    yield idx, {
                        "image": image_path,
                        "processid": row["processid"],
                        "sampleid": row["sampleid"],
                        "taxon": row["taxon"],
                        "phylum": row["phylum"] or None,
                        "class": row["class"] or None,
                        "order": row["order"] or None,
                        "family": row["family"] or None,
                        "subfamily": row["subfamily"] or None,
                        "genus": row["genus"] or None,
                        "species": row["species"] or None,
                        "dna_bin": row["dna_bin"] or None,
                        "dna_barcode": row["dna_barcode"],
                        "country": row["country"] or None,
                        "province_state": row["province_state"] or None,
                        "coord-lat": safe_cast(row["coord-lat"], float),
                        "coord-lon": safe_cast(row["coord-lon"], float),
                        "image_measurement_value": safe_cast(row["image_measurement_value"], float),
                        "area_fraction": safe_cast(row["area_fraction"], float),
                        "scale_factor": safe_cast(row["scale_factor"], float),
                        "inferred_ranks": safe_cast(row["inferred_ranks"], int),
                        "split": row["split"],
                        "index_bioscan_1M_insect": safe_cast(row["index_bioscan_1M_insect"], float),
                        "chunk": row["chunk"] or None,
                    }
        elif metadata_path.endswith(".jsonld"):
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
                for idx, row in enumerate(metadata):
                    if row["split"] != split:
                        continue  # Skip others and keep the chosen split samples

                    processid = row["processid"]
                    chunk = row.get("chunk", "").strip() if row.get("chunk") else ""

                    # Construct expected relative path
                    if chunk == "":
                        rel_path = f"bioscan5m/images/{imgtype}/{split}/{processid}.jpg"
                    else:
                        rel_path = f"bioscan5m/images/{imgtype}/{split}/{chunk}/{processid}.jpg"

                    # Search for the image file inside extracted image_dirs
                    image_path = None
                    for image_dir in image_dirs:
                        potential_path = os.path.join(image_dir, rel_path)
                        if os.path.exists(potential_path):
                            image_path = potential_path
                            break  # Image found; end search

                    if image_path is None:
                        print(f" ---- Image NOT Found! ---- \n{potential_path}")
                        continue

                    yield idx, {
                        "image": image_path,
                        "processid": row["processid"],
                        "sampleid": row["sampleid"],
                        "taxon": row["taxon"],
                        "phylum": row["phylum"] or None,
                        "class": row["class"] or None,
                        "order": row["order"] or None,
                        "family": row["family"] or None,
                        "subfamily": row["subfamily"] or None,
                        "genus": row["genus"] or None,
                        "species": row["species"] or None,
                        "dna_bin": row["dna_bin"] or None,
                        "dna_barcode": row["dna_barcode"],
                        "country": row["country"] or None,
                        "province_state": row["province_state"] or None,
                        "coord-lat": safe_cast(row["coord-lat"], float),
                        "coord-lon": safe_cast(row["coord-lon"], float),
                        "image_measurement_value": safe_cast(row["image_measurement_value"], float),
                        "area_fraction": safe_cast(row["area_fraction"], float),
                        "scale_factor": safe_cast(row["scale_factor"], float),
                        "inferred_ranks": safe_cast(row["inferred_ranks"], int),
                        "split": row["split"],
                        "index_bioscan_1M_insect": safe_cast(row["index_bioscan_1M_insect"], float),
                        "chunk": row["chunk"] or None,
                    }
        else:
            raise ValueError(
                f"Unsupported format: {os.path.splitext(metadata_path.lower())[1]}\n"
                f"Supported formats are:\n  - " + "\n  - ".join(sorted(SUPPORTED_FORMATS.values()))
            )




