def get_dolos_localisation_dataset_paths(dataset):
    paths = dict(
        fake_path=f'datasets/dolos_data/celebahq/fake/{dataset}/images/test',
        masks_path=f'datasets/dolos_data/celebahq/fake/{dataset}/masks/test',
        key=dataset
    )
    return paths

def get_dolos_detection_dataset_paths(dataset):
    paths = dict(
        real_path=f'datasets/dolos_data/celebahq/fake/{dataset}/images/test',
        fake_path=f'datasets/dolos_data/celebahq/real/{dataset}/test',
        masks_path=f'datasets/dolos_data/celebahq/fake/{dataset}/masks/test',
        key=dataset
    ),
    return paths

def get_autosplice_localisation_dataset_paths(compression):
    paths = dict(
        fake_path=f'datasets/AutoSplice/Forged_JPEG{compression}',
        masks_path=f'datasets/AutoSplice/Mask',
        key=f'autosplice_jpeg{compression}'
    )
    return paths

def get_drct_2m_localisation_dataset_paths(dataset):
    paths = dict(
        fake_path=f'/root/autodl-tmp/AIGC_data/DRCT-2M/{dataset}/val2017',
        masks_path=f'/root/autodl-tmp/AIGC_data/DRCT-2M/{dataset}/masks/val2017',
        key=dataset,
    )
    return paths

def get_drct_2m_detection_dataset_paths(dataset):
    paths = dict(
        real_path='/root/autodl-tmp/AIGC_data/MSCOCO/val2017',
        fake_path=f'/root/autodl-tmp/AIGC_data/DRCT-2M/{dataset}/val2017',
        masks_path=f'/root/autodl-tmp/AIGC_data/DRCT-2M/{dataset}/masks/val2017',
        key=dataset
    )
    return paths

# LOCALISATION_DATASET_PATHS = [
#     get_dolos_localisation_dataset_paths('pluralistic'),
#     get_dolos_localisation_dataset_paths('lama'),
#     get_dolos_localisation_dataset_paths('repaint-p2-9k'),
#     get_dolos_localisation_dataset_paths('ldm'),
#     # TO BE PUBLISHED
#     # get_dolos_localisation_dataset_paths('ldm_clean'),
#     # get_dolos_localisation_dataset_paths('ldm_real'),

#     get_autosplice_localisation_dataset_paths("75"),
#     get_autosplice_localisation_dataset_paths("90"),
#     get_autosplice_localisation_dataset_paths("100"),
# ]

# DETECTION_DATASET_PATHS = [
#     get_dolos_detection_dataset_paths('pluralistic'),
#     get_dolos_detection_dataset_paths('lama'),
#     get_dolos_detection_dataset_paths('repaint-p2-9k'),
#     get_dolos_detection_dataset_paths('ldm'),
#     # TO BE PUBLISHED
#     # get_dolos_detection_dataset_paths('ldm_clean'),
#     # get_dolos_detection_dataset_paths('ldm_real'),
# ]

# our localisation dataset paths
LOCALISATION_DATASET_PATHS = [
    get_drct_2m_localisation_dataset_paths("stable-diffusion-2-inpainting"),
]

# our detection dataset paths
DETECTION_DATASET_PATHS = [
    get_drct_2m_detection_dataset_paths("stable-diffusion-2-inpainting"),
]