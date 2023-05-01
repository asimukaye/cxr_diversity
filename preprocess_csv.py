import pandas as pd



def filter_views(df_metadata):
    VIEW_MAP = {
        'AP': 'frontal',
        'PA': 'frontal',
        'LATERAL': 'lateral',
        'LL': 'lateral',
        'LPO': 'other',
        'RAO': 'other',
        'RPO': 'other',
        'LAO': 'other',
        # the below are overwritten in some instances by manual review
        'AP AXIAL': 'other',
        'XTABLE LATERAL': 'other',
        'AP LLD': 'other',
        'PA LLD': 'other',
        'L5 S1': 'other',
        'SWIMMERS': 'other',
        'AP RLD': 'other',
        'PA RLD': 'other',
    }

    PPSD_MAP = {
        'CHEST, LATERAL': 'lateral',
        'CHEST, PA': 'frontal',
        # manually checked 100 records, below is always frontal
        'CHEST, PORTABLE': 'frontal',
        'CHEST, PA X-WISE': 'frontal',
        'CHEST, AP (GRID)': 'frontal',
        'CHEST LAT': 'lateral',
        'CHEST PA': 'frontal',
        'CHEST, AP NON-GRID': 'frontal',
        'CHEST AP NON GRID': 'frontal',
        'CHEST PA X-WISE': 'frontal',
        'CHEST AP GRID': 'frontal',
        'CHEST, PORTABLE X-WISE': 'other',
        # below have < 25 samples each
        'CHEST PORT': 'frontal',
        'CHEST PORT X-WISE': 'frontal',
        # manually classified below
        'SHOULDER': 'other',
        'CHEST, PEDI (4-10 YRS)': 'other',
        'LOWER RIBS': 'other',
        'CHEST, DECUB.': 'other',
        'ABDOMEN, PORTABLE': 'other',
        'UPPER RIBS': 'frontal',
        'STERNUM, LATERAL': 'lateral',
        'KNEE, AP/OBL': 'other',
        'STERNUM, PA/OBL.': 'other',
        'CLAVICLE/ AC JOINTS': 'other',
        'ABDOMEN,GENERAL': 'other',
        'LOWER RIB': 'other',
        'SCOLIOSIS AP': 'frontal'
    }

    df_metadata['view'] = df_metadata['ViewPosition'].map(VIEW_MAP)
    # print(df_metadata.shape)

    good_view = ['frontal', 'lateral']
    idxUpdate = ~df_metadata['view'].isin(good_view)
    # print(idxUpdate.sum())
    c = 'PerformedProcedureStepDescription'
    idx = (df_metadata[c].notnull()) & idxUpdate

    # print(idx.sum())

    df_metadata.loc[idx, 'view'] = df_metadata.loc[idx, c].map(PPSD_MAP)
    # print(df_metadata.shape)
    DICOM_TO_VIEW = {
        '2164992c-f4abb30a-7aaaf4f4-383cab47-4e3eb1c8': ['PA', 'frontal'],
        '5e6881e2-ff4254e0-b99f0c2f-8964482a-031364db': ['LL', 'lateral'],
        'fcdf7a30-3236b74e-65b97587-cdd4cfde-63cd1de0': ['PA', 'frontal'],
        'fb074ec1-6715839c-84fa75e6-adc3f026-448b1481': ['PA', 'frontal'],
        'dfb8080a-8506e43e-840d9d58-0f738f41-82c120b0': ['PA', 'frontal'],
        '4b32608b-c2ead7c4-1fe5565f-42f7ab80-9dad30de': ['LL', 'lateral'],
        '53663e89-8f9ca9bb-df1bf434-8d6b1283-2b612609': ['LL', 'lateral'],
        # below are AP, but incorrectly in View Position
        '8672a4e7-366801a0-26cf2395-9344335c-aac8d728': ['AP', 'frontal'],
        '9800b28e-3ff3b417-18473be2-1a66131d-aca88488': ['AP', 'frontal'],
        '598cfe48-33a8643e-843e27e2-5dd584e7-3cd5f1c0': ['AP', 'frontal']
    }

    # we manually reviewed a few DICOMs to keep them in
    for dcm, row in DICOM_TO_VIEW.items():
        view = row[1]
        idx = df_metadata['dicom_id'] == dcm
        if idx.any():
            df_metadata.loc[idx, 'view'] = view
    # print(df_metadata.shape)

    good_view_df = df_metadata[df_metadata['ViewPosition']=='PA']
    # print(good_view_df.shape)

    return good_view_df

root = '/home/asim.ukaye/physionet.org/files/mimic-cxr-jpg/2.0.0/'
root_2 = '/home/asim.ukaye/ml_proj/mimic_cxr_pa_resized/'

df_split = pd.read_csv(root+ 'mimic-cxr-2.0.0-split.csv.gz')
df_metadata = pd.read_csv(root+ 'mimic-cxr-2.0.0-metadata.csv.gz', header=0, sep=',')

df_filtered = filter_views(df_metadata)
df_merged = df_split.merge(df_filtered.drop(['study_id', 'subject_id'], axis=1),
                   on='dicom_id', how='inner')
df_final = df_merged[df_merged['subject_id']<16000000]

df_final.to_csv(root+ 'mimic-cxr-2.0.0-dataloader.csv')

df_final.to_csv(root_2+ 'mimic-cxr-2.0.0-dataloader.csv')
