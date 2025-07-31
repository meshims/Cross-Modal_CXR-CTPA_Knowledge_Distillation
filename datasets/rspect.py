import pandas as pd
import torch.utils.data as data
import numpy as np
import torch

LABELS_CSV_DIRECTORY_PATH = "" # TODO add your path
NORMALIZED_DRR_DIRECTORY_PATH = "" # TODO add your path
NORMALIZED_CTPA_DIRECTORY_PATH = "" # TODO add your path
RSPECT_PE_COL = "negative_exam_for_pe"
RSPECT_CT_STUDY_COL = "StudyInstanceUID"
RSPECT_CT_SERIES_COL = "SeriesInstanceUID"



class RSPECTDataset(data.Dataset):
    def __init__(self, target=None):

        self.rspect_data  = pd.read_csv(LABELS_CSV_DIRECTORY_PATH+target)

    def __len__(self):

        return len(self.rspect_data)

    def __getitem__(self, idx):

        study = self.rspect_data.loc[idx,RSPECT_CT_STUDY_COL]
        series = self.rspect_data.loc[idx,RSPECT_CT_SERIES_COL]
        filename = study + "_" + series + ".npy"
        xray = torch.from_numpy(np.load(NORMALIZED_DRR_DIRECTORY_PATH + filename).astype(np.float32))
        ct = torch.from_numpy(np.load(NORMALIZED_CTPA_DIRECTORY_PATH+filename).astype(np.float32))

        negative_label = self.rspect_data.loc[idx, RSPECT_PE_COL]
        # the classifier that we compare results to predicts based on presence of PE so we need to adapt the RSPECT which gives opposite label (negative_exam_for_pe)
        label = 1 - negative_label

        return ct, xray, label,

    def get_start(self):
        return 0

    def set_start(self, start):
        self.start=start




