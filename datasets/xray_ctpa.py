import pandas as pd
import torch.utils.data as data
import numpy as np
import torch

CT_ACCESSION_COL = 'CT_Accession_number'
XRAY_ACCESSION_COL='cxr_Accession_number'
PE_COL = "PE"
RSPECT_PE_COL = "negative_exam_for_pe"


LABELS_CSV_DIRECTORY_PATH = "" # TODO add your path
NORMALIZED_CXR_DIRECTORY_PATH = "" # TODO add your path
NORMALIZED_CTPA_DIRECTORY_PATH = "" # TODO add your path




class XrayCTPADataset(data.Dataset):
    def __init__(self, target=None):
        self.data = pd.read_csv(LABELS_CSV_DIRECTORY_PATH + target)
        self.cts=NORMALIZED_CTPA_DIRECTORY_PATH
        self.xrays = NORMALIZED_CXR_DIRECTORY_PATH
        self.start=0

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        ct_accession = self.data.loc[idx, CT_ACCESSION_COL]
        cxr_accession = self.data.loc[idx, XRAY_ACCESSION_COL]
        ct = torch.from_numpy(np.load(self.cts + str(ct_accession) +'.npy').astype(np.float32))
        xray = torch.from_numpy(np.load(self.xrays + str(cxr_accession) + '.npy').astype(np.float32))
        label = self.data.loc[idx, PE_COL]

        return ct, xray, label

    def get_start(self):
        return 0

    def set_start(self, start):
        self.start=start




