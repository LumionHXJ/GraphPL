# Adapted from MUSE (ICLR24)
import os
import pickle
import torch
from torch.utils.data import Dataset
from .tokenizer import eICUTokenizer
from .constants import modalities, demographics_size

def read_text(file):
    table = dict()
    with open(file) as f:
        for idx, l in enumerate(f.readlines()):
            table[l.strip()] = idx
    return table

def category_vector(codelist, table):
    vector = torch.zeros(len(table)).float()
    for code in codelist:
        vector[table[code]] = 1
    return vector

def onehot_vector(tok, vocab_size):
    vector = torch.zeros((vocab_size, )).float()
    vector[tok] = 1
    return vector

class eICUData:

    def __init__(
            self,
            icu_id,
            admission_id,
            patient_id,
            icu_duration,
            hospital_id,
            mortality,
            readmission,
            age,
            gender,
            ethnicity,
    ):
        self.icu_id = icu_id  # str
        self.admission_id = admission_id  # str
        self.patient_id = patient_id  # str
        self.icu_duration = icu_duration  # int
        self.hospital_id = hospital_id  # int
        self.mortality = mortality  # bool, end of icu stay mortality
        self.readmission = readmission  # bool, 15-day icu readmission
        self.age = age  # int
        self.gender = gender  # str
        self.ethnicity = ethnicity  # str

        # list of tuples (timestamp in min (int), type (str), list of codes (str))
        self.diagnosis = []
        self.treatment = []
        self.medication = []

        # (list of types (str), list of codes (str))
        self.trajectory = []

        # labs
        # (timestamp in min (int), list of (item_id, value))
        self.lab = []
        # numpy array
        self.labvectors = None

        # apacheapsvar
        # numpy array
        self.apacheapsvar = None

    def __repr__(self):
        return f"ICU ID-{self.icu_id} ({self.icu_duration} min): " \
               f"mortality-{self.mortality}, " \
               f"readmission-{self.readmission}"

class eICUDataset(Dataset):
    modalities_names = modalities
    def __init__(self, data_dir):
        super().__init__()
        with open(os.path.join(data_dir, "icu_stay_dict.pkl"), "rb") as f:
            self.all_hosp_adm_dict = pickle.load(f)
        self.tokenizer = eICUTokenizer()
        self.diagnosis_table = read_text(os.path.join(data_dir, "diagnosis_code.txt"))
        self.treatment_table = read_text(os.path.join(data_dir, "treatment_code.txt"))
        self.medication_table = read_text(os.path.join(data_dir, "medication_code.txt"))
    
    def __getitem__(self, icu_id):
        icu_stay = self.all_hosp_adm_dict[str(icu_id)]

        age = icu_stay.age
        gender = icu_stay.gender
        ethnicity = icu_stay.ethnicity
        labvectors = torch.FloatTensor(icu_stay.labvectors)
        apacheapsvar = torch.FloatTensor(icu_stay.apacheapsvar)

        age, gender, ethnicity = self.tokenizer(
            age, gender, ethnicity
        )
        age = onehot_vector(age, demographics_size['age'])
        gender = onehot_vector(gender, demographics_size['gender'])
        ethnicity = onehot_vector(ethnicity, demographics_size['ethnicity'])

        return_dict = dict()
        return_dict["demographics"] = torch.cat([age, gender, ethnicity]) # (20, )
        return_dict["lab"] = labvectors.max(dim=0)[0] # (158,), flatten time series
        return_dict["apache"] = apacheapsvar # (36, )
        return_dict['diagnosis'] = category_vector(icu_stay.diagnosis, self.diagnosis_table) 
        return_dict['treatment'] = category_vector(icu_stay.treatment, self.treatment_table)
        return_dict['medication'] = category_vector(icu_stay.medication, self.medication_table)

        labels = torch.tensor([icu_stay.mortality, ]).float()
        return return_dict, labels