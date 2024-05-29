import json
import re
import torch

def load_json(json_path: str):
    f = open(json_path, encoding="utf-8")
    content = json.load(f)
    f.close()
    return content

def save_json(save_path: str, content):
    f = open(save_path, 'w')
    json.dump(content, f,ensure_ascii=False ,indent=4)
    f.close()

def process_labels(sequences):
        seqs = []
        for sequence in sequences:
            sequence = re.split(r"(?<=.)(?=[A-Z]|\[)", sequence)
            if '.-' in sequence:
                sequence.remove('.-')
            if '-.' in sequence:
                sequence.remove('-.')
            if '.' in sequence:
                sequence.remove('.')
            if '-' in sequence:
                sequence.remove('-')
            sequence = [seq.strip('.') for seq in sequence]
            sequence = [seq.strip('.-') for seq in sequence]
            seqs.append(''.join(sequence))
        return seqs

def re_process_peptide(sequence):
    # process predict peptides
    sequence = re.split(r"(?<=.)(?=[A-Z]|\[|\@)", sequence)
    return sequence

def calculate_acc_in_one_peptide(predict, label):
    predict = re_process_peptide(predict)
    label = re_process_peptide(label)
    cnt = 0
    for idx, aa in enumerate(label):
        if predict[idx] == aa:
            cnt += 1
    return cnt / len(label), cnt, len(label)

def calculate_pep_mass(sequence):
    vocabulary = {
            "G": 57.021464,
            "A": 71.037114,
            "S": 87.032028,
            "P": 97.052764,
            "V": 99.068414,
            "T": 101.047670,
            "C+57.021": 160.030649, # 103.009185 + 57.021464
            "L": 113.084064,
            "I": 113.084064,
            "N": 114.042927,
            "D": 115.026943,
            "Q": 128.058578,
            "K": 128.094963,
            "E": 129.042593,
            "M": 131.040485,
            "H": 137.058912,
            "F": 147.068414,
            "R": 156.101111,
            "Y": 163.063329,
            "W": 186.079313,
            # Amino acid modifications.
            "M+15.995": 147.035400,    # Met oxidation:   131.040485 + 15.994915
            "N+0.984": 115.026943,     # Asn deamidation: 114.042927 +  0.984016
            "Q+0.984": 129.042594,     # Gln deamidation: 128.058578 +  0.984016
            # N-terminal modifications.
            "+42.011": 42.010565,      # Acetylation
            "+43.006": 43.005814,      # Carbamylation
            "-17.027": -17.026549,     # NH3 loss
            "+43.006-17.027": 25.980265,      # Carbamylation and NH3 loss
        }
    
    sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
    mass = 0
    for aa in sequence:
        mass += vocabulary[aa]
    return mass