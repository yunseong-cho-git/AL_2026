# mpi_pesticide_score.py
from mpi4py import MPI
import pandas as pd
import glob
import time
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, MolFromSmarts
import math

RDLogger.DisableLog('rdApp.*')

# ── Score 함수 정의 ─────────────────────────────────────
PESTICIDE_PATTERNS = {
    'triazine':        MolFromSmarts('c1ncncn1'),
    'organophosphate': MolFromSmarts('P(=O)(O)O'),
    'carbamate':       MolFromSmarts('OC(=O)N'),
    'urea':            MolFromSmarts('NC(=O)N'),
    'sulfonamide':     MolFromSmarts('S(=O)(=O)N'),
    'halogen':         MolFromSmarts('[F,Cl,Br,I]'),
    'pyridine':        MolFromSmarts('c1ccncc1'),
    'pyrimidine':      MolFromSmarts('c1ccncn1'),
}
NEGATIVE_PATTERNS = {
    'alcohol':         MolFromSmarts('[OX2H]'),
    'aliphatic_amine': MolFromSmarts('[NX3;!$(NC=O);!$(Nc)]'),
}

def property_score(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 0
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    score = 0
    if 150 <= mw   <= 500: score += 1
    if 0   <= logp <= 5.0: score += 1
    if hbd <= 3:           score += 1
    if 1   <= hba  <= 12:  score += 1
    if rotb <= 12:         score += 1
    tpsa_score = math.exp(-0.5 * ((tpsa - 74) / 50) ** 2)
    return (score / 5 * 0.7) + (tpsa_score * 0.3)

def smarts_score(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return 0
    hits = sum(1 for pat in PESTICIDE_PATTERNS.values() if mol.HasSubstructMatch(pat))
    penalty = sum(1 for pat in NEGATIVE_PATTERNS.values() if mol.HasSubstructMatch(pat))
    return max(0, (hits / len(PESTICIDE_PATTERNS)) - (penalty * 0.2))

def pesticide_score(smi):
    return round(0.5 * property_score(smi) + 0.5 * smarts_score(smi), 4)

# ── MPI 설정 ────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
DONE = -1

zinc_files = sorted(glob.glob("./zinc_db/A*.txt"))[:10]

if rank == 0:
    start_time = time.time()
    print(f"[마스터] 총 파일 {len(zinc_files)}개, 워커 {size-1}명으로 시작")

    task_queue = list(range(len(zinc_files)))
    finished = 0
    all_results = []

    while finished < size - 1:
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=0, status=status)
        worker = status.Get_source()

        if result:
            all_results.extend(result)
            print(f"[마스터] 워커 {worker} → {len(result)}개 수신 (누적: {len(all_results)})")

        if task_queue:
            file_idx = task_queue.pop(0)
            comm.send(file_idx, dest=worker, tag=1)
        else:
            comm.send(DONE, dest=worker, tag=1)
            finished += 1

    # 결과 저장
    df = pd.DataFrame(all_results, columns=['smiles', 'score'])
    df = df.sort_values('score', ascending=False)
    df.to_csv('zinc_scored.csv', index=False)
    print(f"[마스터] 완료! 총 {len(all_results)}개 | 소요: {time.time()-start_time:.2f}초")

else:
    comm.send([], dest=0, tag=0)

    while True:
        file_idx = comm.recv(source=0, tag=1)
        if file_idx == DONE:
            break

        fname = zinc_files[file_idx]
        df = pd.read_csv(fname, sep='\t', usecols=['smiles'])
        results = []
        for smi in df['smiles']:
            if Chem.MolFromSmiles(smi):
                s = pesticide_score(smi)
                results.append((smi, s))
        print(f"  [워커 {rank}] {fname} → {len(results)}개 계산 완료")
        comm.send(results, dest=0, tag=0)