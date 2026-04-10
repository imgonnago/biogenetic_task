import numpy as np
import pandas as pd

def data():
    GM = np.load('./biogenetic/X_GM.npy')
    age = np.load('./biogenetic/C_age.npy')
    education = np.load('./biogenetic/C_edu.npy')
    sex = np.load('./biogenetic/C_sex.npy')
    MMSE = np.load('./biogenetic/S_MMSE.npy')
    SNP = np.load('./biogenetic/X_SNP 1.npy')
    dis = np.load('./biogenetic/y_dis.npy')

    return GM, age, education, sex, MMSE, SNP, dis

GM, age, education, sex, MMSE, SNP, dis = data()

list = {'GM': GM, 'age': age, 'education': education, 'sex': sex, 'MMSE': MMSE, 'SNP': SNP, 'dis': dis}

for i, name in enumerate(list):
    print('=' * 50 )
    print(f"Variable: {name}")
    print(list[name].shape)
    print(list[name][0:10])
    if list[name].dtype in [int, float]:
        print(f"Mean: {np.mean(list[name])}\nStd: {np.std(list[name])}\nMin: {np.min(list[name])}\nMax: {np.max(list[name])}")
    else:
        continue
    print("=" * 50)

print(f"males: {np.sum(sex == 'M')}")
print(f"females: {np.sum(sex == 'F')}")

dis = np.load('./biogenetic/y_dis.npy')
print(dis.shape)
print(np.unique(dis, return_counts=True))
