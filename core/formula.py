# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:16:57 2023

@author: DELL
"""


import numpy as np
import pandas as pd
from tqdm import tqdm

from core.pycdk import IsotopeFromString, IsotopeSimilarity, getFormulaExactMass
from core.subformula.mass_spectrum import MassSpectrum


adducts_dict = np.load('data/adducts_dict.npy', allow_pickle=True).item()
formula_database = pd.read_csv('data/AutoMsFormulaDB-v1.0.csv')

def calc_parent_mass(s, precursor_type = '[M+H]+'):
    precursor_mz = s.get('precursor_mz')
    if precursor_mz is None:
        return None
    else:
        return (precursor_mz - adducts_dict[precursor_type]['correction_mass']) / adducts_dict[precursor_type]['mass_multiplier']


def calc_isotopic_score(s, parent_mass, formula):
    adduct_mz = s.get('precursor_mz') - parent_mass
    if 'isotopic_pattern' not in dir(s):
        return None
    if s.isotopic_pattern is None:
        return None
    isotope_mz = s.isotopic_pattern.mz
    isotope_mz = isotope_mz - adduct_mz
    isotope_intensity = s.isotopic_pattern.intensities
    isotope_intensity = isotope_intensity / max(isotope_intensity)
    isotope_pattern = np.vstack((isotope_mz, isotope_intensity)).T
    try:
        isotope_ref = IsotopeFromString(formula, minI=0.001)
        isotope_score = IsotopeSimilarity(isotope_pattern, isotope_ref, 10)
    except:
        isotope_score = 0
    return isotope_score


def calc_exact_mass_score(s, parent_mass, formula):
    exact_mass_score = {}
    formula_mass = getFormulaExactMass(formula)
    diff_mass = abs(formula_mass - parent_mass)
    exact_mass_score = 1 - 10000 * diff_mass / parent_mass
    return exact_mass_score


def calc_fragmentation_tree_score(s, parent_mass, formula):
    formula_mass = getFormulaExactMass(formula)
    formula_mass_ppm = np.max(np.abs(formula_mass - parent_mass) / parent_mass  * 10**6)
    fragmentation = MassSpectrum(dict(zip(s.mz, s.intensities)), s.get('precursor_mz'), formula_mass_ppm)
    annotations = fragmentation.compute_with_candidate_formula([formula], fragmentation.product_scoring_function)
    scores = [fragmentation.product_scoring_function(s) for s in annotations]
    fragmentation_tree_score = scores[0]
    return fragmentation_tree_score


def retrieve_formula_list(s, precursor_type_list = ['[M+H]+'], mz_tol = 0.01):
    parent_mass_list = [calc_parent_mass(s, i) for i in precursor_type_list]
    formula_list = []
    for i, mass in enumerate(parent_mass_list):
        lb = np.searchsorted(formula_database['Exact mass'].values, mass - mz_tol)
        rb = np.searchsorted(formula_database['Exact mass'].values, mass + mz_tol)
        for j in range(lb, rb):
            formula_list.append([formula_database['Formula'].values[j],
                                 formula_database['Exact mass'].values[j], 
                                 precursor_type_list[i],
                                 parent_mass_list[i]])
    if len(formula_list) == 0:
        return None
    else:
        formula_list = pd.DataFrame(formula_list, columns = ['Formula', 'Exact mass', 'Precursor type', 'Parent mass'])
        return formula_list


def get_best_formula(s, precursor_type_list = ['[M+H]+'], mz_tol = 0.01):
    formula_list = retrieve_formula_list(s, precursor_type_list, mz_tol)
    if formula_list is None:
        return None
    isotope_score, exact_mass_score, fragmentation_tree_score = [], [], []
    for i in formula_list.index:
        f = formula_list.loc[i, 'Formula']
        parent_mass = formula_list.loc[i, 'Parent mass']
        isotope_score_ = calc_isotopic_score(s, parent_mass, f)
        exact_mass_score_ = calc_exact_mass_score(s, parent_mass, f)
        try:
            fragmentation_tree_score_ = calc_fragmentation_tree_score(s, parent_mass, f)
        except:
            fragmentation_tree_score_ = 0
        if isotope_score_ is None:
            isotope_score_ = 0
        isotope_score.append(isotope_score_)
        exact_mass_score.append(exact_mass_score_)
        fragmentation_tree_score.append(fragmentation_tree_score_)
    formula_list['isotope_score'] = isotope_score
    formula_list['exact_mass_score'] = exact_mass_score
    formula_list['fragmentation_tree_score'] = fragmentation_tree_score
    formula_list['consesus_score'] = formula_list['isotope_score'] + formula_list['exact_mass_score'] + formula_list['fragmentation_tree_score']
    return formula_list


def assign_formula(feature_table, precursor_type_list = ['[M+H]+'], mz_tol = 0.01):
    if 'Tandem_MS' not in list(feature_table.columns):
        return feature_table
    formula = []
    for i in tqdm(feature_table.index):
        s = feature_table.loc[i, 'Tandem_MS']
        if s is None:
            continue
        formula_list = get_best_formula(s, precursor_type_list, mz_tol)
        if formula_list is None:
            continue
        wh = np.argmax(formula_list['consesus_score'])
        formula = formula_list.loc[wh, 'Formula']
        precursor_type = formula_list.loc[wh, 'Precursor type']
        formula_score = formula_list.loc[wh, 'consesus_score']
        s.set('formula', formula)
        s.set('adduct', precursor_type)
        feature_table.loc[i, 'Tandem_MS'] = s
        feature_table.loc[i, 'Formula'] = formula
        feature_table.loc[i, 'Formula_Score'] = formula_score
        feature_table.loc[i, 'Precursor_Type'] = precursor_type
    return feature_table


if __name__ == '__main__':
    
    from matchms.importing import load_from_mgf
    
    s = [s for s in load_from_mgf("D:/DeepMASS2_GUI/example/minimum_example.mgf")][1]
    
    
    
    
    