"""
STATISTICAL SIGNIFICANCE COMPUTING

Given two sets of test results computed over the exact same
set of test images, this scripts computes the statistical significance 
of the results contained in the "candidate_res_path" compared to those
contained in the "base_res_path".

The statistical significance is computed using the Wilcoxon signed-rank test

"""


import json
import numpy as np

from scipy.stats import wilcoxon


def bootstrap_pvalue(new_met, old_met):
    
    diffs = new_met - old_met
    boot = []

    for _ in range(10000):
        sample = np.random.choice(diffs, size=len(diffs), replace=True)
        boot.append(sample.mean())

    boot = np.array(boot)

    return np.mean(boot <= 0)  # one-sided


def main():
    

    ds_list = ["DSA", "XCAD", "134XCA",  "30XCA",  "DRIVE",  "STARE"]
    
    target_fn = "results.json"
    
    base_res_path = "/path/to/test_results_folder/"
    candidate_res_path = "/path/to/alternative_test_results_folder/"


    base_mets = []
    cand_mets = []
    p_vals_ds = []
    for ds_name in ds_list:

        base_mets_ds = []
        cand_mets_ds = []
        
        base_file = json.load(open(f"{base_res_path}{ds_name}/{target_fn}"))
        candidate_file = json.load(open(f"{candidate_res_path}{ds_name}/{target_fn}"))

        if "INDIVIDUAL" in base_file.keys():
            idp_keys = list(base_file["INDIVIDUAL"].keys())
            base_pats = base_file["INDIVIDUAL"]
        else:
            idp_keys = list(base_file.keys())[:-7]
            base_pats = base_file

        if "INDIVIDUAL" in candidate_file.keys():
            candidate_pats = candidate_file["INDIVIDUAL"]
        else:
            candidate_pats = candidate_file


        for idpk in idp_keys:

            base_mets.append(base_pats[idpk])
            cand_mets.append(candidate_pats[idpk])

            base_mets_ds.append(base_pats[idpk])
            cand_mets_ds.append(candidate_pats[idpk])
            

        base_mets_ds = np.array(base_mets_ds)
        cand_mets_ds = np.array(cand_mets_ds)

        p_value_wil = wilcoxon(cand_mets_ds, base_mets_ds, alternative='greater').pvalue
        p_vals_ds.append(float(p_value_wil))


    print(p_vals_ds)

    base_mets = np.array(base_mets)
    cand_mets = np.array(cand_mets)
    p_value_wil = wilcoxon(cand_mets, base_mets, alternative='greater').pvalue

    print("---")
    print("P-VAL: ", p_value_wil)



if __name__ == "__main__":
    main()