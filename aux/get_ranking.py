"""
TEST PREDICTIONS RANKING BASED ON ITS DSC SCORE

Once an experiment is conducted this script allows to
rank the test results in three cathegories: bad, average and
good depending on its DSC percentile.

"""

import json
import numpy as np


def main():
    ds_list = ["134XCA",  "30XCA",  "DRIVE",  "DSA",  "STARE",  "XCAD"]

    target_fn = "results.json"
    base_res_path = "/path/to/test_results_folder/"

    for ds_name in ds_list:

        base_mets = []
        
        base_file = json.load(open(f"{base_res_path}{ds_name}/{target_fn}"))
        idp_keys = list(base_file.keys())[:-7]

        for idpk in idp_keys:
            base_mets.append(base_file[idpk])

        base_mets = np.array(base_mets)
    
        per_5 = np.percentile(base_mets, 5, method="nearest")
        per_50 = np.percentile(base_mets, 50, method="nearest")
        per_95 = np.percentile(base_mets, 95, method="nearest")

        print(f"\n{ds_name}")
        print(f"Q_5: {per_5}", idp_keys[np.argwhere(base_mets==per_5)[0][0]])
        print(f"Q_50: {per_50}", idp_keys[np.argwhere(base_mets==per_50)[0][0]])
        print(f"Q_95: {per_95}", idp_keys[np.argwhere(base_mets==per_95)[0][0]])

            


if __name__ == "__main__":
    main()