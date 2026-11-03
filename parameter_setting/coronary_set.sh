 #!/usr/bin/bash 

python3 frangi_estim_coronary.py --ds_name DSA --ds_pctg 0.1
python3 linear_estim_coronary.py --ds_name DSA --device 5 --ds_pctg 0.1