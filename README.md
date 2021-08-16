# CACS Forest App

## Description
CACS Forest uses a federated random forest to classify patients into their coronary artery calcification score value 
group, based on clinical parameters. A threshold can be specified to distinguish between values considered low risk
(close to 0) and elevated risk.

## Input
- `calc.csv` file containing the samples
- Columns: `kalk`, `birth_year`, `sex`, `height`, `weight`, `waist`, `bmi`, `chol`, `tri`, `hdl`, `ldl`, `hba`
- Sample files with synthetic data can be found in `sample_data/client*`
