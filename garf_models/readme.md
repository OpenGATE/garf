


## test01

- Intevo
- Lu177, melp
- 2 point-sources + AA


## garf generation

    ./generate_garf_training_dataset.py -s intevo -r tc99m -d v1
    ./generate_garf_training_dataset.py -s intevo -r lu177 -d v1
    ./generate_garf_training_dataset.py -s intevo -r tc99m -d v2
    
    # WIP expected 12 min 200 MB
    ./generate_garf_training_dataset.py -s intevo -r lu177 -d v3 -n 2e9

    ./generate_garf_training_dataset.py -s nm670 -r tc99m -d v1
    ./generate_garf_training_dataset.py -s nm670 -r tc99m -d v2
    ./generate_garf_training_dataset.py -s nm670 -r lu177 -d v1


## Ideal UI ? 

options
- device: intevo, nm670
- collimator: lehr, etc
- rad: 177lu, Tc99m etc
- digit + windowing: ?????? 
- physics, cuts 
- rr=50 ?
- nb particle, root size
- Output: stat + root + simu.json + pth 

Naming: 
- intevo_melp_177lu_digit_2_v034.pth
- nm670_3p8_lehr_99tcm_digit_33_v034.pth
- nm670_5p8_lehr_99tcm_digit_33_v034.pth

  generate_training_dataset -s intevo -r 177lu --digit x 
