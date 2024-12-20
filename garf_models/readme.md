


## test01

- Intevo
- Lu177, melp
- no phantom (in Air)
- 2 point-sources + AA


## garf generation

    # osx = 22 min, 346 MB
    ./generate_garf_training_dataset.py -s intevo -r lu177 -d v3 -n 2e9 -t 4

    ./generate_garf_training_dataset.py -s intevo -r tc99m -d v1
    ./generate_garf_training_dataset.py -s intevo -r lu177 -d v1
    ./generate_garf_training_dataset.py -s intevo -r tc99m -d v2
    
    ./generate_garf_training_dataset.py -s nm670 -r tc99m -d v1
    ./generate_garf_training_dataset.py -s nm670 -r tc99m -d v2
    ./generate_garf_training_dataset.py -s nm670 -r lu177 -d v1

