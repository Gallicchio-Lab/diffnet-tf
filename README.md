# diffnet-tf
Extended DiffNet implementation with TensorFlow 2

## Standard case: Many Ligands, One Receptor

```
python diffnet-tf.py < SAMPL8-manyligs-onercpt.csv
```

## Extended case: Many Ligands, Many Receptors

```
python diffnet-tf.py < SAMPL8-manyligs-manyrcpts-alldata.csv
```

## Binding Selectivity Analysis with Swapping Free Energies

```
python diffnet-tf-hop.py < SAMPL8-selectivity.csv
```
