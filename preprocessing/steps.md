1. Extract participant ids in common in covariates, genotype, and expression files

```bash
python extract_common_participants.py
```

3. Subset the genotype files
    a. subset by participants
    b. filter out non-snps
    c. split by chromosome

```bash
./submit_subset_genotype_jobs.sh
```
| Check SNP filtering stats

4. Subset and merge the expression files 
    a. subset by participants 
    b. all cohorts in single expression file

```bash
python subset_and_merge_expression.py
```

5. Rename / duplicate entries in genotype file 

Per cohort, per chromosome:
    a. make single sample files --> update from participant_id to sample_id --> duplicate if needed
    b. merge sample files

```bash
python rename_genotype_mapping.py # generates participant_id -> sample_id mapping
./submit_update_genotype_ids_jobs.sh
```

6. Merge all cohorts per chromosome
--> Per chromosome, single genotype file (.bed, .bim, .fam)
```bash
./submit_merge_by_chromosome_jobs.sh
```

| Check A1 / A2 flips

| Check distribution of variants (snps) per chromosome

7. Make list of sample ids to keep ordering consistent
--> Taken from 'sample_id' column in covariates.tsv file

8. Make list of variant ids to keep ordering consistent
--> Saved during genotype/annotation matrix creation

9. Map and annotate variants
(Optional: make ref allele lookup tables to match annotations)
```bash
./create_ref_lookup_tables.sh 
```

--> Per gene Z matrices (variants x annotations)
```bash
./submit_annotate_jobs.sh
```

10. Make genotype matrices
--> Per gene G matrices (individuals x variants)
```bash
./submit_genotype_jobs.sh
```

11. Add additional annotation columns
```bash
python add_annotations.py
```

12. Train / Test Split
Example: Create holdout set (ROSMAP DLPFC)
```bash
sbatch tr
```