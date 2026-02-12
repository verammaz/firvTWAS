from collections import defaultdict
import os
import pandas as pd
import re
import sys

chrom = sys.argv[1]

if __name__ == "__main__":
    genotype_dir = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/genotypes"
    annotation_dir = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/annotations"


    ## add index column back to annotation matrices
    ref_dir = "/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/reference"
    print(f"Processing chromosome {chrom}")
        
    n_processed, n_total = 0, (len(os.listdir(os.path.join(annotation_dir, f"chr{chrom}")))-2)/2
    for file in os.listdir(os.path.join(annotation_dir, f"chr{chrom}")):
        if not file.endswith("_annotations.tsv.gz"):
            continue
        n_processed += 1
        print(f"Processing {file} ({n_processed}/{n_total})")
        gene = re.sub(r"_annotations\.tsv\.gz$", "", file)
        
        
        anno_df = pd.read_csv(os.path.join(annotation_dir, f"chr{chrom}", file), sep = "\t")
        if 'variant_id' in anno_df.columns:
            # remove variant_id column
            anno_df.drop(columns=['variant_id'], inplace=True)
            
        geno_df = pd.read_csv(os.path.join(genotype_dir, f"chr{chrom}", f"{gene}_genotypes.tsv.gz"), sep = "\t", index_col = "IID")
        variant_ids = geno_df.columns
        pos_to_alleles = dict()
        for variant_id in variant_ids:
            chr, pos_alleles = variant_id.split(":") 
            pos, counted, other = pos_alleles.split("_")
            pos_to_alleles[pos] = (counted, other)

        ref_table = pd.read_csv(os.path.join(ref_dir, f"chr{chrom}_ref.tsv.gz"), sep = "\t",
                    compression="gzip",
                    names=["chr", "pos", "ref"],
                    dtype={"chr": str, "pos": int, "ref": str})
        anno_df2 = pd.merge(anno_df, ref_table, on = ['pos'], how = 'left')
        anno_varids = []
        for _, row in anno_df2.iterrows():
            pos = row['pos']
            ref = row['ref']
            alt = ''
            counted_other = pos_to_alleles[str(int(pos))]
            if ref == counted_other[0]:
                alt = counted_other[1]
            else:
                alt = counted_other[0]
            anno_varids.append(f"{int(row['chr_x'])}:{row['pos']}_{ref}_{alt}")
        assert len(anno_varids) == len(anno_df), f"Mismatch: {len(anno_varids)} genotype variants vs {len(anno_df)} annotations"
        assert len(geno_df.columns) == len(anno_varids), f"Mismatch: {len(geno_df.columns)} genotype variants vs {len(anno_varids)} annotations"
        anno_df2['variant_id'] = anno_varids
        anno_df2.drop(columns=['ref', 'chr_y'], inplace=True)
        anno_df2.rename(columns={'chr_x': 'chr'}, inplace=True)
        anno_df2 = anno_df2.reset_index(drop=True)
        anno_df2.set_index('variant_id', inplace=True)
        anno_df.index.name = 'variant_id'
        assert anno_df.values.shape == anno_df2.values.shape, f"Mismatch: {anno_df.values.shape} annotation matrix vs {anno_df2.values.shape} annotation matrix"
        assert (anno_df.values == anno_df2.values).all(), "Annotation matrices do not match"
        anno_df.to_csv(os.path.join(annotation_dir, f"chr{chrom}", file), sep = "\t", index=True, compression="gzip")
