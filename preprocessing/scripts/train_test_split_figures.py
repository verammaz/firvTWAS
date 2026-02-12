import os
import sys
import pandas as pd
import torch
import shutil

COHORT = "ROSMAP"
TISSUE = "DLPFC"

if '--cohort' in sys.argv:
    COHORT = sys.argv[sys.argv.index('--cohort') + 1]
if '--tissue' in sys.argv:
    TISSUE = sys.argv[sys.argv.index('--tissue') + 1]

COVARIATES_FILE = f"/gpfs/commons/groups/knowles_lab/vmazeeva/BigBrain/Processed/covariates.tsv"


def make_figures(train_covariates, holdout_covariates):

    fig_dir = f"/gpfs/commons/home/vmazeeva/firvTWAS/preprocessing/figures"
    os.makedirs(fig_dir, exist_ok=True)

    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 5))

    # plot 1: number of samples and and number of participants in train and test sets
    x = np.arange(2)
    width = 0.2
    ax.bar(x - width/2, [len(train_covariates['sample_id'].unique()), len(holdout_covariates['sample_id'].unique())], width, label='N Samples')
    ax.bar(x + width/2, [len(train_covariates['participant_id'].unique()), len(holdout_covariates['participant_id'].unique())], width, label='N Participants')
    ax.set_xlabel('Set')
    ax.set_ylabel('Count')
    ax.set_title('Train and Test Set Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(['Train', 'Test'])
    ax.legend()
    plt.savefig(os.path.join(fig_dir, f'train_test_set_statistics_{COHORT}_{TISSUE}.png'))
    plt.close()
    
    # plot 2: distribution of participant ancestry in train and test sets as stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Count participants by predicted_ancestry for each set
    train_ancestry_counts = train_covariates.groupby('predicted_ancestry')['participant_id'].nunique().sort_index()
    holdout_ancestry_counts = holdout_covariates.groupby('predicted_ancestry')['participant_id'].nunique().sort_index()
    
    # Get all unique ancestries and ensure both dataframes have the same categories
    all_ancestries = sorted(set(train_ancestry_counts.index) | set(holdout_ancestry_counts.index))
    train_ancestry_counts = train_ancestry_counts.reindex(all_ancestries, fill_value=0)
    holdout_ancestry_counts = holdout_ancestry_counts.reindex(all_ancestries, fill_value=0)
    
    # Create color map for ancestries
    import matplotlib.cm as cm
    colors = cm.get_cmap('tab20', len(all_ancestries))
    ancestry_colors = {anc: colors(i) for i, anc in enumerate(all_ancestries)}
    
    # Create stacked bar chart
    x_pos = [0, 1]
    bottom_train = 0
    bottom_holdout = 0
    
    # Create legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=ancestry_colors[anc], label=anc) for anc in all_ancestries]
    
    for anc in all_ancestries:
        train_count = train_ancestry_counts[anc]
        holdout_count = holdout_ancestry_counts[anc]
        
        if train_count > 0:
            ax.bar(0, train_count, bottom=bottom_train, 
                   color=ancestry_colors[anc], width=0.6)
            bottom_train += train_count
        
        if holdout_count > 0:
            ax.bar(1, holdout_count, bottom=bottom_holdout, 
                   color=ancestry_colors[anc], width=0.6)
            bottom_holdout += holdout_count
    
    ax.set_xlabel('Set')
    ax.set_ylabel('Number of Participants')
    ax.set_title('Distribution of Predicted Ancestry in Train and Test Sets')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Train', 'Test'])
    ax.legend(handles=legend_elements, title='Predicted Ancestry', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'train_test_ancestry_distribution_{COHORT}_{TISSUE}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Helper function to create stacked bar charts for categorical variables
    def plot_categorical_distribution(train_df, holdout_df, column_name, title, ylabel, filename_suffix):
        """Create a stacked bar chart comparing train and test sets for a categorical variable"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Count participants by the categorical variable for each set
        train_counts = train_df.groupby(column_name)['participant_id'].nunique().sort_index()
        holdout_counts = holdout_df.groupby(column_name)['participant_id'].nunique().sort_index()
        
        # Get all unique categories and ensure both dataframes have the same categories
        all_categories = sorted(set(train_counts.index) | set(holdout_counts.index))
        train_counts = train_counts.reindex(all_categories, fill_value=0)
        holdout_counts = holdout_counts.reindex(all_categories, fill_value=0)
        
        # Create color map
        colors = cm.get_cmap('tab20', len(all_categories))
        category_colors = {cat: colors(i) for i, cat in enumerate(all_categories)}
        
        # Create stacked bar chart
        x_pos = [0, 1]
        bottom_train = 0
        bottom_holdout = 0
        
        # Create legend patches
        legend_elements = [Patch(facecolor=category_colors[cat], label=cat) for cat in all_categories]
        
        for cat in all_categories:
            train_count = train_counts[cat]
            holdout_count = holdout_counts[cat]
            
            if train_count > 0:
                ax.bar(0, train_count, bottom=bottom_train, 
                       color=category_colors[cat], width=0.6)
                bottom_train += train_count
            
            if holdout_count > 0:
                ax.bar(1, holdout_count, bottom=bottom_holdout, 
                       color=category_colors[cat], width=0.6)
                bottom_holdout += holdout_count
        
        ax.set_xlabel('Set')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Train', 'Test'])
        ax.legend(handles=legend_elements, title=column_name.replace('_', ' ').title(), 
                 bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'train_test_{filename_suffix}_{COHORT}_{TISSUE}.png'), 
                         dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Sex distribution
    if 'estimated_sex' in train_covariates.columns:
        plot_categorical_distribution(
            train_covariates, holdout_covariates, 
            'estimated_sex', 
            'Distribution of Estimated Sex in Train and Test Sets',
            'Number of Participants',
            'sex_distribution'
        )
    
    # Plot 4: Disease status distribution
    if 'disease' in train_covariates.columns:
        plot_categorical_distribution(
            train_covariates, holdout_covariates,
            'disease',
            'Distribution of Disease Status in Train and Test Sets',
            'Number of Participants',
            'disease_distribution'
        )
    
    # Plot 5: Cohort distribution
    if 'cohort' in train_covariates.columns:
        plot_categorical_distribution(
            train_covariates, holdout_covariates,
            'cohort',
            'Distribution of Cohorts in Train and Test Sets',
            'Number of Participants',
            'cohort_distribution'
        )
    
    # Plot 6: Brain region distribution
    if 'major_region' in train_covariates.columns:
        plot_categorical_distribution(
            train_covariates, holdout_covariates,
            'major_region',
            'Distribution of Brain Regions in Train and Test Sets',
            'Number of Participants',
            'brain_region_distribution'
        )
    
    # Plot 7: Samples per participant distribution
    train_samples_per_participant = train_covariates.groupby('participant_id').size()
    holdout_samples_per_participant = holdout_covariates.groupby('participant_id').size()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    max_train = train_samples_per_participant.max() if len(train_samples_per_participant) > 0 else 1
    max_holdout = holdout_samples_per_participant.max() if len(holdout_samples_per_participant) > 0 else 1
    max_samples = max(max_train, max_holdout)
    ax.hist([train_samples_per_participant, holdout_samples_per_participant], 
            bins=range(1, max_samples + 2),
            label=['Train', 'Test'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Samples per Participant')
    ax.set_ylabel('Number of Participants')
    ax.set_title('Distribution of Samples per Participant in Train and Test Sets')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'train_test_samples_per_participant_{COHORT}_{TISSUE}.png'), 
                     dpi=300, bbox_inches='tight')
    plt.close()
    


if __name__ == "__main__":

    covariates = pd.read_csv(COVARIATES_FILE, sep="\t")
    print(f"Total number of samples: {len(covariates)}")
    
    # get holdout set
    holdout = covariates[covariates['cohort_tissue'].str.contains(TISSUE)]
    holdout = holdout[holdout['cohort'] == COHORT]

    # check if holdout set is empty
    if len(holdout) == 0:
        print(f"No holdout set found for {COHORT}_{TISSUE}")
        print(f"\tcohort-tissue pairs in covariates: {covariates['cohort_tissue'].unique()}")
        sys.exit(1)

    # get remaining samples for train set
    holdout_indices = holdout.index
    train_indices = covariates.index.difference(holdout_indices)
    train = covariates.loc[train_indices]

    # check if train set is empty
    if len(train) == 0:
        print(f"No train samples remaining after removing holdout set.")
        sys.exit(1)
    
    # print stats
    print(f"TRAIN SET:")
    print(f"\tTotal number of samples: {len(train)}")
    print(f"\tTotal number of participants: {train['participant_id'].nunique()}")
    print("================================================")
    print(f"TEST SET:")
    print(f"\tTotal number of samples: {len(holdout)}")
    print(f"\tTotal number of participants: {holdout['participant_id'].nunique()}")
    print("================================================")
    
    # generate figures
    make_figures(train, holdout)