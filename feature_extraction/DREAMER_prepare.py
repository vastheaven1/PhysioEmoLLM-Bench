import os
import argparse
import numpy as np
import scipy.io

# Split the original DREAMER.mat into one .mat file per subject (subject_01.mat ... subject_23.mat).
# Each file stores a 'combined_data' array of shape (18, 7): one row per trial with
# [eeg_baseline, eeg_stimuli, ecg_baseline, ecg_stimuli, score_valence, score_arousal, score_dominance],
# which is the input format expected by feature_extraction/DREAMER.py.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dreamer_mat', help='path to the original DREAMER.mat', default='DREAMER.mat')
    parser.add_argument('--output_dir', help='directory to save the per-subject .mat files', default='data_subject')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    mat_cont = scipy.io.loadmat(args.dreamer_mat)

    # Data of the 23 subjects
    data_all_subjects = mat_cont['DREAMER']['Data'][0][0][0]

    os.makedirs(args.output_dir, exist_ok=True)
    for i, subject_data in enumerate(data_all_subjects):
        subject_id = i + 1

        # EEG and ECG recordings (baseline and stimuli), one cell per trial
        eeg_baseline = subject_data[0][0]['EEG']['baseline'][0][0]
        eeg_stimuli = subject_data[0][0]['EEG']['stimuli'][0][0]
        ecg_baseline = subject_data[0][0]['ECG']['baseline'][0][0]
        ecg_stimuli = subject_data[0][0]['ECG']['stimuli'][0][0]

        # Self-assessment ratings, one per trial
        score_valence = subject_data[0][0]['ScoreValence']
        score_arousal = subject_data[0][0]['ScoreArousal']
        score_dominance = subject_data[0][0]['ScoreDominance']

        combined_data = np.hstack((
            eeg_baseline, eeg_stimuli, ecg_baseline, ecg_stimuli,
            score_valence, score_arousal, score_dominance
        ))

        output_path = os.path.join(args.output_dir, f'subject_{subject_id:02d}.mat')
        scipy.io.savemat(output_path, {'combined_data': combined_data})
        print(f'Subject {subject_id}: combined data shape {combined_data.shape}, saved to {output_path}')
