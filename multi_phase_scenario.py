from util import *
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# Set constants
SCENARIO = "multi_phase_scenario"
os.makedirs(SCENARIO, exist_ok=True)

# Set up for nice plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the ALSFRS longitudinal dataset
data = pd.read_csv('multi_phase_scenario.tsv', sep='\t')
# Basic data inspection
print("Shape of the dataset:", data.shape)

# Subtract 1 from all columns except 'subject_id'
cols_to_modify = data.columns.difference(['subject_id'])
data[cols_to_modify] = data[cols_to_modify] - 1

# Rename columns by removing the '_t1' suffix if present
data.rename(columns=lambda col: col[:-3] if col.endswith('_t1') else col, inplace=True)

print("Number of unique patients:", data['subject_id'].nunique())

# Create Total ALSFRS score
alsfrs_columns = ['Q1Speech', 'Q2Salivation', 'Q3Swallowing', 'Q4Handwriting', 'Q5Cutting', 'Q6Dressing', 'Q7Turning',
                  'Q8Walking', 'Q9Climbing', 'Q10Respiratory']
data['ALSFRS_Total'] = data[alsfrs_columns].sum(axis=1)

# One-hot encode the OnsetSite variable (the other categorical variables are already binary)
data = pd.get_dummies(data, columns=['OnsetSite'], prefix='OnsetSite')
data = data.astype(int)

# Count number of visits per patient
visits_per_patient = data.groupby('subject_id').size().reset_index(name='number_of_visits')
# Summary statistics for visits_per_patient
print(visits_per_patient[['number_of_visits']].describe())

# Create visit number per patient (starting from 1)
data['visit_number'] = data.groupby('subject_id').cumcount() + 1

# Plot ALSFRS progression for a few sample patients
# Sample patients
sample_patients = data['subject_id'].drop_duplicates().sample(5, random_state=42)
sample_data = data[data['subject_id'].isin(sample_patients)]

# Plot ALSFRS progression with visit_number on x-axis
plt.figure(figsize=(12, 8))
sns.lineplot(data=sample_data, x='visit_number', y='ALSFRS_Total', hue='subject_id', marker="o")
plt.title("ALSFRS Progression Over Visits for Sample Patients")
plt.ylabel("ALSFRS Total Score")
plt.xlabel("Visit Number")
plt.legend(title="Patient ID")
plt.savefig(f"{SCENARIO}/sample_progression.png", dpi=300, bbox_inches="tight")
plt.close()

excluded_cols = ['subject_id', 'ALSFRS_Total', 'visit_number']
feature_cols = [col for col in data.columns if col not in excluded_cols]

subject_ids, sequences, targets, target_weights, time_since_onset = build_patient_sequences(data, feature_cols, alsfrs_columns)
# Padding sequences to the same length
# Get the max sequence length
max_seq_length = max(len(seq) for seq in sequences)

# Pad the input features (X)
X = pad_sequences(sequences, maxlen=max_seq_length, padding='post', dtype='float32', value=0.0)
# Pad the target sequences (y)
y = pad_sequences(targets, maxlen=max_seq_length, padding='post', dtype='float32', value=0.0)
# Pad the target_weights (crucial for loss masking)
sample_weights = pad_sequences(target_weights, maxlen=max_seq_length, padding='post', dtype='float32', value=0.0)
# Pad the time_since_onset
time_since_onset = pad_sequences(time_since_onset, maxlen=max_seq_length, padding='post', dtype='int', value=-1)

phases = np.zeros(4800, dtype=int)

# Phase 0: indices from 0 to 999
phases[0:1000] = 0
# Phase 1: indices from 1000 to 1999
phases[1000:2000] = 1
# Phase 2: indices from 2000 to 2249
phases[2000:2250] = 2
# Phase 3 (0->1): indices from 2250 to 4249
phases[2250:4250] = 3
# Phase 4 (1->2): indices from 4250 to 4749
phases[4250:4750] = 4
# Phase 5 (0->1->2): indices from 4750 to 4799
phases[4750:4800] = 5

# First split: 60% training, 40% remaining
X_train, X_temp, y_train, y_temp, sample_weights_train, sample_weights_temp, time_since_onset_train, time_since_onset_temp, phases_train, phases_temp = train_test_split(
    X, y, sample_weights, time_since_onset, phases,
    train_size=0.6,
    random_state=42,
    stratify=phases)

# Split remaining data equally into validation and test set (20% / 20%)
X_val, X_test, y_val, y_test, sample_weights_val, sample_weights_test, time_since_onset_val, time_since_onset_test, phases_val, phases_test = train_test_split(
    X_temp, y_temp, sample_weights_temp, time_since_onset_temp, phases_temp,
    test_size=0.5,
    random_state=42,
    stratify=phases_temp)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, sample_weights_train shape: {sample_weights_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}, sample_weights_val shape: {sample_weights_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, sample_weights_test shape: {sample_weights_test.shape}")

input_shape = X_train.shape[1:]  # (timesteps, features)
output_dim = y_train.shape[-1]


def train_and_eval(model_builder, model_name):
    best_model, best_hp = run_tuning(
        model_builder, SCENARIO, model_name,
        X_train, y_train, sample_weights_train,
        X_val, y_val, sample_weights_val
    )

    # Save best HP
    save_json(best_hp.values, f"{SCENARIO}/{model_name}_best_hp.json")

    # Evaluate
    test_metrics, subscores_metrics, phase_metrics = evaluate_model(
        best_model, X_test, y_test, sample_weights_test, time_since_onset_test
    )

    # Save metrics
    results = {"overall": test_metrics,
               "subscores": subscores_metrics,
               "phase_metrics": phase_metrics}
    save_json(results, f"{SCENARIO}/{model_name}_metrics.json")

    # Save plots
    plot_metrics(subscores_metrics, SCENARIO, model_name)
    plot_phase_metrics(phase_metrics, SCENARIO, model_name)

    # Transition metrics
    transition_metrics = evaluate_transitions_by_offset(
        best_model, X_test, y_test, sample_weights_test, time_since_onset_test
    )
    save_json(transition_metrics, f"{SCENARIO}/{model_name}_transition_metrics.json")
    plot_transition_metrics(transition_metrics, SCENARIO, model_name, metric="RMSE")

    return best_model, best_hp


# Run for all model types
models = {
    "lstm": lstm_builder(input_shape, output_dim),
    "gru": gru_builder(input_shape, output_dim),
    "stacked_lstm": build_stacked_lstm(input_shape, output_dim),
    "stacked_gru": build_stacked_gru(input_shape, output_dim),
    "lstm_attention": build_lstm_with_attention(input_shape, output_dim),
    "gru_attention": build_gru_with_attention(input_shape, output_dim),
    "transformer": build_transformer_model(input_shape, output_dim)
}

for name, builder in models.items():
    train_and_eval(builder, name)
