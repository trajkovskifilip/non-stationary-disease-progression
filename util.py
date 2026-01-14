import numpy as np
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.layers import Input, Masking, LSTM, GRU, TimeDistributed, Dense, LayerNormalization, MultiHeadAttention, Dropout, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt


def build_patient_sequences(data, feature_cols, alsfrs_columns):
    """
    Build sequences, next-step targets, temporal weights, and TimeSinceOnset for ALS patients.
    For the last time step, the target is the current (last) ALSFRS score, but
    its weight is 0 to exclude it from loss calculation. TimeSinceOnset is returned per patient,
    excluding the first visit as it is not predicted.

    Parameters:
    - data: DataFrame with ALS visit data, must contain 'subject_id', 'visit_number',
            feature columns, and ALSFRS subscore columns, and 'TimeSinceOnset'.
    - feature_cols: list of column names to use as input features.
    - alsfrs_columns: list of ALSFRS subscore column names (targets).

    Returns:
    - subject_ids: list of patient IDs
    - sequences: list of input feature sequences (one per patient)
    - targets: list of target sequences (ALSFRS subscores at next time step)
    - target_weights: list of weight sequences (1 for valid targets, 0 for last real visit)
    - time_since_onset: list of TimeSinceOnset sequences (one per patient, excluding first visit)
    """
    subject_ids = []
    sequences = []
    targets = []
    target_weights = []
    time_since_onset = []

    for patient_id, patient_data in data.groupby('subject_id'):
        # Sort visits chronologically
        patient_data = patient_data.sort_values('visit_number')

        # Extract and scale feature inputs
        # feature_data = feature_scaler.transform(patient_data[feature_cols])
        feature_data = patient_data[feature_cols].values

        # Extract and scale ALSFRS subscore targets
        # alsfrs_subscores = target_scaler.transform(patient_data[alsfrs_columns])
        alsfrs_subscores = patient_data[alsfrs_columns].values

        # Shift ALSFRS targets to align with next timestep
        target = np.zeros_like(alsfrs_subscores)
        target[:-1] = alsfrs_subscores[1:]

        # For the last timestep, the target is the ALSFRS score at that last visit
        target[-1] = alsfrs_subscores[-1]

        # Create the weight array for targets
        weights = np.ones(alsfrs_subscores.shape[0])

        # Set the weight for the last real timestep to 0 (to omit from loss)
        weights[-1] = 0

        # Extract TimeSinceOnset per visit (excluding the first one)
        tso = patient_data["TimeSinceOnset"].values
        time_since_onset.append(tso[1:])  # exclude first visit

        # Collect data
        sequences.append(feature_data)
        targets.append(target)
        target_weights.append(weights)
        subject_ids.append(patient_id)

    return subject_ids, sequences, targets, target_weights, time_since_onset


def plot_training_history(history, output_dir, project_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    fname = f"{output_dir}/{project_name}_training_history.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def lstm_builder(input_shape, output_units=10):
    """
    Returns a builder function that only takes hp and uses input_shape.
    :param input_shape: should be tuple, e.g. (timesteps, num_features).
    :param output_units: is the output size (number of time-distributed units).
    :return: builder
    """
    def builder(hp):
        recurrent_units = hp.Choice("recurrent_units", [16, 32, 64, 128])
        recurrent_dropout = hp.Choice("recurrent_dropout", [0.0, 0.2])
        dense_dropout = hp.Choice("dense_dropout", [0.0, 0.2])
        # how many dense layers after recurrent
        n_layers = hp.Choice("n_dense_layers", [1, 2, 3])
        # starting width of the first dense layer
        start_units = hp.Choice("start_dense_units", [32, 64, 128])

        # construct halving config
        dense_config = []
        units = start_units
        for i in range(n_layers - 1):
            dense_config.append(units)
            units = units // 2  # halve each time
        dense_config.append(output_units)  # final fixed output size

        inputs = Input(shape=input_shape)
        x = Masking(mask_value=0.0)(inputs)

        x = LSTM(recurrent_units, recurrent_dropout=recurrent_dropout, return_sequences=True)(x)

        for units in dense_config[:-1]:
            x = TimeDistributed(Dense(units, activation="relu"))(x)
            if dense_dropout > 0.0:
                x = Dropout(dense_dropout)(x)

        outputs = TimeDistributed(Dense(dense_config[-1]))(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    return builder


def gru_builder(input_shape, output_units=10):
    """
    Returns a builder function that only takes hp and uses input_shape.
    :param input_shape: should be tuple, e.g. (timesteps, num_features).
    :param output_units: is the output size (number of time-distributed units).
    :return: builder
    """
    def builder(hp):
        recurrent_units = hp.Choice("recurrent_units", [16, 32, 64, 128])
        recurrent_dropout = hp.Choice("recurrent_dropout", [0.0, 0.2])
        dense_dropout = hp.Choice("dense_dropout", [0.0, 0.2])
        # how many dense layers after recurrent
        n_layers = hp.Choice("n_dense_layers", [1, 2, 3])
        # starting width of the first dense layer
        start_units = hp.Choice("start_dense_units", [32, 64, 128])

        # construct halving config
        dense_config = []
        units = start_units
        for i in range(n_layers - 1):
            dense_config.append(units)
            units = units // 2  # halve each time
        dense_config.append(output_units)  # final fixed output size

        inputs = Input(shape=input_shape)
        x = Masking(mask_value=0.0)(inputs)

        x = GRU(recurrent_units, recurrent_dropout=recurrent_dropout, return_sequences=True)(x)

        for units in dense_config[:-1]:
            x = TimeDistributed(Dense(units, activation="relu"))(x)
            if dense_dropout > 0.0:
                x = Dropout(dense_dropout)(x)

        outputs = TimeDistributed(Dense(dense_config[-1]))(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    return builder


def build_stacked_lstm(input_shape, output_units=10):
    """
    Returns a builder function that only takes hp and uses input_shape.
    :param input_shape: should be tuple, e.g. (timesteps, num_features).
    :param output_units: is the output size (number of time-distributed units).
    :return: builder
    """
    def builder(hp):
        # --- Recurrent part ---
        n_recurrent_layers = 2  # fixed to 2 stacked layers
        start_units = hp.Choice("lstm_start_units", [32, 64, 128])
        recurrent_dropout = hp.Choice("lstm_dropout", [0.0, 0.2])

        # build recurrent units halving at each layer
        lstm_units = [start_units // (2**i) for i in range(n_recurrent_layers)]

        inputs = Input(shape=input_shape)
        x = Masking(mask_value=0.0)(inputs)

        for i in range(n_recurrent_layers):
            # always return sequences if you want dense TimeDistributed later
            x = LSTM(lstm_units[i], return_sequences=True,
                     recurrent_dropout=recurrent_dropout)(x)
            x = LayerNormalization()(x)

        # --- Dense part ---
        n_dense_layers = hp.Choice("n_dense_layers", [1, 2, 3])
        start_dense_units = hp.Choice("dense_start_units", [32, 64, 128])
        dense_dropout = hp.Choice("dense_dropout", [0.0, 0.2])

        dense_units = []
        units = start_dense_units
        for i in range(n_dense_layers - 1):
            dense_units.append(units)
            units = units // 2
        dense_units.append(output_units)  # always finish with 10 neurons

        for units in dense_units[:-1]:
            x = TimeDistributed(Dense(units, activation="relu"))(x)
            if dense_dropout > 0.0:
                x = Dropout(dense_dropout)(x)

        outputs = TimeDistributed(Dense(dense_units[-1]))(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    return builder


def build_stacked_gru(input_shape, output_units=10):
    """
    Returns a builder function that only takes hp and uses input_shape.
    :param input_shape: should be tuple, e.g. (timesteps, num_features).
    :param output_units: is the output size (number of time-distributed units).
    :return: builder
    """
    def builder(hp):
        # --- Recurrent part ---
        n_recurrent_layers = 2
        start_units = hp.Choice("gru_start_units", [32, 64, 128])
        recurrent_dropout = hp.Choice("gru_dropout", [0.0, 0.2])

        gru_units = [start_units // (2**i) for i in range(n_recurrent_layers)]

        inputs = Input(shape=input_shape)
        x = Masking(mask_value=0.0)(inputs)

        for i in range(n_recurrent_layers):
            x = GRU(gru_units[i], return_sequences=True,
                    recurrent_dropout=recurrent_dropout)(x)
            x = LayerNormalization()(x)

        # --- Dense part ---
        n_dense_layers = hp.Choice("n_dense_layers", [1, 2, 3])
        start_dense_units = hp.Choice("dense_start_units", [32, 64, 128])
        dense_dropout = hp.Choice("dense_dropout", [0.0, 0.2])

        dense_units = []
        units = start_dense_units
        for i in range(n_dense_layers - 1):
            dense_units.append(units)
            units = units // 2
        dense_units.append(output_units)

        for units in dense_units[:-1]:
            x = TimeDistributed(Dense(units, activation="relu"))(x)
            if dense_dropout > 0.0:
                x = Dropout(dense_dropout)(x)

        outputs = TimeDistributed(Dense(dense_units[-1]))(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    return builder


class CombinedMask(tf.keras.layers.Layer):
    def call(self, inputs):
        # inputs: (batch_size, seq_len, num_features)
        # Padding Mask: True for non-zero (non-padded) elements, False for zero (padded)
        padding_mask = tf.reduce_any(tf.not_equal(inputs, 0.0), axis=-1)  # (batch_size, seq_len)
        padding_mask = tf.cast(padding_mask, tf.float32)
        # Expand for broadcasting with causal mask later
        padding_mask = tf.expand_dims(padding_mask, 1)  # (batch_size, 1, seq_len)

        # Causal (Look-ahead) Mask: Lower triangular matrix
        seq_len = tf.shape(inputs)[1]
        # True for allowed connections (current/past), False for disallowed (future/look-ahead)
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0)  # (seq_len, seq_len)
        # Expand for broadcasting with padding mask later
        causal_mask = tf.expand_dims(causal_mask, 0)  # (1, seq_len, seq_len)

        # Combine Masks: Element-wise logical AND
        # A position (query_i, key_j) is valid for attention only if it's NOT a padding token (j) AND it's causal (i <= j).
        # The result will be (batch_size, seq_len, seq_len)
        # True where attention is ALLOWED, False where it should be MASKED.
        combined_mask = padding_mask * causal_mask  # (batch_size, 1, seq_len, seq_len)

        combined_mask = tf.cast(combined_mask > 0, tf.bool)
        return combined_mask

    def compute_output_shape(self, input_shape):
        batch_size, seq_len, _ = input_shape
        return batch_size, 1, seq_len, seq_len

    def get_config(self):
        config = super().get_config()
        return config


def build_lstm_with_attention(input_shape, output_units=10):
    """
    Returns a builder function that only takes hp and uses input_shape.
    :param input_shape: should be tuple, e.g. (timesteps, num_features).
    :param output_units: is the output size (number of time-distributed units).
    :return: builder
    """
    def builder(hp):
        recurrent_units = hp.Choice("lstm_units", [16, 32, 64])
        recurrent_dropout = hp.Choice("lstm_dropout", [0.0, 0.2])
        num_heads = hp.Choice("attention_heads", [2, 4, 8, 16])
        dense_dropout = hp.Choice("dense_dropout", [0.0, 0.2])
        n_dense_layers = hp.Choice("n_dense_layers", [1, 2, 3])
        start_dense_units = hp.Choice("dense_start_units", [32, 64, 128])

        inputs = Input(shape=input_shape)
        attention_mask = CombinedMask()(inputs)

        x = Masking(mask_value=0.0)(inputs)

        # Recurrent layer
        x = LSTM(recurrent_units, return_sequences=True, recurrent_dropout=recurrent_dropout)(x)
        x = LayerNormalization()(x)

        # Self-Attention
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=recurrent_units)(
            query=x, value=x, key=x, attention_mask=attention_mask
        )
        attn_out = Dropout(dense_dropout)(attn_out)
        x = LayerNormalization()(x + attn_out)

        # Dense layers (halving rule)
        units = start_dense_units
        for i in range(n_dense_layers - 1):
            x = Dense(units, activation="relu")(x)
            if dense_dropout > 0.0:
                x = Dropout(dense_dropout)(x)
            units = units // 2

        outputs = Dense(output_units)(x)

        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    return builder


def build_gru_with_attention(input_shape, output_units=10):
    """
    Returns a builder function that only takes hp and uses input_shape.
    :param input_shape: should be tuple, e.g. (timesteps, num_features).
    :param output_units: is the output size (number of time-distributed units).
    :return: builder
    """
    def builder(hp):
        recurrent_units = hp.Choice("gru_units", [16, 32, 64])
        recurrent_dropout = hp.Choice("gru_dropout", [0.0, 0.2])
        num_heads = hp.Choice("attention_heads", [2, 4, 8, 16])
        dense_dropout = hp.Choice("dense_dropout", [0.0, 0.2])
        n_dense_layers = hp.Choice("n_dense_layers", [1, 2, 3])
        start_dense_units = hp.Choice("dense_start_units", [32, 64, 128])

        inputs = Input(shape=input_shape)
        attention_mask = CombinedMask()(inputs)

        x = Masking(mask_value=0.0)(inputs)

        # Recurrent layer
        x = GRU(recurrent_units, return_sequences=True, recurrent_dropout=recurrent_dropout)(x)
        x = LayerNormalization()(x)

        # Self-Attention
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=recurrent_units)(
            query=x, value=x, key=x, attention_mask=attention_mask
        )
        attn_out = Dropout(dense_dropout)(attn_out)
        x = LayerNormalization()(x + attn_out)

        # Dense layers (halving rule)
        units = start_dense_units
        for i in range(n_dense_layers - 1):
            x = Dense(units, activation="relu")(x)
            if dense_dropout > 0.0:
                x = Dropout(dense_dropout)(x)
            units = units // 2

        outputs = Dense(output_units)(x)

        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    return builder


class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        # Use a Keras Embedding layer for learnable positional embeddings
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = tf.shape(x)[1]
        # Create positional indices: [0, 1, 2, ..., seq_len-1]
        positions = tf.range(start=0, limit=seq_len, delta=1, dtype=tf.int32)
        # Look up embeddings for these positions
        embedded_positions = self.pos_embedding(positions)
        # Add positional embeddings to the input embeddings
        return x + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model,
        })
        return config


# ---- Transformer Block ----
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        # Ensure d_model is divisible by num_heads
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        # key_dim for MultiHeadAttention is the dimension of the key/query/value vectors PER HEAD.
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model),
        ])
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, x, mask, training=False):
        # attention_mask: A boolean mask for the attention scores.
        # Elements set to `False` in the mask will be ignored (masked out).
        attn_output = self.att(query=x, value=x, key=x, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)

        # Feed Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        })
        return config


def build_transformer_model(input_shape, output_units=10):
    """
    Returns a builder function that only takes hp and uses input_shape.
    :param input_shape: should be tuple, e.g. (timesteps, num_features).
    :param output_units: is the output size (number of time-distributed units).
    :return: builder
    """
    def builder(hp):
        d_model = hp.Choice("d_model", [32, 64, 128])
        num_heads = hp.Choice("num_heads", [2, 4, 8, 16])
        ff_dim = hp.Choice("ff_dim", [64, 128])
        num_layers = hp.Choice("num_layers", [2, 3])
        dropout_rate = hp.Choice("dropout_rate", [0.0, 0.2])

        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        inputs = Input(shape=input_shape)  # (seq_len, num_features)

        # Project input features to d_model dimension
        x = TimeDistributed(Dense(d_model, activation="relu"))(inputs)

        # Learnable positional encoding
        x = LearnablePositionalEncoding(max_len=input_shape[0], d_model=d_model)(x)

        # Build padding+causal mask
        mask = CombinedMask()(inputs)

        # Stack Transformer blocks
        for _ in range(num_layers):
            x = TransformerBlock(d_model, num_heads, ff_dim, dropout=dropout_rate)(x, mask)

        outputs = TimeDistributed(Dense(output_units))(x)
        model = Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    return builder


def run_tuning(model_builder, output_dir, project_name,
               X_train, y_train, sample_weights_train,
               X_val, y_val, sample_weights_val,
               max_trials=20, epochs=50, batch_size=32):
    stop_early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # tuner setup (you can choose RandomSearch, BayesianOptimization, or Hyperband)
    # Bayesian tuner setup
    tuner = kt.BayesianOptimization(
        hypermodel=model_builder,
        objective="val_loss",
        max_trials=max_trials,
        num_initial_points=10,
        executions_per_trial=1,
        overwrite=True,
        directory=f"{output_dir}/tuner_results",
        project_name=project_name
    )

    tuner.search(
        X_train, y_train,
        sample_weight=sample_weights_train,
        validation_data=(X_val, y_val, sample_weights_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[stop_early],
        verbose=1,
        shuffle=True
    )

    # Best hyperparams
    best_hps = tuner.get_best_hyperparameters(1)[0]
    print("\nBest hyperparameters found:")
    for k, v in best_hps.values.items():
        print(f"{k}: {v}")

    # Rebuild model with best hyperparams
    best_model = tuner.hypermodel.build(best_hps)

    # Merge train + val for final training
    X_final = np.concatenate([X_train, X_val], axis=0)
    y_final = np.concatenate([y_train, y_val], axis=0)
    sample_weights_final = np.concatenate([sample_weights_train, sample_weights_val], axis=0)

    final_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    history = best_model.fit(
        X_final, y_final,
        sample_weight=sample_weights_final,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[final_stop],
        verbose=1,
        shuffle=True
    )

    # Save final model
    best_model.save(f"{output_dir}/best_{project_name}_final.keras")
    return best_model, best_hps


def get_indices(ranges):
    indices = []
    for start, end in ranges:
        indices.extend(range(start, end + 1))
    return np.array(indices)


def relative_mse(y_true, mse):
    var = np.var(y_true)
    return mse / var if var != 0 else np.nan


def relative_rmse(y_true, rmse):
    std = np.std(y_true)
    return rmse / std if std != 0 else np.nan


def relative_mae(y_true, mae):
    mean_abs = np.mean(np.abs(y_true))
    return mae / mean_abs if mean_abs != 0 else np.nan


def evaluate_model(model, X_test, y_test, sample_weights_test, phases_test):
    """
    Evaluates the trained model, considering only the relevant timesteps
    as defined by sample_weights_test.

    Parameters:
    - model: The trained model.
    - X_test: Padded input features for the test set.
    - y_test: Padded target labels for the test set.
    - sample_weights_test: Padded sample weights for the test set (1 for valid, 0 for excluded).
    - phases_test: Patient visit's phases in {0, 1, 2} or -1 for padding
    """
    # --- Temporarily enable eager execution for model.predict ---
    # This can sometimes resolve OperatorNotAllowedInGraphError issues
    # by preventing strict graph compilation for the prediction step.
    original_tf_function_run_eagerly = tf.config.functions_run_eagerly()
    tf.config.run_functions_eagerly(True)

    try:
        # Predict using the model (outputs predicitons for all timesteps including padded and last real)
        y_pred = model.predict(X_test)
    finally:
        # --- IMPORTANT: Revert to original eager execution setting ---
        tf.config.run_functions_eagerly(original_tf_function_run_eagerly)

    # Inverse transform both y_test and y_pred from [0, 1] back to [0, 4]
    # Flatten to 2D for scaler, then reshape back
    # y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[-1])).reshape(y_pred.shape)
    # y_test = target_scaler.inverse_transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)

    # Round and clip to enforce integer outputs in [0, 4]
    # y_pred = np.clip(np.round(y_pred), 0, 4).astype(int)
    # y_test = np.clip(np.round(y_test), 0, 4).astype(int)

    # Convert sample_weights_test to a boolean mask.
    # This mask will be True where sample_weight is 1 (valid next-step predictions)
    # and False where sample_weight is 0 (padded or last visit).
    boolean_mask = sample_weights_test.astype(bool)

    # Get the output dimension (number of ALSFRS subscores)
    n_patients, max_timesteps, output_dim = y_test.shape

    # Create a full 3D boolean mask to match the shape of y_test/y_pred (num_samples, max_len, output_dim)
    # This allows us to use direct boolean indexing on the 3D arrays.
    full_boolean_mask = np.repeat(np.expand_dims(boolean_mask, axis=-1), output_dim, axis=-1)

    # Flatten and mask true and predicted values
    # This will now correctly exclude padded values AND the last real visit's predictions
    y_true_masked = y_test[full_boolean_mask]
    y_pred_masked = y_pred[full_boolean_mask]

    # Compute overall metrics
    mse = mean_squared_error(y_true_masked, y_pred_masked)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_masked, y_pred_masked)
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2_score(y_true_masked, y_pred_masked),
        'rel_mse': relative_mse(y_true_masked, mse),
        'rel_rmse': relative_rmse(y_true_masked, rmse),
        'rel_mae': relative_mae(y_true_masked, mae)
    }

    print(f"Overall Metrics: MSE={metrics['mse']:.4f} ({metrics['rel_mse']:.4f}), "
          f"RMSE={metrics['rmse']:.4f} ({metrics['rel_rmse']:.4f}), MAE={metrics['mae']:.4f} ({metrics['rel_mae']:.4f}), "
          f"R²={metrics['r2']:.4f}")

    # Per-subscore metrics
    subscores_metrics = []
    for i in range(output_dim):
        # Select the specific subscore from true and predicted values, then apply the 2D boolean mask
        y_true_sub = y_test[..., i][boolean_mask]
        y_pred_sub = y_pred[..., i][boolean_mask]

        mse_sub = mean_squared_error(y_true_sub, y_pred_sub)
        rmse_sub = np.sqrt(mse_sub)
        mae_sub = mean_absolute_error(y_true_sub, y_pred_sub)
        sub_metrics = {
            'mse': mse_sub,
            'rmse': rmse_sub,
            'mae': mae_sub,
            'r2': r2_score(y_true_sub, y_pred_sub),
            'rel_mse': relative_mse(y_true_sub, mse_sub),
            'rel_rmse': relative_rmse(y_true_sub, rmse_sub),
            'rel_mae': relative_mae(y_true_sub, mae_sub)
        }
        subscores_metrics.append(sub_metrics)

        print(f"ALSFRS-{i+1}: "
              f"MSE={sub_metrics['mse']:.4f} ({sub_metrics['rel_mse']:.4f}), "
              f"RMSE={sub_metrics['rmse']:.4f} ({sub_metrics['rel_rmse']:.4f}), "
              f"MAE={sub_metrics['mae']:.4f} ({sub_metrics['rel_mae']:.4f}), "
              f"R²={sub_metrics['r2']:.4f}")

    # --- Metrics by TimeSinceOnset Phase ---
    phase_metrics = {}
    phases_to_eval = [0, 1, 2]

    print("\n--- Metrics by TimeSinceOnset Phase ---")
    for phase_value in phases_to_eval:
        print(f"\n--- Phase {phase_value} ---")

        # Build mask for visits that belong to this phase AND are valid
        phase_mask = (phases_test == phase_value) & boolean_mask
        full_phase_mask = np.repeat(np.expand_dims(phase_mask, axis=-1), output_dim, axis=-1)

        y_true_masked_phase = y_test[full_phase_mask]
        y_pred_masked_phase = y_pred[full_phase_mask]

        if len(y_true_masked_phase) == 0:
            print(f"No valid predictions to evaluate for Phase {phase_value}.")
            phase_metrics[phase_value] = {'overall': {}, 'subscores': []}
            continue

        # Compute overall metrics for this phase
        mse_phase = mean_squared_error(y_true_masked_phase, y_pred_masked_phase)
        rmse_phase = np.sqrt(mse_phase)
        mae_phase = mean_absolute_error(y_true_masked_phase, y_pred_masked_phase)
        metrics_phase_overall = {
            'mse': mse_phase,
            'rmse': rmse_phase,
            'mae': mae_phase,
            'r2': r2_score(y_true_masked_phase, y_pred_masked_phase),
            'rel_mse': relative_mse(y_true_masked_phase, mse_phase),
            'rel_rmse': relative_rmse(y_true_masked_phase, rmse_phase),
            'rel_mae': relative_mae(y_true_masked_phase, mae_phase)
        }
        print(f"Overall Metrics (Phase {phase_value}): MSE={metrics_phase_overall['mse']:.4f} ({metrics_phase_overall['rel_mse']:.4f}), "
              f"RMSE={metrics_phase_overall['rmse']:.4f} ({metrics_phase_overall['rel_rmse']:.4f}), MAE={metrics_phase_overall['mae']:.4f} ({metrics_phase_overall['rel_mae']:.4f}), "
              f"R²={metrics_phase_overall['r2']:.4f}")

        # Per-subscore metrics for this phase
        subscores_metrics_phase = []
        for i in range(output_dim):
            y_true_sub_phase = y_test[..., i][phase_mask]
            y_pred_sub_phase = y_pred[..., i][phase_mask]

            # Handle cases where a subscore might have no valid data in this phase (e.g., all masked out)
            if len(y_true_sub_phase) == 0:
                sub_metrics = {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
                               'rel_mse': np.nan, 'rel_rmse': np.nan, 'rel_mae': np.nan}
            else:
                mse_sub_phase = mean_squared_error(y_true_sub_phase, y_pred_sub_phase)
                rmse_sub_phase = np.sqrt(mse_sub_phase)
                mae_sub_phase = mean_absolute_error(y_true_sub_phase, y_pred_sub_phase)
                sub_metrics = {
                    'mse': mse_sub_phase,
                    'rmse': rmse_sub_phase,
                    'mae': mae_sub_phase,
                    'r2': r2_score(y_true_sub_phase, y_pred_sub_phase),
                    'rel_mse': relative_mse(y_true_sub_phase, mse_sub_phase),
                    'rel_rmse': relative_rmse(y_true_sub_phase, rmse_sub_phase),
                    'rel_mae': relative_mae(y_true_sub_phase, mae_sub_phase)
                }
            subscores_metrics_phase.append(sub_metrics)

            print(f"Subscore {i+1} (Phase {phase_value}): "
                  f"MSE={sub_metrics['mse']:.4f} ({sub_metrics['rel_mse']:.4f}), "
                  f"RMSE={sub_metrics['rmse']:.4f} ({sub_metrics['rel_rmse']:.4f}), "
                  f"MAE={sub_metrics['mae']:.4f} ({sub_metrics['rel_mae']:.4f}), "
                  f"R²={sub_metrics['r2']:.4f}")

        phase_metrics[phase_value] = {'overall': metrics_phase_overall, 'subscores': subscores_metrics_phase}

    return metrics, subscores_metrics, phase_metrics


def evaluate_model_equal_visits(model, X_test, y_test, sample_weights_test):
    """
    Evaluates the trained model, considering only the relevant timesteps
    as defined by sample_weights_test. Splits performance into phases:
    - Phase 0: visits 0-6 (predictions for visits 1-6)
    - Phase 1: visits 7-13
    - Phase 2: visits 14-20

    Parameters:
    - model: The trained model.
    - X_test: Padded input features for the test set.
    - y_test: Padded target labels for the test set.
    - sample_weights_test: Padded sample weights for the test set (1 for valid, 0 for excluded).
    """
    # --- Temporarily enable eager execution for model.predict ---
    # This can sometimes resolve OperatorNotAllowedInGraphError issues
    # by preventing strict graph compilation for the prediction step.
    original_tf_function_run_eagerly = tf.config.functions_run_eagerly()
    tf.config.run_functions_eagerly(True)

    try:
        # Predict using the model (outputs predicitons for all timesteps including padded and last real)
        y_pred = model.predict(X_test)
    finally:
        # --- IMPORTANT: Revert to original eager execution setting ---
        tf.config.run_functions_eagerly(original_tf_function_run_eagerly)

    # Inverse transform both y_test and y_pred from [0, 1] back to [0, 4]
    # Flatten to 2D for scaler, then reshape back
    # y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[-1])).reshape(y_pred.shape)
    # y_test = target_scaler.inverse_transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)

    # Round and clip to enforce integer outputs in [0, 4]
    # y_pred = np.clip(np.round(y_pred), 0, 4).astype(int)
    # y_test = np.clip(np.round(y_test), 0, 4).astype(int)

    # Convert sample_weights_test to a boolean mask.
    # This mask will be True where sample_weight is 1 (valid next-step predictions)
    # and False where sample_weight is 0 (padded or last visit).
    boolean_mask = sample_weights_test.astype(bool)

    # Get the output dimension (number of ALSFRS subscores)
    output_dim = y_test.shape[-1]

    # Create a full 3D boolean mask to match the shape of y_test/y_pred (num_samples, max_len, output_dim)
    # This allows us to use direct boolean indexing on the 3D arrays.
    full_boolean_mask = np.repeat(np.expand_dims(boolean_mask, axis=-1), output_dim, axis=-1)

    # Flatten and mask true and predicted values
    # This will now correctly exclude padded values AND the last real visit's predictions
    y_true_masked = y_test[full_boolean_mask]
    y_pred_masked = y_pred[full_boolean_mask]

    # Compute overall metrics
    mse = mean_squared_error(y_true_masked, y_pred_masked)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_masked, y_pred_masked)
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2_score(y_true_masked, y_pred_masked),
        'rel_mse': relative_mse(y_true_masked, mse),
        'rel_rmse': relative_rmse(y_true_masked, rmse),
        'rel_mae': relative_mae(y_true_masked, mae)
    }

    print(f"Overall Metrics: MSE={metrics['mse']:.4f} ({metrics['rel_mse']:.4f}), "
          f"RMSE={metrics['rmse']:.4f} ({metrics['rel_rmse']:.4f}), MAE={metrics['mae']:.4f} ({metrics['rel_mae']:.4f}), "
          f"R²={metrics['r2']:.4f}")

    # Per-subscore metrics
    subscores_metrics = []
    for i in range(output_dim):
        # Select the specific subscore from true and predicted values, then apply the 2D boolean mask
        y_true_sub = y_test[..., i][boolean_mask]
        y_pred_sub = y_pred[..., i][boolean_mask]

        mse_sub = mean_squared_error(y_true_sub, y_pred_sub)
        rmse_sub = np.sqrt(mse_sub)
        mae_sub = mean_absolute_error(y_true_sub, y_pred_sub)
        sub_metrics = {
            'mse': mse_sub,
            'rmse': rmse_sub,
            'mae': mae_sub,
            'r2': r2_score(y_true_sub, y_pred_sub),
            'rel_mse': relative_mse(y_true_sub, mse_sub),
            'rel_rmse': relative_rmse(y_true_sub, rmse_sub),
            'rel_mae': relative_mae(y_true_sub, mae_sub)
        }
        subscores_metrics.append(sub_metrics)

        print(f"ALSFRS-{i+1}: "
              f"MSE={sub_metrics['mse']:.4f} ({sub_metrics['rel_mse']:.4f}), "
              f"RMSE={sub_metrics['rmse']:.4f} ({sub_metrics['rel_rmse']:.4f}), "
              f"MAE={sub_metrics['mae']:.4f} ({sub_metrics['rel_mae']:.4f}), "
              f"R²={sub_metrics['r2']:.4f}")

     # --- Metrics by TimeSinceOnset Phase ---
    phase_ranges = {
        0: range(0, 6),   # visits 1–6
        1: range(6, 13),  # visits 7–13
        2: range(13, 20)   # visits 14–20
    }
    phase_metrics = {}

    print("\n--- Metrics by TimeSinceOnset Phase ---")
    for phase, visit_range in phase_ranges.items():
        print(f"\n--- Phase {phase} ---")

        phase_mask = np.zeros_like(boolean_mask, dtype=bool)
        phase_mask[:, visit_range] = boolean_mask[:, visit_range]
        full_phase_mask = np.repeat(np.expand_dims(phase_mask, axis=-1), output_dim, axis=-1)

        # Slice the data (true labels, predictions, and sample weights) for this phase's patients
        y_true_phase = y_test[full_phase_mask]
        y_pred_phase = y_pred[full_phase_mask]

        # Compute overall metrics for this phase
        mse_phase = mean_squared_error(y_true_phase, y_pred_phase)
        rmse_phase = np.sqrt(mse_phase)
        mae_phase = mean_absolute_error(y_true_phase, y_pred_phase)
        metrics_phase_overall = {
            'mse': mse_phase,
            'rmse': rmse_phase,
            'mae': mae_phase,
            'r2': r2_score(y_true_phase, y_pred_phase),
            'rel_mse': relative_mse(y_true_phase, mse_phase),
            'rel_rmse': relative_rmse(y_true_phase, rmse_phase),
            'rel_mae': relative_mae(y_true_phase, mae_phase)
        }
        print(f"Overall Metrics (Phase {phase}): MSE={metrics_phase_overall['mse']:.4f} ({metrics_phase_overall['rel_mse']:.4f}), "
              f"RMSE={metrics_phase_overall['rmse']:.4f} ({metrics_phase_overall['rel_rmse']:.4f}), MAE={metrics_phase_overall['mae']:.4f} ({metrics_phase_overall['rel_mae']:.4f}), "
              f"R²={metrics_phase_overall['r2']:.4f}")

        # Per-subscore metrics for this phase
        subscores_metrics_phase = []
        for i in range(output_dim):
            y_true_sub_phase = y_test[..., i][phase_mask]
            y_pred_sub_phase = y_pred[..., i][phase_mask]

            mse_sub_phase = mean_squared_error(y_true_sub_phase, y_pred_sub_phase)
            rmse_sub_phase = np.sqrt(mse_sub_phase)
            mae_sub_phase = mean_absolute_error(y_true_sub_phase, y_pred_sub_phase)
            sub_metrics = {
                'mse': mse_sub_phase,
                'rmse': rmse_sub_phase,
                'mae': mae_sub_phase,
                'r2': r2_score(y_true_sub_phase, y_pred_sub_phase),
                'rel_mse': relative_mse(y_true_sub_phase, mse_sub_phase),
                'rel_rmse': relative_rmse(y_true_sub_phase, rmse_sub_phase),
                'rel_mae': relative_mae(y_true_sub_phase, mae_sub_phase)
            }
            subscores_metrics_phase.append(sub_metrics)

            print(f"Subscore {i+1} (Phase {phase}): "
                  f"MSE={sub_metrics['mse']:.4f} ({sub_metrics['rel_mse']:.4f}), "
                  f"RMSE={sub_metrics['rmse']:.4f} ({sub_metrics['rel_rmse']:.4f}), "
                  f"MAE={sub_metrics['mae']:.4f} ({sub_metrics['rel_mae']:.4f}), "
                  f"R²={sub_metrics['r2']:.4f}")

        phase_metrics[phase] = {'overall': metrics_phase_overall, 'subscores': subscores_metrics_phase}

    return metrics, subscores_metrics, phase_metrics

def plot_metrics(subscores_metrics, output_dir, project_name, title="Metrics per ALSFRS Subscore"):
    """
    Plots MSE, RMSE, MAE, and R² for each ALSFRS subscore.

    Parameters:
    - subscores_metrics: A list of dictionaries, where each dictionary
                         contains metrics for a single subscore (e.g., from subscores_metrics_overall).
    - title: Title for the overall plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metric_names = ['mse', 'rmse', 'mae', 'r2']
    plot_titles = ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)',
                   'Mean Absolute Error (MAE)', 'R-squared (R²)']

    # Prepare data for plotting
    plot_data = {
        metric: [subscore[metric] for subscore in subscores_metrics]
        for metric in metric_names
    }

    for ax, metric, plot_title in zip(axes.flat, metric_names, plot_titles):
        values = plot_data[metric]
        # Filter out NaN values for plotting if any
        valid_values = [v for v in values if not np.isnan(v)]
        valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]

        if len(valid_values) > 0:
            ax.bar(np.array(valid_indices) + 1, valid_values)  # Add 1 for 1-based subscore indexing
            ax.set_title(plot_title)
            ax.set_xlabel('ALSFRS Subscore')
            ax.set_ylabel(metric.upper())  # Use uppercase for y-axis label
            ax.set_xticks(np.arange(len(subscores_metrics)) + 1)  # Ensure all subscore ticks are shown
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            ax.set_title(f"{plot_title} (No valid data)")
            ax.text(0.5, 0.5, "No data to plot", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.suptitle(title, fontsize=16)  # Overall title for the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle

    fname = f"{output_dir}/{project_name}_subscores_metrics.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_phase_metrics(phase_metrics, output_dir, project_name, metric_to_plot='rmse', title="Model Performance by Phase"):
    """
    Plots a specific metric (e.g., RMSE) for the overall performance across different phases.

    Parameters:
    - phase_metrics: The dictionary returned by evaluate_model containing phase-specific metrics.
    - metric_to_plot: The name of the metric to visualize (e.g., 'mse', 'rmse', 'mae', 'r2').
    - title: Title for the overall plot.
    """
    phases = sorted(phase_metrics.keys())
    metric_values = []

    for phase in phases:
        if 'overall' in phase_metrics[phase] and metric_to_plot in phase_metrics[phase]['overall']:
            metric_values.append(phase_metrics[phase]['overall'][metric_to_plot])
        else:
            metric_values.append(np.nan)  # Append NaN if metric not found for a phase

    # Filter out phases with NaN values for plotting
    valid_phases = [p for i, p in enumerate(phases) if not np.isnan(metric_values[i])]
    valid_metric_values = [v for v in metric_values if not np.isnan(v)]

    if len(valid_phases) == 0:
        print(f"No valid data to plot {metric_to_plot} across phases.")
        return

    plt.figure(figsize=(8, 6))
    plt.bar([str(p) for p in valid_phases], valid_metric_values, color='skyblue')
    plt.xlabel('Phase')
    plt.ylabel(metric_to_plot.upper())
    plt.title(f"{title} ({metric_to_plot.upper()})")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    fname = f"{output_dir}/{project_name}_phase_metrics.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_transitions_by_offset(model, X_test, y_test, sample_weights_test, TimeSinceOnset_test):
    """
    Evaluate model predictions around transition points (0->1 and 1->2) in the test dataset.

    Returns a nested dictionary of metrics per transition type, offset, and subscore.
    """
    y_pred = model.predict(X_test)

    rel_offsets = [-2, -1, 0, 1]
    n_subscores = y_test.shape[-1]

    results = {
        '0->1': {offset: {s: {'y_true': [], 'y_pred': []} for s in range(n_subscores)} for offset in rel_offsets},
        '1->2': {offset: {s: {'y_true': [], 'y_pred': []} for s in range(n_subscores)} for offset in rel_offsets}
    }

    for i in range(TimeSinceOnset_test.shape[0]):  # per patient
        valid_idxs = np.where(sample_weights_test[i] == 1)[0]
        times = TimeSinceOnset_test[i, valid_idxs]

        for t in range(1, len(times)):
            prev_phase, curr_phase = times[t-1], times[t]

            # Transition 0->1
            if prev_phase == 0 and curr_phase == 1:
                trans_type = '0->1'
                center = valid_idxs[t]

            # Transition 1->2
            elif prev_phase == 1 and curr_phase == 2:
                trans_type = '1->2'
                center = valid_idxs[t]
            else:
                continue

            for offset in rel_offsets:
                idx = center + offset
                if 0 <= idx < y_test.shape[1] and sample_weights_test[i, idx] == 1:
                    for s in range(n_subscores):
                        results[trans_type][offset][s]['y_true'].append(y_test[i, idx, s])
                        results[trans_type][offset][s]['y_pred'].append(y_pred[i, idx, s])

    # Compute metrics
    metrics = {}
    for trans in results:
        metrics[trans] = {}
        for offset in results[trans]:
            metrics[trans][offset] = {}
            for s in results[trans][offset]:
                y_t = np.array(results[trans][offset][s]['y_true'])
                y_p = np.array(results[trans][offset][s]['y_pred'])

                if len(y_t) == 0:
                    continue

                mse = mean_squared_error(y_t, y_p)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_t, y_p)
                r2 = r2_score(y_t, y_p)

                metrics[trans][offset][s] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                }

            # Compute average across subscores
            all_metrics = [metrics[trans][offset][s] for s in metrics[trans][offset]]
            if all_metrics:
                avg = {
                    k: np.mean([m[k] for m in all_metrics]) for k in ['MSE', 'RMSE', 'MAE', 'R2']
                }
                metrics[trans][offset]['average'] = avg

    return metrics


def plot_transition_metrics(metrics, output_dir, project_name, metric='RMSE'):
    """
    Visualize a selected metric across transitions, offsets, and subscores.
    """
    rows = []
    for trans in metrics:
        for offset in metrics[trans]:
            for s in metrics[trans][offset]:
                label = f"Q{s+1}" if isinstance(s, int) else s  # subscore index or 'average'
                rows.append({
                    'Transition': trans,
                    'Offset': offset,
                    'Subscore': label,
                    metric: metrics[trans][offset][s][metric]
                })

    df = pd.DataFrame(rows)

    for trans in df['Transition'].unique():
        plt.figure(figsize=(10, 6))
        data = df[df['Transition'] == trans]
        pivot = data.pivot(index='Subscore', columns='Offset', values=metric)
        pivot = pivot.loc[[s for s in pivot.index if s != 'average'] + ['average']]  # avg at bottom

        sns.heatmap(pivot, annot=True, fmt=".3f", cmap='coolwarm', cbar_kws={'label': metric})
        plt.title(f'{metric} by Offset and Subscore ({trans} transition)')
        plt.ylabel("Subscore")
        plt.xlabel("Relative Visit Offset")
        plt.tight_layout()
        fname = f"{output_dir}/{project_name}_transition_metrics_{trans}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()


# Without per subscore evaluation
def evaluate_nonstationarity_window(model, X, y, sample_weights, phases):
    y_pred = model.predict(X)

    rel_offsets = [-2, -1, 0, 1]
    unique_phases = np.unique(phases)
    results = {phase: {offset: {'y_true': [], 'y_pred': []} for offset in rel_offsets} for phase in unique_phases}

    for i in range(X.shape[0]):
        valid_idxs = np.where(sample_weights[i] == 1)[0]

        mid = len(valid_idxs) // 2
        phase = phases[i]

        if len(valid_idxs) < 4:
            print("Patient with less than 5 visits in total!")
            continue # skip short sequences

        for offset in rel_offsets:
            idx = mid + offset
            if 0 <= idx < len(valid_idxs):
                visit_idx = valid_idxs[idx]
                results[phase][offset]['y_true'].append(y[i, visit_idx])
                results[phase][offset]['y_pred'].append(y_pred[i, visit_idx])
            else:
                print("Something wrong with the indices!!!")

    # Compute metrics
    metrics_by_phase = {}
    for phase in unique_phases:
        metrics_by_phase[phase] = {}
        for offset in rel_offsets:
            y_true = np.array(results[phase][offset]['y_true'])
            y_pred = np.array(results[phase][offset]['y_pred'])

            if len(y_true) == 0:
                continue

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            metrics_by_phase[phase][offset] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }

    return metrics_by_phase


def plot_nonstationarity_metrics(metrics_by_phase, output_dir, project_name, metric='RMSE'):
    """
    Print and visualize metrics over relative visit offsets by phase.

    Parameters:
    - metrics_by_phase: dict from evaluate_nonstationarity_window()
    - metric_to_plot: one of 'MSE', 'RMSE', 'MAE', 'R2'
    """
    print(f"\n==== Nonstationarity Metrics by Phase and Visit Offset ({metric}) ====\n")

    rows = []
    for phase in sorted(metrics_by_phase.keys()):
        for offset in sorted(metrics_by_phase[phase].keys()):
            metrics = metrics_by_phase[phase][offset]
            rows.append({
                'Phase': phase,
                'Offset': offset,
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2']
            })
            print(f"Phase {phase}, Offset {offset}: "
                  f"MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
                  f"MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")

    # Convert to DataFrame for visualization
    df = pd.DataFrame(rows)

    # Create pivot table for heatmap
    pivot = df.pivot(index='Phase', columns='Offset', values=metric)

    # Plot heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap='coolwarm', cbar_kws={'label': metric})
    plt.title(f'{metric} Across Visit Offsets by Phase')
    plt.ylabel("Phase")
    plt.xlabel("Relative Visit Offset")
    plt.tight_layout()
    fname = f"{output_dir}/{project_name}_transition_metrics.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_nonstationarity_window_per_score(model, X, y, sample_weights, phases):
    """
    Compute evaluation metrics (MSE, RMSE, MAE, R2) for each ALSFRS subscore,
    at relative visit offsets (-2, -1, 0, 1), stratified by disease phase.

    Parameters:
    - model: trained model with .predict(X) returning shape (n_samples, n_visits, n_subscores)
    - X: input array of shape (n_samples, n_visits, n_features)
    - y: true labels of shape (n_samples, n_visits, n_subscores)
    - sample_weights: binary mask of shape (n_samples, n_visits)
    - phases: list/array of shape (n_samples,) indicating the phase of each patient

    Returns:
    - metrics_by_phase: nested dict [phase][offset][subscore] → metric dict
    """
    y_pred = model.predict(X)

    rel_offsets = [-2, -1, 0, 1]
    unique_phases = np.unique(phases)
    n_subscores = y.shape[-1]

    # Store true and predicted values per offset, phase, and subscore
    data = {
        phase: {
            offset: {s: {'y_true': [], 'y_pred': []} for s in range(n_subscores)}
            for offset in rel_offsets
        }
        for phase in unique_phases
    }

    for i in range(X.shape[0]):
        valid_idxs = np.where(sample_weights[i] == 1)[0]

        mid = len(valid_idxs) // 2
        phase = phases[i]

        if len(valid_idxs) < 4:
            print("Patient with less than 5 visits in total!")
            continue # skip short sequences

        for offset in rel_offsets:
            idx = mid + offset
            if 0 <= idx < len(valid_idxs):
                visit_idx = valid_idxs[idx]
                for s in range(n_subscores):
                    data[phase][offset][s]['y_true'].append(y[i, visit_idx, s])
                    data[phase][offset][s]['y_pred'].append(y_pred[i, visit_idx, s])
            else:
                print("Something wrong with the indices!!!")

    # Compute metrics
    results = {
        phase: {
            offset: {} for offset in rel_offsets
        }
        for phase in unique_phases
    }
    for phase in unique_phases:
        for offset in rel_offsets:
            for s in range(n_subscores):
                y_true = np.array(data[phase][offset][s]['y_true'])
                y_pred = np.array(data[phase][offset][s]['y_pred'])

                if len(y_true) == 0:
                    continue

                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                results[phase][offset][s] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                }

    return results


def plot_nonstationarity_metrics_per_score(metrics_by_phase, output_dir, project_name, metric='RMSE'):
    """
    Visualize and print per-subscore metrics over visit offsets and phases.

    Parameters:
    - metrics_by_phase: output of evaluate_nonstationarity_window()
    - metric_to_plot: which metric to visualize ('MSE', 'RMSE', 'MAE', 'R2')
    - subscore_names: list of subscore names (optional, defaults to Q1–Qn)
    """
    print(f"\n==== Nonstationarity Metrics by Phase and Visit Offset ({metric}) ====\n")

    all_rows = []
    n_subscores = max(max(max(d.keys()) for d in offsets.values()) for offsets in metrics_by_phase.values()) + 1
    subscore_names = [f"Q{i+1}" for i in range(n_subscores)]

    for phase in sorted(metrics_by_phase.keys()):
        for offset in sorted(metrics_by_phase[phase].keys()):
            for s in sorted(metrics_by_phase[phase][offset].keys()):
                metrics = metrics_by_phase[phase][offset][s]
                row = {
                    'Phase': phase,
                    'Offset': offset,
                    'Subscore': subscore_names[s],
                    'MSE': metrics['MSE'],
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'R2': metrics['R2']
                }
                all_rows.append(row)
                # print(f"Phase {phase}, Offset {offset}, {subscore_names[s]} → "
                #       f"MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
                #       f"MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")

    df = pd.DataFrame(all_rows)

    # Plot: One heatmap per subscore
    n_cols = 4
    n_rows = int(np.ceil(n_subscores / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)

    for i, subscore in enumerate(subscore_names):
        ax = axes[i // n_cols][i % n_cols]
        pivot = df[df['Subscore'] == subscore].pivot(index='Phase', columns='Offset', values=metric)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap='coolwarm', ax=ax, cbar=False)
        ax.set_title(f"{subscore} ({metric})")
        ax.set_xlabel("Offset")
        ax.set_ylabel("Phase")

    # Hide any empty subplots
    for j in range(i+1, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    plt.tight_layout()
    fname = f"{output_dir}/{project_name}_transition_metrics_per_score.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def save_json(obj, path):
    def convert(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, (bytes, bytearray)):
            return o.decode("utf-8", errors="ignore")
        return str(o)

    def sanitize_keys(o):
        if isinstance(o, dict):
            return {str(convert(k)): sanitize_keys(v) for k, v in o.items()}
        elif isinstance(o, (list, tuple)):
            return [sanitize_keys(v) for v in o]
        else:
            return convert(o)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(sanitize_keys(obj), f, indent=4)
