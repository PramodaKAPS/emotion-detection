from datasets import load_dataset
import pandas as pd
import os
from imblearn.over_sampling import RandomOverSampler
from transformers import AutoTokenizer, DataCollatorWithPadding, TFDistilBertForSequenceClassification, create_optimizer
import tensorflow as tf

# Clear the cache for the 'go_emotions' dataset
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets", "go_emotions")
if os.path.exists(cache_dir):
    print(f"Clearing cache directory: {cache_dir}")
    os.system(f"rm -rf {cache_dir}")

# Load the GoEmotions dataset
try:
    dataset = load_dataset("go_emotions")  # Load without 'simplified'

    # Define the 10 emotions
    selected_emotions = ["anger", "sadness", "joy", "disgust", "fear", "surprise", "gratitude", "remorse", "curiosity", "neutral"]

    # Convert to DataFrame for easier filtering
    train_df = pd.DataFrame(dataset["train"])
    valid_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])

    # Filter for the selected emotions
    emotion_mapping = dataset["train"].features["labels"].feature.names
    selected_indices = [emotion_mapping.index(emotion) for emotion in selected_emotions]

    # Function to filter rows with at least one of the selected emotions
    def filter_emotions(df):
        df["labels"] = df["labels"].apply(lambda x: [label for label in x if label in selected_indices])
        df = df[df["labels"].apply(len) > 0]  # Keep rows with at least one selected label
        return df

    # Apply filtering
    train_df = filter_emotions(train_df)
    valid_df = filter_emotions(valid_df)
    test_df = filter_emotions(test_df)

    # Convert labels to single-label format (take the first matching label)
    train_df["label"] = train_df["labels"].apply(lambda x: x[0])
    valid_df["label"] = valid_df["labels"].apply(lambda x: x[0])
    test_df["label"] = test_df["labels"].apply(lambda x: x[0])

    # Drop the original 'labels' column
    train_df = train_df[["text", "label"]]
    valid_df = valid_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    print("Filtered training data shape:", train_df.shape)
    print("Filtered validation data shape:", valid_df.shape)
    print("Filtered test data shape:", test_df.shape)

    # Oversample the training data
    X_train = train_df["text"].values.reshape(-1, 1)
    y_train = train_df["label"]
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    emotions_train = pd.DataFrame({"text": X_resampled.flatten(), "label": y_resampled})
    print("Class distribution after oversampling:")
    print(emotions_train["label"].value_counts())

    # Load the tokenizer
    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Tokenization function
    def tokenize_function(example):
        return tokenizer(example["text"], padding=True, truncation=True)

    # Convert DataFrames to Dataset format for tokenization
    train_dataset = Dataset.from_pandas(emotions_train)
    valid_dataset = Dataset.from_pandas(valid_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Tokenize the datasets
    tokenized_datasets_train = train_dataset.map(tokenize_function, batched=True, batch_size=None)
    tokenized_datasets_val = valid_dataset.map(tokenize_function, batched=True, batch_size=None)
    tokenized_datasets_test = test_dataset.map(tokenize_function, batched=True, batch_size=None)

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    # Define the label mapping
    emotion_mapping_dict = {idx: i for i, idx in enumerate(selected_indices)}  # Map original indices to 0-9

    # Remap labels in the tokenized datasets
    tokenized_datasets_train = tokenized_datasets_train.map(lambda x: {"label": emotion_mapping_dict[x["label"]]})
    tokenized_datasets_val = tokenized_datasets_val.map(lambda x: {"label": emotion_mapping_dict[x["label"]]})
    tokenized_datasets_test = tokenized_datasets_test.map(lambda x: {"label": emotion_mapping_dict[x["label"]]})

    # Convert to TensorFlow datasets
    tf_train_dataset = tokenized_datasets_train.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["label"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_dataset = tokenized_datasets_val.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["label"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_dataset = tokenized_datasets_test.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["label"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    # Load the model with 10 labels
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=10)

    # Define optimizer and learning rate schedule
    batch_size = 16
    num_epochs = 3
    num_train_steps = len(tf_train_dataset) * num_epochs
    optimizer, schedule = create_optimizer(
        init_lr=2e-5,
        num_warmup_steps=0,
        num_train_steps=num_train_steps,
    )

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Train the model
    model.fit(
        tf_train_dataset,
        validation_data=tf_validation_dataset,
        epochs=num_epochs,
    )

    # Evaluate on the test set
    test_results = model.evaluate(tf_test_dataset)
    print("Test set results:", test_results)

    # Define the path to save the model
    save_path = "/root/emotion_model"

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")

except ValueError as e:
    print(f"An error occurred: {e}")
    print("If the error persists, there might be an issue with the dataset on the Hugging Face Hub.")
