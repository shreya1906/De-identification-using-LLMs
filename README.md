# De-identification-using-LLMs

## Objective

This research project focuses on utilizing Large Language Models (LLMs) to remove personally identifiable information (PII) from forum posts. The project explores the effectiveness of both OpenAI's GPT-4 model and Meta's LLaMA3 model (accessed via the Fireworks platform) in de-identifying sensitive data.

## Project Structure

### Data

- **original_files**: Contains the original data with PII.
- **human_redacted_files**: Contains the files that have been de-identified by humans.
- **results**:
  - **OpenAI_redacted_files**: Files generated by the OpenAI GPT-4 model.
  - **LLaMA_redacted_files**: Files generated by the LLaMA3 model accessed via Fireworks.
- **prompts.csv**: A CSV file containing a list of prompts used during the de-identification process and for evaluation.

### Code Files

- **de_identified_csv_generator.py**: This script processes the original files and generates de-identified CSV files using either the OpenAI GPT-4 model or the LLaMA model accessed via Fireworks, based on user selection.
- **de_identified_csv_evaluator.py**: This script evaluates the reliability of the de-identification process by comparing the model outputs with human-redacted files. It calculates metrics such as accuracy, precision, recall, and Cohen's kappa.

## Prerequisites

Before getting started, ensure you have the following:

- **Python Environment**: Python 3.6 or higher is recommended.
- **API Keys**:
  - **OpenAI API Key**: Obtain an API key from OpenAI to access the GPT-4 model.
  - **Fireworks API Key**: Obtain an API key from Fireworks to access the LLaMA model.
- **Python Packages**: Install the required packages using pip:

  ```bash
  pip install openai fireworks pandas
  pip install --upgrade fireworks-ai
  ```

- **Input Data**:
  - **original_files**: Place your original CSV files containing PII in this folder.
  - **human_redacted_files**: Place the corresponding human-redacted CSV files in this folder.

## Getting Started

Here's how to initiate the project:

**Step 1:** Organize Data
Place your original files with PII in the original_files folder and human redacted files in human_redacted_files folder. **We have artificially created some sample files for your reference in the folder.**

**Step 2:** Run the `de_identified_csv_generator.py`
Execute the de_identified_csv_generator script, providing the necessary input files and the output folder (which will be automatically created):
This script will process the files, remove PII, and generate an OpenAI de-identified CSV file within an OpenAI_redacted_files folder inside the results folder.

**Step 3:** Run the `de_identified_csv_evaluator.py`
To evaluate the accuracy of the de-identification process, run the De-identified CSV Evaluator script:
This script will analyze the de-identified CSV files in the results folder and update the metrics csv with accuracy, precision, recall and kappa values. 


