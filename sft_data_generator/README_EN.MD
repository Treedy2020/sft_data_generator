# SFT Data Generator

This is a tool for generating data for Supervised Fine-Tuning (SFT) using large language models like GPT-3.5. It can read data from CSV or Excel files and generate corresponding output data using OpenAI's API based on a given prompt.

## Features

- Supports reading CSV and Excel files
- Uses OpenAI's Chat Completion API to generate data
- Supports customizable prompts and models
- Supports batch generation with customizable batch size
- Supports setting the number of generation epochs, i.e., generating multiple rounds for the entire dataset
- Supports JSON format output
- Can set OpenAI API URL and key via command line arguments or environment variables

## Usage

1. Install dependencies:
   
   ```
   pip install -r requirements.txt
   ```

2. Prepare the input data file (CSV or Excel) and prompt template.

3. Run the command:

   ```
   python data_generator.py --file_path <input_file_path> --prompt <prompt_template> --model <model_name> [other optional arguments]
   ```

   Required arguments:
   - `--file_path`: Path to the input data file, must be a CSV or Excel file.
   - `--prompt`: Prompt template used for generating data.
   - `--model`: Name of the OpenAI model used for generating data, e.g., `gpt-3.5-turbo`.

   Optional arguments:
   - `--output_file`: Output file path, defaults to input file path + `.output.jsonl`.
   - `--batch_size`: Batch size, i.e., the number of samples per concurrent request, defaults to 1.
   - `--generate_epoch`: Number of generation epochs, i.e., the number of rounds to generate for the entire dataset, defaults to 1.
   - `--openai_base_url`: Base URL for the OpenAI API, if not given, will use the environment variable `OPENAI_API_BASE`.
   - `--openai_api_key`: OpenAI API key, if not given, will use the environment variable `OPENAI_API_KEY`.
   - `--json_output`: Whether to use JSON format output, defaults to False.

4. The generated data will be saved in the specified output file, with each line being a JSON object.

## Example

```
python data_generator.py --file_path data.csv --prompt "Please generate a question-answer pair based on the given data:" --model gpt-3.5-turbo --batch_size 10 --generate_epoch 3
```

This will read the `data.csv` file, use the prompt "Please generate a question-answer pair based on the given data:", call the `gpt-3.5-turbo` model with a batch size of 10, and generate 3 epochs of data. The output will be saved in the `data.csv.output.jsonl` file.