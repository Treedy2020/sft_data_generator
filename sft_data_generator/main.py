# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Richard_Cui on 2024/5/15 10:45.
import os
import asyncio
import sys

import filetype
import argparse
import logging
from typing import Optional

import jsonlines
import openai
from openai import AsyncClient
import pandas as pd
from openai._types import NOT_GIVEN
from tqdm import tqdm

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger.addHandler(handler)


class DataGenerator:
    """To Define and Generate the Data Class for the SFT by LLM."""

    def __init__(self, file_path: str,
                 prompt: str,
                 model: str,
                 output_file: Optional[str] = None,
                 batch_size: int = 10,
                 json_output: bool = False,
                 generate_epoch: int = 1,
                 openai_base_url: Optional[str] = None,
                 openai_api_key: Optional[str] = None
                 ):

        kind = filetype.guess(file_path)
        assert kind is not None and kind.extension in ['csv', 'xlsx'], \
            f"File path should be a csv or xlsx file, but got {kind.extension} file."
        if kind.extension == 'csv':
            self.data = pd.read_csv(file_path)
        else:
            self.data = pd.read_excel(file_path)

        self.output_file = output_file if output_file is not None else file_path + '.output.jsonl'
        self.client = AsyncClient(
            base_url=openai_base_url if openai_base_url is not None else os.environ.get("OPENAI_API_BASE"),
            api_key=openai_api_key if openai_api_key is not None else os.environ.get("OPENAI_API_KEY")
        )
        self.prompt = prompt
        self.model = model
        self.batch_size = batch_size
        self.json_output = {"type": "json_object"} if json_output else NOT_GIVEN
        self.generate_epoch = generate_epoch

    # TODO Add the retry decorator for the get_data_sample function.
    async def get_data_sample(self, row_data: dict) -> dict | None:
        messages = [{"role": "system", "content": self.prompt},
                    {"role": "user", "content": f"Please Generate data follow the given data: {row_data}"}
                    ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=self.json_output
            )
            full_response = response.choices[0].message.content
            row_data['gpt-response'] = full_response
            return row_data
        except openai.OpenAIError as e:
            logger.error(f"Error happened when trying to get openai response, check the RPM/TPM or account balance if "
                         f"you meet this error: {e}")
            return None

    async def main_loop(self) -> None:
        with jsonlines.open(self.output_file, 'w') as f:
            for e in tqdm(range(self.generate_epoch), desc="Main Epoch Loop", position=0):
                for batch_start in tqdm(range(0, len(self.data), self.batch_size), position=1, desc="Processing Epoch",
                                        leave=False):
                    tasks = [asyncio.create_task(self.get_data_sample(data.to_dict())) for _, data in
                             self.data[batch_start: batch_start + self.batch_size].iterrows()]
                    result_ = await asyncio.gather(*tasks)
                    result_ = [x for x in result_ if x is not None]
                    for generate_data in result_:
                        f.write(generate_data)
        logger.info(f"Data Generation Finished, the output file is saved as {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description="Data Generator for the SFT by LLM.")

    parser.add_argument("--file_path", type=str, required=True, help="file path of the input data.")
    parser.add_argument("--prompt", type=str, required=True, help="prompt for the data generation.")
    parser.add_argument("--model", type=str, required=True, help="OpenAI model for the data generation.")

    parser.add_argument("--output_file", type=str, required=False,
                        help="[Optional] output file path for the data generation.")
    parser.add_argument("--batch_size", type=int, required=False, default=1,
                        help="[Optional] Batch size for the data generation, default as 1.")

    parser.add_argument("--generate_epoch", type=int, required=False, default=1,
                        help="generate epoch for the whole table file, default as 1.")

    parser.add_argument("--openai_base_url", required=False, type=str,
                        help="OpenAI API base url, If not given, will use the env "
                             "variable.")

    parser.add_argument("--openai_api_key", required=False, type=str,
                        help="OpenAI API key, If not given, will use the env "
                             "variable.")

    parser.add_argument("--json_output", required=False, action="store_true",
                        help="[Optional] Response format for the chat completion API of OpenAI.")

    if len(sys.argv) == 1:
        sys.argv.append('--help')

    args = parser.parse_args()

    data_generator = DataGenerator(**dict(vars(args)))
    asyncio.run(data_generator.main_loop())


if __name__ == '__main__':
    main()
