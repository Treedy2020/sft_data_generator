好的,以下是关于这个工具的中文和英文README:

---

# SFT数据生成工具

这是一个使用大语言模型(如GPT-3.5)为监督微调(Supervised Fine-Tuning, SFT)生成数据的工具。它可以读取CSV或Excel文件中的数据,并根据给定的提示(prompt)使用OpenAI的API生成对应的输出数据。

## 功能特点

- 支持读取CSV和Excel文件
- 使用OpenAI的Chat Completion API生成数据
- 支持自定义提示(prompt)和模型
- 支持批量生成,可自定义批大小
- 支持设置生成轮数,即对整个数据集生成多轮
- 支持JSON格式输出
- 可通过命令行参数或环境变量设置OpenAI API的URL和密钥

## 使用方法

1. 安装依赖包:
   
   ```
   pip install -r requirements.txt
   ```

2. 准备输入数据文件(CSV或Excel)和提示模板。

3. 运行命令:

   ```
   python data_generator.py --file_path <输入文件路径> --prompt <提示模板> --model <模型名称> [其他可选参数]
   ```

   必需参数:
   - `--file_path`: 输入数据文件的路径,必须是CSV或Excel文件。
   - `--prompt`: 用于生成数据的提示模板。
   - `--model`: 用于生成数据的OpenAI模型名称,如`gpt-3.5-turbo`。

   可选参数:
   - `--output_file`: 输出文件路径,默认为输入文件路径+`.output.jsonl`。
   - `--batch_size`: 批大小,即每次并发请求的样本数,默认为1。
   - `--generate_epoch`: 生成轮数,即对整个数据集生成的轮次,默认为1。
   - `--openai_base_url`: OpenAI API的基础URL,如果未给定,将使用环境变量`OPENAI_API_BASE`。
   - `--openai_api_key`: OpenAI API密钥,如果未给定,将使用环境变量`OPENAI_API_KEY`。
   - `--json_output`: 是否使用JSON格式输出(指定chat.completion.create的参数 json_output为 {"type": "json_object"}), 默认为False。

4. 生成的数据将保存在指定的输出文件中,每行为一个JSON对象。

## 示例

```
python data_generator.py --file_path data.csv --prompt "请根据给定的数据生成一个问答:" --model gpt-3.5-turbo --batch_size 10 --generate_epoch 3
```

这将读取`data.csv`文件,使用提示"请根据给定的数据生成一个问答:",调用`gpt-3.5-turbo`模型,批大小为10,生成3轮数据。输出将保存在`data.csv.output.jsonl`文件中。

---