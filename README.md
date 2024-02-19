# QLoRA for CodeLlama

This repository is a fork of the [poludmik/qlora_for_codegen](https://github.com/poludmik/qlora_for_codegen) repository.  This is a development repository where we conduct experiments using code LLMs.

Example command to fine-tune:

`
$ python3 qlora.py --model_name_or_path checkpoints/codellama/CodeLlama-70b-Instruct-hf/ --ddp_find_unused_parameters False --bf16
`

