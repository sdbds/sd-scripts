# Training Stable Cascade Stage C 

This is an experimental feature. There may be bugs.

__Feb 25, 2024 Update:__  Fixed a bug that the LoRA weights trained can be loaded in ComfyUI. If you still have a problem, please let me know.

__Feb 25, 2024 Update:__ Fixed a bug that Stage C training with mixed precision behaves the same as `--full_bf16` (fp16) regardless of `--full_bf16` (fp16) specified. 

This is because the Stage C weights were loaded in bf16/fp16. With this fix, the memory usage without `--full_bf16` (fp16) specified will increase, so you may need to specify `--full_bf16` (fp16) as needed.

__Feb 22, 2024 Update:__ Fixed a bug that LoRA is not applied to some modules (to_q/k/v and to_out) in Attention. Also, the model structure of Stage C has been changed, and you can choose xformers and SDPA (SDPA was used before). Please specify `--sdpa` or `--xformers` option.

__Feb 20, 2024 Update:__ There was a problem with the preprocessing of the EfficientNetEncoder, and the latents became invalid (the saturation of the training results decreases). If you have cached `_sc_latents.npz` files with `--cache_latents_to_disk`, please delete them before training.

## Usage

Training is run with `stable_cascade_train_stage_c.py`.

The main options are the same as `sdxl_train.py`. The following options have been added.

- `--effnet_checkpoint_path`: Specifies the path to the EfficientNetEncoder weights.
- `--stage_c_checkpoint_path`: Specifies the path to the Stage C weights.
- `--text_model_checkpoint_path`: Specifies the path to the Text Encoder weights. If omitted, the model from Hugging Face will be used.
- `--save_text_model`: Saves the model downloaded from Hugging Face to `--text_model_checkpoint_path`.
- `--previewer_checkpoint_path`: Specifies the path to the Previewer weights. Used to generate sample images during training.
- `--adaptive_loss_weight`: Uses [Adaptive Loss Weight](https://github.com/Stability-AI/StableCascade/blob/master/gdf/loss_weights.py) . If omitted, P2LossWeight is used. The official settings use Adaptive Loss Weight.

The learning rate is set to 1e-4 in the official settings.

The first time, specify `--text_model_checkpoint_path` and `--save_text_model` to save the Text Encoder weights. From the next time, specify `--text_model_checkpoint_path` to load the saved weights.

Sample image generation during training is done with Perviewer. Perviewer is a simple decoder that converts EfficientNetEncoder latents to images.

Some of the options for SDXL are simply ignored or cause an error (especially noise-related options such as `--noise_offset`). `--vae_batch_size` and `--no_half_vae` are applied directly to the EfficientNetEncoder (when `bf16` is specified for mixed precision, `--no_half_vae` is not necessary).

Options for latents and Text Encoder output caches can be used as is, but since the EfficientNetEncoder is much lighter than the VAE, you may not need to use the cache unless memory is particularly tight.

`--gradient_checkpointing`, `--full_bf16`, and `--full_fp16` (untested) to reduce memory consumption can be used as is.

A scale of about 4 is suitable for sample image generation.

Since the official settings use `bf16` for training, training with `fp16` may be unstable.

The code for training the Text Encoder is also written, but it is untested.

### Command line sample

```batch
accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 stable_cascade_train_stage_c.py --mixed_precision bf16 --save_precision bf16 --max_data_loader_n_workers 2 --persistent_data_loader_workers --gradient_checkpointing --learning_rate 1e-4 --optimizer_type adafactor --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False" --max_train_epochs 10 --save_every_n_epochs 1 --save_precision bf16 --output_dir ../output --output_name sc_test - --stage_c_checkpoint_path ../models/stage_c_bf16.safetensors --effnet_checkpoint_path ../models/effnet_encoder.safetensors --previewer_checkpoint_path ../models/previewer.safetensors --dataset_config ../dataset/config_bs1.toml --sample_every_n_epochs 1 --sample_prompts ../dataset/prompts.txt --adaptive_loss_weight
```

### About the dataset for fine tuning

If the latents cache files for SD/SDXL exist (extension `*.npz`), it will be read and an error will occur during training. Please move them to another location in advance.

After that, run `finetune/prepare_buckets_latents.py` with the `--stable_cascade` option to create latents cache files for Stable Cascade (suffix `_sc_latents.npz` is added).

## LoRA training

`stable_cascade_train_c_network.py` is used for LoRA training. The main options are the same as `train_network.py`, and the same options as `stable_cascade_train_stage_c.py` have been added.

__This is an experimental feature, so the format of the saved weights may change in the future and become incompatible.__

There is no compatibility with the official LoRA, and the implementation of Text Encoder embedding training (Pivotal Tuning) in the official implementation is not implemented here.

Text Encoder LoRA training is implemented, but untested.

## Image generation

Basic image generation functionality is available in `stable_cascade_gen_img.py`. See `--help` for usage.

When using LoRA, specify `--network_module networks.lora --network_mul 1 --network_weights lora_weights.safetensors`.

The following prompt options are available.

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.
  * `--t` Specifies the t_start of the generation.
  * `--f` Specifies the shift of the generation.

# Stable Cascade Stage C の学習

実験的機能です。不具合があるかもしれません。

__2024/2/25 追記:__ 学習される LoRA の重みが ComfyUI で読み込めるよう修正しました。依然として不具合がある場合にはご連絡ください。

__2024/2/25 追記:__ Mixed precision 時のStage C の学習が、 `--full_bf16` (fp16) の指定に関わらず `--full_bf16` (fp16) 指定時と同じ動作となる（と思われる）不具合を修正しました。

Stage C の重みを bf16/fp16 で読み込んでいたためです。この修正により `--full_bf16` (fp16) 未指定時のメモリ使用量が増えますので、必要に応じて `--full_bf16` (fp16) を指定してください。

__2024/2/22 追記:__ LoRA が一部のモジュール（Attention の to_q/k/v および to_out）に適用されない不具合を修正しました。また Stage C のモデル構造を変更し xformers と SDPA を選べるようになりました（今までは SDPA が使用されていました）。`--sdpa` または `--xformers` オプションを指定してください。

__2024/2/20 追記:__ EfficientNetEncoder の前処理に不具合があり、latents が不正になっていました（学習結果の彩度が低下する現象が起きます）。`--cache_latents_to_disk` でキャッシュした `_sc_latents.npz` がある場合、いったん削除してから学習してください。

## 使い方

学習は `stable_cascade_train_stage_c.py` で行います。

主なオプションは `sdxl_train.py` と同様です。以下のオプションが追加されています。

- `--effnet_checkpoint_path` : EfficientNetEncoder の重みのパスを指定します。
- `--stage_c_checkpoint_path` : Stage C の重みのパスを指定します。
- `--text_model_checkpoint_path` : Text Encoder の重みのパスを指定します。省略時は Hugging Face のモデルを使用します。
- `--save_text_model` : `--text_model_checkpoint_path` にHugging Face からダウンロードしたモデルを保存します。
- `--previewer_checkpoint_path` : Previewer の重みのパスを指定します。学習中のサンプル画像生成に使用します。
- `--adaptive_loss_weight` :  [Adaptive Loss Weight](https://github.com/Stability-AI/StableCascade/blob/master/gdf/loss_weights.py) を用います。省略時は P2LossWeight が使用されます。公式では Adaptive Loss Weight が使用されているようです。

学習率は、公式の設定では 1e-4 のようです。

初回は `--text_model_checkpoint_path` と `--save_text_model` を指定して、Text Encoder の重みを保存すると良いでしょう。次からは `--text_model_checkpoint_path` を指定して、保存した重みを読み込むことができます。

学習中のサンプル画像生成は Perviewer で行われます。Previewer は EfficientNetEncoder の latents を画像に変換する簡易的な decoder です。

SDXL の向けの一部のオプションは単に無視されるか、エラーになります（特に `--noise_offset` などのノイズ関係）。`--vae_batch_size` および `--no_half_vae` はそのまま EfficientNetEncoder に適用されます（mixed precision に `bf16` 指定時は `--no_half_vae` は不要のようです）。

latents および Text Encoder 出力キャッシュのためのオプションはそのまま使用できますが、EfficientNetEncoder は VAE よりもかなり軽量のため、メモリが特に厳しい場合以外はキャッシュを使用する必要はないかもしれません。

メモリ消費を抑えるための `--gradient_checkpointing` 、`--full_bf16`、`--full_fp16`（未テスト）はそのまま使用できます。

サンプル画像生成時の Scale には 4 程度が適しているようです。

公式の設定では学習に `bf16` を用いているため、`fp16` での学習は不安定かもしれません。

Text Encoder 学習のコードも書いてありますが、未テストです。

### コマンドラインのサンプル

[Command-line-sample](#command-line-sample)を参照してください。


###  fine tuning方式のデータセットについて

SD/SDXL 向けの latents キャッシュファイル（拡張子 `*.npz`）が存在するとそれを読み込んでしまい学習時にエラーになります。あらかじめ他の場所に退避しておいてください。

その後、`finetune/prepare_buckets_latents.py` をオプション `--stable_cascade` を指定して実行すると、Stable Cascade 向けの latents キャッシュファイル（接尾辞 `_sc_latents.npz` が付きます）が作成されます。


## LoRA 等の学習

LoRA の学習は `stable_cascade_train_c_network.py` で行います。主なオプションは `train_network.py` と同様で、`stable_cascade_train_stage_c.py` と同様のオプションが追加されています。

__実験的機能のため、保存される重みのフォーマットは将来的に変更され、互換性がなくなる可能性があります。__

公式の LoRA と重みの互換性はありません。また公式で実装されている Text Encoder の embedding 学習（Pivotal Tuning）も実装されていません。

Text Encoder の LoRA 学習は実装してありますが、未テストです。

## 画像生成

最低限の画像生成機能が `stable_cascade_gen_img.py` にあります。使用法は `--help` を参照してください。

LoRA 使用時は `--network_module networks.lora --network_mul 1 --network_weights lora_weights.safetensors` のように指定します。

プロンプトオプションとして以下が使用できます。

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.
  * `--t` Specifies the t_start of the generation.
  * `--f` Specifies the shift of the generation.


---  

__SDXL is now supported. The sdxl branch has been merged into the main branch. If you update the repository, please follow the upgrade instructions. Also, the version of accelerate has been updated, so please run accelerate config again.__ The documentation for SDXL training is [here](./README.md#sdxl-training).

This repository contains training, generation and utility scripts for Stable Diffusion.

[__Change History__](#change-history) is moved to the bottom of the page. 
更新履歴は[ページ末尾](#change-history)に移しました。

[日本語版READMEはこちら](./README-ja.md)

For easier use (GUI and PowerShell scripts etc...), please visit [the repository maintained by bmaltais](https://github.com/bmaltais/kohya_ss). Thanks to @bmaltais!

This repository contains the scripts for:

* DreamBooth training, including U-Net and Text Encoder
* Fine-tuning (native training), including U-Net and Text Encoder
* LoRA training
* Textual Inversion training
* Image generation
* Model conversion (supports 1.x and 2.x, Stable Diffision ckpt/safetensors and Diffusers)

## About requirements.txt

These files do not contain requirements for PyTorch. Because the versions of them depend on your environment. Please install PyTorch at first (see installation guide below.) 

The scripts are tested with Pytorch 2.0.1. 1.12.1 is not tested but should work.

## Links to usage documentation

Most of the documents are written in Japanese.

[English translation by darkstorm2150 is here](https://github.com/darkstorm2150/sd-scripts#links-to-usage-documentation). Thanks to darkstorm2150!

* [Training guide - common](./docs/train_README-ja.md) : data preparation, options etc... 
  * [Chinese version](./docs/train_README-zh.md)
* [Dataset config](./docs/config_README-ja.md) 
* [DreamBooth training guide](./docs/train_db_README-ja.md)
* [Step by Step fine-tuning guide](./docs/fine_tune_README_ja.md):
* [training LoRA](./docs/train_network_README-ja.md)
* [training Textual Inversion](./docs/train_ti_README-ja.md)
* [Image generation](./docs/gen_img_README-ja.md)
* note.com [Model conversion](https://note.com/kohya_ss/n/n374f316fe4ad)

## Windows Required Dependencies

Python 3.10.6 and Git:

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

## Windows Installation

Open a regular Powershell terminal and type the following inside:

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.20

accelerate config
```

__Note:__ Now bitsandbytes is optional. Please install any version of bitsandbytes as needed. Installation instructions are in the following section.

<!-- 
cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py
-->
Answers to accelerate config:

```txt
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

note: Some user reports ``ValueError: fp16 mixed precision requires a GPU`` is occurred in training. In this case, answer `0` for the 6th question: 
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:`` 

(Single GPU with id `0` will be used.)

### Optional: Use `bitsandbytes` (8bit optimizer)

For 8bit optimizer, you need to install `bitsandbytes`. For Linux, please install `bitsandbytes` as usual (0.41.1 or later is recommended.)

For Windows, there are several versions of `bitsandbytes`:

- `bitsandbytes` 0.35.0: Stable version. AdamW8bit is available. `full_bf16` is not available.
- `bitsandbytes` 0.41.1: Lion8bit, PagedAdamW8bit and PagedLion8bit are available. `full_bf16` is available.

Note: `bitsandbytes`above 0.35.0 till 0.41.0 seems to have an issue: https://github.com/TimDettmers/bitsandbytes/issues/659

Follow the instructions below to install `bitsandbytes` for Windows.

### bitsandbytes 0.35.0 for Windows

Open a regular Powershell terminal and type the following inside:

```powershell
cd sd-scripts
.\venv\Scripts\activate
pip install bitsandbytes==0.35.0

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py
```

This will install `bitsandbytes` 0.35.0 and copy the necessary files to the `bitsandbytes` directory.

### bitsandbytes 0.41.1 for Windows

Install the Windows version whl file from [here](https://github.com/jllllll/bitsandbytes-windows-webui) or other sources, like:

```powershell
python -m pip install bitsandbytes==0.41.1 --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

## Credits

The implementation for LoRA is based on [cloneofsimo's repo](https://github.com/cloneofsimo/lora). Thank you for great work!

The LoRA expansion to Conv2d 3x3 was initially released by cloneofsimo and its effectiveness was demonstrated at [LoCon](https://github.com/KohakuBlueleaf/LoCon) by KohakuBlueleaf. Thank you so much KohakuBlueleaf!

## License

The majority of scripts is licensed under ASL 2.0 (including codes from Diffusers, cloneofsimo's and LoCon), however portions of the project are available under separate license terms:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause


## SDXL training

The documentation in this section will be moved to a separate document later.

### Training scripts for SDXL

- `sdxl_train.py` is a script for SDXL fine-tuning. The usage is almost the same as `fine_tune.py`, but it also supports DreamBooth dataset.
  - `--full_bf16` option is added. Thanks to KohakuBlueleaf!
    - This option enables the full bfloat16 training (includes gradients). This option is useful to reduce the GPU memory usage. 
    - The full bfloat16 training might be unstable. Please use it at your own risk.
  - The different learning rates for each U-Net block are now supported in sdxl_train.py. Specify with `--block_lr` option. Specify 23 values separated by commas like `--block_lr 1e-3,1e-3 ... 1e-3`.
    - 23 values correspond to `0: time/label embed, 1-9: input blocks 0-8, 10-12: mid blocks 0-2, 13-21: output blocks 0-8, 22: out`.
- `prepare_buckets_latents.py` now supports SDXL fine-tuning.

- `sdxl_train_network.py` is a script for LoRA training for SDXL. The usage is almost the same as `train_network.py`.

- Both scripts has following additional options:
  - `--cache_text_encoder_outputs` and `--cache_text_encoder_outputs_to_disk`: Cache the outputs of the text encoders. This option is useful to reduce the GPU memory usage. This option cannot be used with options for shuffling or dropping the captions.
  - `--no_half_vae`: Disable the half-precision (mixed-precision) VAE. VAE for SDXL seems to produce NaNs in some cases. This option is useful to avoid the NaNs.

- `--weighted_captions` option is not supported yet for both scripts.

- `sdxl_train_textual_inversion.py` is a script for Textual Inversion training for SDXL. The usage is almost the same as `train_textual_inversion.py`.
  - `--cache_text_encoder_outputs` is not supported.
  - There are two options for captions:
    1. Training with captions. All captions must include the token string. The token string is replaced with multiple tokens.
    2. Use `--use_object_template` or `--use_style_template` option. The captions are generated from the template. The existing captions are ignored.
  - See below for the format of the embeddings.

- `--min_timestep` and `--max_timestep` options are added to each training script. These options can be used to train U-Net with different timesteps. The default values are 0 and 1000.

### Utility scripts for SDXL

- `tools/cache_latents.py` is added. This script can be used to cache the latents to disk in advance. 
  - The options are almost the same as `sdxl_train.py'. See the help message for the usage.
  - Please launch the script as follows:
    `accelerate launch  --num_cpu_threads_per_process 1 tools/cache_latents.py ...`
  - This script should work with multi-GPU, but it is not tested in my environment.

- `tools/cache_text_encoder_outputs.py` is added. This script can be used to cache the text encoder outputs to disk in advance. 
  - The options are almost the same as `cache_latents.py` and `sdxl_train.py`. See the help message for the usage.

- `sdxl_gen_img.py` is added. This script can be used to generate images with SDXL, including LoRA, Textual Inversion and ControlNet-LLLite. See the help message for the usage.

### Tips for SDXL training

- The default resolution of SDXL is 1024x1024.
- The fine-tuning can be done with 24GB GPU memory with the batch size of 1. For 24GB GPU, the following options are recommended __for the fine-tuning with 24GB GPU memory__:
  - Train U-Net only.
  - Use gradient checkpointing.
  - Use `--cache_text_encoder_outputs` option and caching latents.
  - Use Adafactor optimizer. RMSprop 8bit or Adagrad 8bit may work. AdamW 8bit doesn't seem to work.
- The LoRA training can be done with 8GB GPU memory (10GB recommended). For reducing the GPU memory usage, the following options are recommended:
  - Train U-Net only.
  - Use gradient checkpointing.
  - Use `--cache_text_encoder_outputs` option and caching latents.
  - Use one of 8bit optimizers or Adafactor optimizer.
  - Use lower dim (4 to 8 for 8GB GPU).
- `--network_train_unet_only` option is highly recommended for SDXL LoRA. Because SDXL has two text encoders, the result of the training will be unexpected.
- PyTorch 2 seems to use slightly less GPU memory than PyTorch 1.
- `--bucket_reso_steps` can be set to 32 instead of the default value 64. Smaller values than 32 will not work for SDXL training.

Example of the optimizer settings for Adafactor with the fixed learning rate:
```toml
optimizer_type = "adafactor"
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False" ]
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 100
learning_rate = 4e-7 # SDXL original learning rate
```

### Format of Textual Inversion embeddings for SDXL

```python
from safetensors.torch import save_file

state_dict = {"clip_g": embs_for_text_encoder_1280, "clip_l": embs_for_text_encoder_768}
save_file(state_dict, file)
```

### ControlNet-LLLite

ControlNet-LLLite, a novel method for ControlNet with SDXL, is added. See [documentation](./docs/train_lllite_README.md) for details.


## Change History

### Working in progress

- Colab seems to stop with log output. Try specifying `--console_log_simple` option in the training script to disable rich logging.
- `train_network.py` and `sdxl_train_network.py` are modified to record some dataset settings in the metadata of the trained model (`caption_prefix`, `caption_suffix`, `keep_tokens_separator`, `secondary_separator`, `enable_wildcard`).
- Some features are added to the dataset subset settings.
  - `secondary_separator` is added to specify the tag separator that is not the target of shuffling or dropping. 
    - Specify `secondary_separator=";;;"`. When you specify `secondary_separator`, the part is not shuffled or dropped. See the example below.
  - `enable_wildcard` is added. When set to `true`, the wildcard notation `{aaa|bbb|ccc}` can be used. See the example below.
  - `keep_tokens_separator` is updated to be used twice in the caption. When you specify `keep_tokens_separator="|||"`, the part divided by the second `|||` is not shuffled or dropped and remains at the end.
  - The existing features `caption_prefix` and `caption_suffix` can be used together. `caption_prefix` and `caption_suffix` are processed first, and then `enable_wildcard`, `keep_tokens_separator`, shuffling and dropping, and `secondary_separator` are processed in order.
  - The examples are [shown below](#example-of-dataset-settings--データセット設定の記述例).


- Colab での動作時、ログ出力で停止してしまうようです。学習スクリプトに `--console_log_simple` オプションを指定し、rich のロギングを無効してお試しください。
- `train_network.py` および `sdxl_train_network.py` で、学習したモデルのメタデータに一部のデータセット設定が記録されるよう修正しました（`caption_prefix`、`caption_suffix`、`keep_tokens_separator`、`secondary_separator`、`enable_wildcard`）。
- データセットのサブセット設定にいくつかの機能を追加しました。
  - シャッフルの対象とならないタグ分割識別子の指定 `secondary_separator` を追加しました。`secondary_separator=";;;"` のように指定します。`secondary_separator` で区切ることで、その部分はシャッフル、drop 時にまとめて扱われます。詳しくは記述例をご覧ください。
  - `enable_wildcard` を追加しました。`true` にするとワイルドカード記法 `{aaa|bbb|ccc}` が使えます。詳しくは記述例をご覧ください。
  - `keep_tokens_separator` をキャプション内に 2 つ使えるようにしました。たとえば `keep_tokens_separator="|||"` と指定したとき、`1girl, hatsune miku, vocaloid ||| stage, mic ||| best quality, rating: general` とキャプションを指定すると、二番目の `|||` で分割された部分はシャッフル、drop されず末尾に残ります。
  - 既存の機能 `caption_prefix` と `caption_suffix` とあわせて使えます。`caption_prefix` と `caption_suffix` は一番最初に処理され、その後、ワイルドカード、`keep_tokens_separator`、シャッフルおよび drop、`secondary_separator` の順に処理されます。

#### Example of dataset settings / データセット設定の記述例:

```toml
[general]
flip_aug = true
color_aug = false
resolution = [1024, 1024]

[[datasets]]
batch_size = 6
enable_bucket = true
bucket_no_upscale = true
caption_extension = ".txt"
keep_tokens_separator= "|||"
shuffle_caption = true
caption_tag_dropout_rate = 0.1
secondary_separator = ";;;" # subset 側に書くこともできます / can be written in the subset side
enable_wildcard = true # 同上 / same as above

  [[datasets.subsets]]
  image_dir = "/path/to/image_dir"
  num_repeats = 1

  # ||| の前後はカンマは不要です（自動的に追加されます） / No comma is required before and after ||| (it is added automatically)
  caption_prefix = "1girl, hatsune miku, vocaloid |||" 
  
  # ||| の後はシャッフル、drop されず残ります / After |||, it is not shuffled or dropped and remains
  # 単純に文字列として連結されるので、カンマなどは自分で入れる必要があります / It is simply concatenated as a string, so you need to put commas yourself
  caption_suffix = ", anime screencap ||| masterpiece, rating: general"
```

#### Example of caption, secondary_separator notation: `secondary_separator = ";;;"`

```txt
1girl, hatsune miku, vocaloid, upper body, looking at viewer, sky;;;cloud;;;day, outdoors
```
The part `sky;;;cloud;;;day` is replaced with `sky,cloud,day` without shuffling or dropping. When shuffling and dropping are enabled, it is processed as a whole (as one tag). For example, it becomes `vocaloid, 1girl, upper body, sky,cloud,day, outdoors, hatsune miku` (shuffled) or `vocaloid, 1girl, outdoors, looking at viewer, upper body, hatsune miku` (dropped).

#### Example of caption, enable_wildcard notation: `enable_wildcard = true`

```txt
1girl, hatsune miku, vocaloid, upper body, looking at viewer, {simple|white} background
```
`simple` or `white` is randomly selected, and it becomes `simple background` or `white background`.

```txt
1girl, hatsune miku, vocaloid, {{retro style}}
```
If you want to include `{` or `}` in the tag string, double them like `{{` or `}}` (in this example, the actual caption used for training is `{retro style}`).

#### Example of caption, `keep_tokens_separator` notation: `keep_tokens_separator = "|||"`

```txt
1girl, hatsune miku, vocaloid ||| stage, microphone, white shirt, smile ||| best quality, rating: general
```
It becomes `1girl, hatsune miku, vocaloid, microphone, stage, white shirt, best quality, rating: general` or `1girl, hatsune miku, vocaloid, white shirt, smile, stage, microphone, best quality, rating: general` etc.


#### キャプション記述例、secondary_separator 記法：`secondary_separator = ";;;"` の場合

```txt
1girl, hatsune miku, vocaloid, upper body, looking at viewer, sky;;;cloud;;;day, outdoors
```
`sky;;;cloud;;;day` の部分はシャッフル、drop されず `sky,cloud,day` に置換されます。シャッフル、drop が有効な場合、まとめて（一つのタグとして）処理されます。つまり `vocaloid, 1girl, upper body, sky,cloud,day, outdoors, hatsune miku` （シャッフル）や `vocaloid, 1girl, outdoors, looking at viewer, upper body, hatsune miku` （drop されたケース）などになります。

#### キャプション記述例、ワイルドカード記法： `enable_wildcard = true` の場合

```txt
1girl, hatsune miku, vocaloid, upper body, looking at viewer, {simple|white} background
```
ランダムに `simple` または `white` が選ばれ、`simple background` または `white background` になります。

```txt
1girl, hatsune miku, vocaloid, {{retro style}}
```
タグ文字列に `{` や `}` そのものを含めたい場合は `{{` や `}}` のように二つ重ねてください（この例では実際に学習に用いられるキャプションは `{retro style}` になります）。

#### キャプション記述例、`keep_tokens_separator` 記法： `keep_tokens_separator = "|||"` の場合

```txt
1girl, hatsune miku, vocaloid ||| stage, microphone, white shirt, smile ||| best quality, rating: general
```
`1girl, hatsune miku, vocaloid, microphone, stage, white shirt, best quality, rating: general` や `1girl, hatsune miku, vocaloid, white shirt, smile, stage, microphone, best quality, rating: general` などになります。


### Feb 24, 2024 / 2024/2/24: v0.8.4

- The log output has been improved. PR [#905](https://github.com/kohya-ss/sd-scripts/pull/905) Thanks to shirayu!
  - The log is formatted by default. The `rich` library is required. Please see [Upgrade](#upgrade) and update the library.
  - If `rich` is not installed, the log output will be the same as before.
  - The following options are available in each training script:
  - `--console_log_simple` option can be used to switch to the previous log output.
  - `--console_log_level` option can be used to specify the log level. The default is `INFO`.
  - `--console_log_file` option can be used to output the log to a file. The default is `None` (output to the console).
- The sample image generation during multi-GPU training is now done with multiple GPUs. PR [#1061](https://github.com/kohya-ss/sd-scripts/pull/1061) Thanks to DKnight54!
- The support for mps devices is improved. PR [#1054](https://github.com/kohya-ss/sd-scripts/pull/1054) Thanks to akx! If mps device exists instead of CUDA, the mps device is used automatically.
- The `--new_conv_rank` option to specify the new rank of Conv2d is added to `networks/resize_lora.py`. PR [#1102](https://github.com/kohya-ss/sd-scripts/pull/1102) Thanks to mgz-dev!
- An option `--highvram` to disable the optimization for environments with little VRAM is added to the training scripts. If you specify it when there is enough VRAM, the operation will be faster.
  - Currently, only the cache part of latents is optimized.
- The IPEX support is improved. PR [#1086](https://github.com/kohya-ss/sd-scripts/pull/1086) Thanks to Disty0!
- Fixed a bug that `svd_merge_lora.py` crashes in some cases. PR [#1087](https://github.com/kohya-ss/sd-scripts/pull/1087) Thanks to mgz-dev!
- DyLoRA is fixed to work with SDXL. PR [#1126](https://github.com/kohya-ss/sd-scripts/pull/1126) Thanks to tamlog06!
- The common image generation script `gen_img.py` for SD 1/2 and SDXL is added. The basic functions are the same as the scripts for SD 1/2 and SDXL, but some new features are added.
  - External scripts to generate prompts can be supported. It can be called with `--from_module` option. (The documentation will be added later)
  - The normalization method after prompt weighting can be specified with `--emb_normalize_mode` option. `original` is the original method, `abs` is the normalization with the average of the absolute values, `none` is no normalization.
- Gradual Latent Hires fix is added to each generation script. See [here](./docs/gen_img_README-ja.md#about-gradual-latent) for details.

- ログ出力が改善されました。 PR [#905](https://github.com/kohya-ss/sd-scripts/pull/905) shirayu 氏に感謝します。
  - デフォルトでログが成形されます。`rich` ライブラリが必要なため、[Upgrade](#upgrade) を参照し更新をお願いします。
  - `rich` がインストールされていない場合は、従来のログ出力になります。
  - 各学習スクリプトでは以下のオプションが有効です。
  - `--console_log_simple` オプションで従来のログ出力に切り替えられます。
  - `--console_log_level` でログレベルを指定できます。デフォルトは `INFO` です。
  - `--console_log_file` でログファイルを出力できます。デフォルトは `None`（コンソールに出力） です。
- 複数 GPU 学習時に学習中のサンプル画像生成を複数 GPU で行うようになりました。 PR [#1061](https://github.com/kohya-ss/sd-scripts/pull/1061) DKnight54 氏に感謝します。
- mps デバイスのサポートが改善されました。 PR [#1054](https://github.com/kohya-ss/sd-scripts/pull/1054) akx 氏に感謝します。CUDA ではなく mps が存在する場合には自動的に mps デバイスを使用します。
- `networks/resize_lora.py` に Conv2d の新しいランクを指定するオプション `--new_conv_rank` が追加されました。 PR [#1102](https://github.com/kohya-ss/sd-scripts/pull/1102) mgz-dev 氏に感謝します。
- 学習スクリプトに VRAMが少ない環境向け最適化を無効にするオプション `--highvram` を追加しました。VRAM に余裕がある場合に指定すると動作が高速化されます。
  - 現在は latents のキャッシュ部分のみ高速化されます。
- IPEX サポートが改善されました。 PR [#1086](https://github.com/kohya-ss/sd-scripts/pull/1086) Disty0 氏に感謝します。
- `svd_merge_lora.py` が場合によってエラーになる不具合が修正されました。 PR [#1087](https://github.com/kohya-ss/sd-scripts/pull/1087) mgz-dev 氏に感謝します。
- DyLoRA が SDXL で動くよう修正されました。PR [#1126](https://github.com/kohya-ss/sd-scripts/pull/1126) tamlog06 氏に感謝します。
- SD 1/2 および SDXL 共通の生成スクリプト `gen_img.py` を追加しました。基本的な機能は SD 1/2、SDXL 向けスクリプトと同じですが、いくつかの新機能が追加されています。
  - プロンプトを動的に生成する外部スクリプトをサポートしました。 `--from_module` で呼び出せます。（ドキュメントはのちほど追加します）
  - プロンプト重みづけ後の正規化方法を `--emb_normalize_mode` で指定できます。`original` は元の方法、`abs` は絶対値の平均値で正規化、`none` は正規化を行いません。
- Gradual Latent Hires fix を各生成スクリプトに追加しました。詳細は [こちら](./docs/gen_img_README-ja.md#about-gradual-latent)。


### Jan 27, 2024 / 2024/1/27: v0.8.3

- Fixed a bug that the training crashes when `--fp8_base` is specified with `--save_state`. PR [#1079](https://github.com/kohya-ss/sd-scripts/pull/1079) Thanks to feffy380!
  - `safetensors` is updated. Please see [Upgrade](#upgrade) and update the library.
- Fixed a bug that the training crashes when `network_multiplier` is specified with multi-GPU training. PR [#1084](https://github.com/kohya-ss/sd-scripts/pull/1084) Thanks to fireicewolf!
- Fixed a bug that the training crashes when training ControlNet-LLLite.

- `--fp8_base` 指定時に `--save_state` での保存がエラーになる不具合が修正されました。 PR [#1079](https://github.com/kohya-ss/sd-scripts/pull/1079) feffy380 氏に感謝します。
  - `safetensors` がバージョンアップされていますので、[Upgrade](#upgrade) を参照し更新をお願いします。
- 複数 GPU での学習時に `network_multiplier` を指定するとクラッシュする不具合が修正されました。 PR [#1084](https://github.com/kohya-ss/sd-scripts/pull/1084) fireicewolf 氏に感謝します。
- ControlNet-LLLite の学習がエラーになる不具合を修正しました。 

Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.
最近の更新情報は [Release](https://github.com/kohya-ss/sd-scripts/releases) をご覧ください。

### Naming of LoRA

The LoRA supported by `train_network.py` has been named to avoid confusion. The documentation has been updated. The following are the names of LoRA types in this repository.

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers)

    LoRA for Linear layers and Conv2d layers with 1x1 kernel

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers)

    In addition to 1., LoRA for Conv2d layers with 3x3 kernel 
    
LoRA-LierLa is the default LoRA type for `train_network.py` (without `conv_dim` network arg). LoRA-LierLa can be used with [our extension](https://github.com/kohya-ss/sd-webui-additional-networks) for AUTOMATIC1111's Web UI, or with the built-in LoRA feature of the Web UI.

To use LoRA-C3Lier with Web UI, please use our extension.

### LoRAの名称について

`train_network.py` がサポートするLoRAについて、混乱を避けるため名前を付けました。ドキュメントは更新済みです。以下は当リポジトリ内の独自の名称です。

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers、リエラと読みます)

    Linear 層およびカーネルサイズ 1x1 の Conv2d 層に適用されるLoRA

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers、セリアと読みます)

    1.に加え、カーネルサイズ 3x3 の Conv2d 層に適用されるLoRA

LoRA-LierLa は[Web UI向け拡張](https://github.com/kohya-ss/sd-webui-additional-networks)、またはAUTOMATIC1111氏のWeb UIのLoRA機能で使用することができます。

LoRA-C3Lierを使いWeb UIで生成するには拡張を使用してください。

## Sample image generation during training
  A prompt file might look like this, for example

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following can be used.

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.

  The prompt weighting such as `( )` and `[ ]` are working.

## サンプル画像生成
プロンプトファイルは例えば以下のようになります。

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  `#` で始まる行はコメントになります。`--n` のように「ハイフン二個＋英小文字」の形でオプションを指定できます。以下が使用可能できます。

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.

  `( )` や `[ ]` などの重みづけも動作します。

