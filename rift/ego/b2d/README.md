
## Install
- **STEP 1: Set environment variables**
    ```
    # cuda 11.8 and GCC 9.4 is strongly recommended. Otherwise, it might encounter errors. (cuda 11.7 and GCC 9.5 is also fine)
    export PATH=YOUR_GCC_PATH/bin:$PATH
    export CUDA_HOME=YOUR_CUDA_PATH/
    ```
- **STEP 2: Install ninja and packaging**
    ```
    pip install ninja packaging
    ```
- **STEP 3: Install the mvcc**
    ```
    pip install -v -e .
    ```

- **STEP 4: Prepare pretrained weights.**
    create directory `ckpts`

    ```
    mkdir ckpts 
    ```
    Download `resnet50-19c8e357.pth` form [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/resnet50-19c8e357.pth) or [Baidu Cloud](https://pan.baidu.com/s/1LlSrbYvghnv3lOlX1uLU5g?pwd=1234 ) or from Pytorch official website.
  
    Download `r101_dcn_fcos3d_pretrain.pth` form [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/r101_dcn_fcos3d_pretrain.pth) or [Baidu Cloud](https://pan.baidu.com/s/1o7owaQ5G66xqq2S0TldwXQ?pwd=1234) or from BEVFormer official repo.

## Generate E2E Video

```Python
python rift/ego/b2d/generate_video.py -i YOUR_IMAGE_FOLDER_PATH  # e.g. ('log/eval/vad-standard-rule-seed5')
```