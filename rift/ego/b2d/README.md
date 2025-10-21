
## Install
- **STEP 1: Set environment variables**
    ```bash
    # cuda 11.8 and GCC 9.4 is strongly recommended. Otherwise, it might encounter errors. (cuda 11.7 and GCC 9.5 is also fine)
    export PATH=YOUR_GCC_PATH/bin:$PATH
    export CUDA_HOME=YOUR_CUDA_PATH/
    ```
- **STEP 2: Install ninja and packaging**
    ```bash
    pip install ninja packaging
    ```
- **STEP 3: Install the mvcc**
    ```bash
    ## under b2d folder
    pip install -r requirements.txt
    pip install -v -e .
    ```

- **STEP 4: Install the mmdet_plugin/opt for SparseDrive**
    ```bash
    cd adzoo/sparsedrive/mmdet3d_plugin/ops
    python setup.py develop
    ```

- **STEP 5: Prepare pretrained weights.**
    create directory `ckpts`

    ```bash
    ## Back to b2d folder
    mkdir ckpts 
    ```
    **Download Model Ckpt**

    |       Name        |                         Google Drive                         | Approx. Size |         Storage Place         |
    | :---------------: | :----------------------------------------------------------: | :----------: | :---------------------------: |
    |    resnet50 (required)    | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/r101_dcn_fcos3d_pretrain.pth)/[Baidu Cloud](https://pan.baidu.com/s/1o7owaQ5G66xqq2S0TldwXQ?pwd=1234) |    98 MB    |   [b2d/ckpt](./ckpt)   |
    | UniAD | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/uniad_base_b2d.pth)/[Baidu Cloud](https://pan.baidu.com/s/11p9IUGqTax1f4W_qsdLCRw?pwd=1234) |    951 MB     | [b2d/ckpt/uniad](./ckpt/uniad) |
    | UniAD motion anchor | [GoogleDrive](https://drive.google.com/file/d/123-78TAUuISMOCXNO0qKjU4JmEG8_vkA/view?usp=drive_link) |    4 kb     | [b2d/ckpt/uniad](./ckpt/uniad) | 
    | VAD | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/vad_b2d_base.pth)/[Baidu Cloud](https://pan.baidu.com/s/1rK7Z_D-JsA7kBJmEUcMMyg?pwd=1234) |    668 MB     | [b2d/ckpt/vad](./ckpt/vad) |
    | SparseDrive | [Google Drive](https://drive.google.com/drive/folders/1AvvNE9585wdBSCvCciAWh-4OkVqIBSis?usp=sharing) |    781 MB     | [b2d/ckpt/sparsedrive](./ckpt/sparsedrive) |
  
- **STEP 6: Running simulation (E2E Agent as AV).**

    Use the following commands to evaluate different E2E agents against CBV. Add `--render` to save video.  

    ``````bash
    # UniAD as AV, rift as CBV
    CUDA_VISIBLE_DEVICES=0 python scripts/run.py --ego_cfg uniad.yaml --cbv_cfg rift_pluto.yaml --mode eval -rep 1 --render
    ``````

    ``````bash
    # VAD as AV, rift as CBV
    CUDA_VISIBLE_DEVICES=0 python scripts/run.py --ego_cfg vad.yaml --cbv_cfg rift_pluto.yaml --mode eval -rep 1 --render
    ``````

    ``````bash
    # SparseDrive as AV, rift as CBV
    CUDA_VISIBLE_DEVICES=0 python scripts/run.py --ego_cfg sparsedrive.yaml --cbv_cfg rift_pluto.yaml --mode eval -rep 1 --render
    ``````

    > `--cbv_cfg` can be `standard.yaml`, `rift_pluto.yaml`, `ppo.yaml`, etc.