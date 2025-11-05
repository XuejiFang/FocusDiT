## Quickly Start
1. Download the pre-trained model weights:
   - FocusDiT Transformer: `HakimZJU/FocusDiT` from [ModelScope](https://www.modelscope.cn/models/HakimZJU/FocusDiT) Model Hub.
   - Mochi VAE: `genmo/mochi-1-preview/vae` from [HuggingFace](https://huggingface.co/genmo/mochi-1-preview) Model Hub.
   - Text Encoder: `google/flan-t5-xxl` from [HuggingFace](https://huggingface.co/google/flan-t5-xxl) Model Hub.
2. Modify prompts in `infer.py` as needed.
3. Run inference:
    ```bash
    python infer.py
    ``` 

## Training
Please refer to [OpenSoraPlan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) for the training code and instructions.