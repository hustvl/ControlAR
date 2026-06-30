Prepare the preprocessing model

Hed: https://huggingface.co/lllyasviel/Annotators/blob/main/ControlNetHED.pth\
Lineart: https://huggingface.co/spaces/awacke1/Image-to-Line-Drawings/resolve/main/model.pth\
depth: https://huggingface.co/lllyasviel/Annotators/blob/main/dpt_hybrid-midas-501f0c75.pt (hybrid for inference)\
       https://huggingface.co/Intel/dpt-large (large for test conditional consistency and fid)\

We recommend storing them in the following paths

    |---condition
        |---ckpts
            |---dpt_large
                |---config.json
                |---preprocessor_config.json
                |---pytorch_model.bin
            |---ControlNetHED.pth
            |---dpt_hybrid-midas-501f0c75.pt
            |---model.pth
        |---example
        |---midas
        .
        .
        .