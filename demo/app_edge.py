import gradio as gr
import random


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, 100000000)
    return seed


examples = [
    [
        "condition/example/t2i/landscape.jpg",
        "Landscape photos with snow on the mountains in the distance and clear reflections in the lake near by",
    ],
    [
        "condition/example/t2i/girl.jpg",
        "A girl with blue hair",
    ],
    [
        "condition/example/t2i/eye.png",
        "A vivid drawing of an eye with a few pencils nearby",
    ],
]


def create_demo(process):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image = gr.Image()
                prompt = gr.Textbox(label="Prompt")
                run_button = gr.Button("Run")
                with gr.Accordion("Advanced options", open=False):
                    preprocessor_name = gr.Radio(
                        label="Preprocessor",
                        choices=[
                            "Hed",
                            "Canny",
                            "Lineart",
                            "No preprocess",
                        ],
                        type="value",
                        value="Hed",
                        info='Edge type.',
                    )
                    canny_low_threshold = gr.Slider(
                        label="Canny low threshold",
                        minimum=0,
                        maximum=255,
                        value=100,
                        step=50)
                    canny_high_threshold = gr.Slider(
                        label="Canny high threshold",
                        minimum=0,
                        maximum=255,
                        value=200,
                        step=50)
                    cfg_scale = gr.Slider(label="Guidance scale",
                                          minimum=0.1,
                                          maximum=30.0,
                                          value=4,
                                          step=0.1)
                    control_strength = gr.Slider(minimum=0., maximum=1.0, step=0.1, value=0.6, label="control_strength")
                    # relolution = gr.Slider(label="(H, W)",
                    #                        minimum=384,
                    #                        maximum=768,
                    #                        value=512,
                    #                        step=16)
                    top_k = gr.Slider(minimum=1,
                                      maximum=16384,
                                      step=1,
                                      value=2000,
                                      label='Top-K')
                    top_p = gr.Slider(minimum=0.,
                                      maximum=1.0,
                                      step=0.1,
                                      value=1.0,
                                      label="Top-P")
                    temperature = gr.Slider(minimum=0.,
                                            maximum=1.0,
                                            step=0.1,
                                            value=1.0,
                                            label='Temperature')
                    seed = gr.Slider(label="Seed",
                                     minimum=0,
                                     maximum=100000000,
                                     step=1,
                                     value=0)
                    randomize_seed = gr.Checkbox(label="Randomize seed",
                                                 value=True)
            with gr.Column():
                result = gr.Gallery(label="Output",
                                    show_label=False,
                                    height='800px',
                                    columns=2,
                                    object_fit="scale-down")
        gr.Examples(
            examples=examples,
            inputs=[
                image,
                prompt,
                # relolution,
            ]
        )
        inputs = [
            image,
            prompt,
            cfg_scale,
            temperature,
            top_k,
            top_p,
            seed,
            canny_low_threshold,
            canny_high_threshold,
            control_strength,
            preprocessor_name,
        ]
        # prompt.submit(
        #     fn=randomize_seed_fn,
        #     inputs=[seed, randomize_seed],
        #     outputs=seed,
        #     queue=False,
        #     api_name=False,
        # ).then(
        #     fn=process,
        #     inputs=inputs,
        #     outputs=result,
        #     api_name=False,
        # )
        run_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=process,
            inputs=inputs,
            outputs=result,
            api_name="edge",
        )
    return demo


if __name__ == "__main__":
    from model import Model
    model = Model()
    demo = create_demo(model.process_edge)
    demo.queue().launch(share=False, server_name="0.0.0.0")
