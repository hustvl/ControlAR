import gradio as gr
import random


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, 100000000)
    return seed


examples = [
            [
                "condition/example/t2i/bird.jpg",
                "A bird made of blue crystal"
            ],
            [
                "condition/example/t2i/sofa.png",
                "The red sofa in the living room has several pillows on it"
            ],
            [
                "condition/example/t2i/house.jpg",
                "A brick house with a chimney under a starry sky.",
            ]
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
                            "depth",
                            "No preprocess",
                        ],
                        type="value",
                        value="depth",
                        info='depth.',
                    )
                    cfg_scale = gr.Slider(label="Guidance scale",
                                          minimum=0.1,
                                          maximum=30.0,
                                          value=4,
                                          step=0.1)
                    control_strength = gr.Slider(minimum=0., maximum=1.0, step=0.1, value=0.6, label="control_strength")
                    # resolution = gr.Slider(label="(H, W)",
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
                # resolution,
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
            control_strength,
            preprocessor_name
        ]
        prompt.submit(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=process,
            inputs=inputs,
            outputs=result,
            api_name=False,
        )
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
            api_name="depth",
        )
    return demo


if __name__ == "__main__":
    from model import Model
    model = Model()
    demo = create_demo(model.process_depth)
    demo.queue().launch(share=False, server_name="0.0.0.0")

