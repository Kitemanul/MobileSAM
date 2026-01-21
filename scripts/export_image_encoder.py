# Export MobileSAM Image Encoder to ONNX

import torch
import argparse
import warnings

from mobile_sam import sam_model_registry

try:
    import onnxruntime
    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

parser = argparse.ArgumentParser(
    description="Export the MobileSAM image encoder to ONNX."
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to the SAM model checkpoint."
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="The filename to save the ONNX model to."
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="Model type: 'vit_t', 'vit_b', 'vit_l', or 'vit_h'.",
)

parser.add_argument(
    "--opset",
    type=int,
    default=16,
    help="The ONNX opset version to use.",
)

parser.add_argument(
    "--quantize-out",
    type=str,
    default=None,
    help="If set, will quantize the model and save it with this name.",
)


def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    opset: int,
):
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    # Extract image encoder
    image_encoder = sam.image_encoder
    image_encoder.eval()

    # Create dummy input (3-channel RGB image of size 1024x1024)
    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)

    print("Testing image encoder forward pass...")
    with torch.no_grad():
        output_test = image_encoder(dummy_input)
    print(f"Image encoder output shape: {output_test.shape}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        print(f"Exporting image encoder to {output}...")

        with torch.no_grad():
            torch.onnx.export(
                image_encoder,
                dummy_input,
                output,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=['input_image'],
                output_names=['image_embeddings'],
                dynamic_axes={
                    'input_image': {0: 'batch'},
                    'image_embeddings': {0: 'batch'}
                }
            )

    print(f"Image encoder exported successfully to {output}")

    if onnxruntime_exists:
        try:
            print("Validating with ONNXRuntime...")
            ort_inputs = {'input_image': dummy_input.numpy()}
            providers = ["CPUExecutionProvider"]
            ort_session = onnxruntime.InferenceSession(output, providers=providers)
            ort_outputs = ort_session.run(None, ort_inputs)
            print(f"ONNXRuntime validation successful!")
            print(f"Output shape: {ort_outputs[0].shape}")
        except Exception as e:
            print(f"ONNXRuntime validation failed: {e}")
            print("But ONNX file was created. You can try to use it anyway.")


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        opset=args.opset,
    )

    if args.quantize_out is not None:
        assert onnxruntime_exists, "onnxruntime is required to quantize the model."
        from onnxruntime.quantization import QuantType
        from onnxruntime.quantization.quantize import quantize_dynamic

        print(f"Quantizing model and writing to {args.quantize_out}...")
        quantize_dynamic(
            model_input=args.output,
            model_output=args.quantize_out,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("Done!")
