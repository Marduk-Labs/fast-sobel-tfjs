import * as tf from "@tensorflow/tfjs";

export class Sobel {
  private sobelXKernel: tf.Tensor4D;
  private sobelYKernel: tf.Tensor4D;

  // Accept number of channels (e.g., 3 for RGB) as a parameter.
  constructor(public channels: number) {
    // Define the Sobel kernels for X and Y directions.
    const sobelX = tf.tensor2d([
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ], [3, 3], 'float32');
    const sobelY = tf.tensor2d([
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1],
    ], [3, 3], 'float32');

    // Reshape kernels to [filterHeight, filterWidth, inChannels, channelMultiplier]
    // and tile to match the input channels.
    this.sobelXKernel = sobelX
      .reshape([3, 3, 1, 1])
      .tile([1, 1, channels, 1]) as tf.Tensor4D;
    this.sobelYKernel = sobelY
      .reshape([3, 3, 1, 1])
      .tile([1, 1, channels, 1]) as tf.Tensor4D;

    // Dispose the original 2D tensors as they are no longer needed.
    sobelX.dispose();
    sobelY.dispose();
  }

  // Applies the Sobel filter to a [height, width, channels] tensor.
  public apply(input: tf.Tensor3D): tf.Tensor3D {
    // Ensure input is float32
    const floatInput = input.dtype === 'float32' ? input : input.toFloat();
    
    // Expand to 4D: [1, height, width, channels]
    const input4D: tf.Tensor4D = floatInput.expandDims(0);
    
    // Only dispose the float conversion if we created a new tensor
    if (floatInput !== input) {
      floatInput.dispose();
    }

    // Compute gradients in the X and Y directions.
    const gradX = tf.depthwiseConv2d(input4D, this.sobelXKernel, 1, "same");
    const gradY = tf.depthwiseConv2d(input4D, this.sobelYKernel, 1, "same");

    // Compute the gradient magnitude: sqrt(gradX^2 + gradY^2)
    const magnitude = tf.sqrt(tf.add(tf.square(gradX), tf.square(gradY)));

    // Squeeze back to 3D: [height, width, channels]
    const output: tf.Tensor3D = magnitude.squeeze();

    // Clean up intermediate tensors.
    input4D.dispose();
    gradX.dispose();
    gradY.dispose();
    magnitude.dispose();

    return output;
  }

  // Call dispose when the filter is no longer needed to free GPU memory.
  public dispose(): void {
    this.sobelXKernel.dispose();
    this.sobelYKernel.dispose();
  }
}
