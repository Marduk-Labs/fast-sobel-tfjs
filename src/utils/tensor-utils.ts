import * as tf from '@tensorflow/tfjs';
import { KERNELS } from '../kernels';
import { KernelSize } from '../types';

/**
 * Creates a Sobel kernel tensor for the specified direction and number of channels
 * 
 * @param direction 'x' or 'y' for horizontal or vertical gradient
 * @param kernelSize Size of the kernel (3, 5, or 7)
 * @param channels Number of input channels
 * @returns Tensor4D representing the kernel
 */
export function createSobelKernel(
  direction: 'x' | 'y',
  kernelSize: KernelSize,
  channels: number
): tf.Tensor4D {
  const kernelArray = KERNELS[direction][kernelSize];

  return tf.tidy(() => {
    // Create 2D kernel tensor
    const kernel2d = tf.tensor2d(kernelArray);

    // Reshape to 4D: [kernelSize, kernelSize, 1, 1]
    const kernel4d = kernel2d.reshape([kernelSize, kernelSize, 1, 1]);

    // Tile across the channels dimension and return
    return kernel4d.tile([1, 1, channels, 1]);
  });
}

/**
 * Normalizes a tensor to a specific range for display or output
 * 
 * @param tensor Input tensor to normalize
 * @param min Minimum value of the target range
 * @param max Maximum value of the target range
 * @returns Normalized tensor
 */
export function normalizeTensor(
  tensor: tf.Tensor,
  min: number = 0,
  max: number = 255
): tf.Tensor {
  return tf.tidy(() => {
    // Get min and max values
    const minVal = tensor.min();
    const maxVal = tensor.max();

    // Avoid division by zero
    const range = tf.maximum(tf.sub(maxVal, minVal), tf.scalar(1e-6));

    // Normalize to [0, 1] range
    const normalized = tf.div(tf.sub(tensor, minVal), range);

    // Scale to target range
    return tf.add(tf.mul(normalized, tf.scalar(max - min)), tf.scalar(min));
  });
}

/**
 * Converts an RGB tensor to grayscale if needed
 * 
 * @param input Input tensor
 * @param grayscale Whether to convert to grayscale
 * @returns Processed tensor and indicator if a new tensor was created
 */
export function ensureGrayscaleIfNeeded(
  input: tf.Tensor3D,
  grayscale: boolean
): { tensor: tf.Tensor3D, newTensorCreated: boolean } {
  const channels = input.shape[2];

  // If grayscale conversion is requested and input is RGB
  if (grayscale && channels === 3) {
    // Convert RGB to grayscale
    return {
      tensor: tf.image.rgbToGrayscale(input),
      newTensorCreated: true
    };
  }

  // No conversion needed
  return {
    tensor: input,
    newTensorCreated: false
  };
}