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
    // Original normalization logic for all cases
    console.log('[normalizeTensor] Applying standard normalization.');
    const minVal = tensor.min();
    const maxVal = tensor.max();
    // Avoid division by zero
    const range = tf.maximum(tf.sub(maxVal, minVal), tf.scalar(1e-6));

    // Normalize to [0, 1] range
    const normalized = tf.div(tf.sub(tensor, minVal), range);

    // Scale to target range
    const result = tf.add(tf.mul(normalized, tf.scalar(max - min)), tf.scalar(min));

    // Clean up
    minVal.dispose();
    maxVal.dispose();
    range.dispose();
    normalized.dispose();

    return result;
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
): { tensor: tf.Tensor3D; newTensorCreated: boolean } {
  const [height, width, channels] = input.shape;

  // Ensure float32
  const floatInput = input.dtype === 'float32' ? input : input.toFloat();
  const createdFloatTensor = floatInput !== input;

  if (createdFloatTensor) {
    console.log("Converted input to float32");
  }

  // --- Channel Handling Logic ---

  // Case 1: Grayscale is ON
  if (grayscale) {
    if (channels === 4) {
      // RGBA input, need grayscale -> Slice alpha, then convert RGB to Gray
      console.log("Grayscale ON: Slicing off alpha from RGBA.");
      const rgbTensor = tf.tidy(() => floatInput.slice([0, 0, 0], [-1, -1, 3]) as tf.Tensor3D);
      if (createdFloatTensor) floatInput.dispose(); // Dispose intermediate float tensor

      console.log("Grayscale ON: Converting sliced RGB to grayscale.");
      const grayTensor = tf.image.rgbToGrayscale(rgbTensor);
      rgbTensor.dispose(); // Dispose intermediate RGB tensor

      return {
        tensor: grayTensor,
        newTensorCreated: true // Always true since we converted
      };
    } else if (channels === 3) {
      // RGB input, need grayscale -> Convert RGB to Gray
      console.log("Grayscale ON: Converting RGB to grayscale.");
      if (createdFloatTensor) floatInput.dispose(); // Dispose intermediate float tensor if created
      const grayTensor = tf.image.rgbToGrayscale(floatInput);
      return {
        tensor: grayTensor,
        newTensorCreated: true // Always true since we converted
      };
    } else if (channels === 1) {
      // Already grayscale, do nothing
      console.log("Grayscale ON: Input is already 1 channel.");
      return { tensor: floatInput, newTensorCreated: createdFloatTensor };
    } else {
      // Unsupported channel count for grayscale
      console.warn(`Grayscale ON: Unsupported channel count ${channels}. Returning original tensor.`);
      return { tensor: floatInput, newTensorCreated: createdFloatTensor };
    }
  }

  // Case 2: Grayscale is OFF
  else {
    if (channels === 4) {
      // RGBA input, grayscale OFF -> Slice off alpha, process as RGB
      console.log("Grayscale OFF: Slicing off alpha from RGBA to process as RGB.");
      const rgbTensor = tf.tidy(() => floatInput.slice([0, 0, 0], [-1, -1, 3]) as tf.Tensor3D);
      if (createdFloatTensor) floatInput.dispose(); // Dispose intermediate float tensor
      return {
        tensor: rgbTensor,
        newTensorCreated: true // Always true since we sliced
      };
    } else if (channels === 3) {
      // RGB input, grayscale OFF -> Process as is
      console.log("Grayscale OFF: Processing RGB input as is.");
      return { tensor: floatInput, newTensorCreated: createdFloatTensor };
    } else if (channels === 1) {
      // Grayscale input, grayscale OFF -> Process as is
      console.log("Grayscale OFF: Processing 1-channel input as is.");
      return { tensor: floatInput, newTensorCreated: createdFloatTensor };
    } else {
      // Unsupported channel count
      console.warn(`Grayscale OFF: Unsupported channel count ${channels}. Returning original tensor.`);
      return { tensor: floatInput, newTensorCreated: createdFloatTensor };
    }
  }
}
