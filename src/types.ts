import * as tf from '@tensorflow/tfjs';

/**
 * Supported kernel sizes for the Sobel filter
 */
export type KernelSize = 3 | 5 | 7;

/**
 * Supported output formats for the Sobel filter
 */
export type OutputFormat = 'magnitude' | 'x' | 'y' | 'direction' | 'normalized';

/**
 * Options for configuring the Sobel filter
 */
export interface SobelOptions {
  /** Kernel size to use (3×3, 5×5, or 7×7) */
  kernelSize?: KernelSize;

  /** Output format to produce */
  output?: OutputFormat;

  /** Range to normalize output values to (when using 'normalized' output) */
  normalizationRange?: [number, number];

  /** Whether to convert RGB images to grayscale before processing */
  grayscale?: boolean;
}

/**
 * Function type for gradient processing strategies
 */
export type GradientProcessor = (
  gradX: tf.Tensor4D,
  gradY: tf.Tensor4D,
  options?: SobelOptions
) => tf.Tensor;

/**
 * Result of gradient component extraction containing both magnitude and direction
 */
export interface GradientComponents {
  /** Gradient magnitude tensor */
  magnitude: tf.Tensor3D;

  /** Gradient direction tensor (in radians) */
  direction: tf.Tensor3D;
}