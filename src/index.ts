import * as tf from "@tensorflow/tfjs";
import { KernelSize, OutputFormat, SobelOptions } from "./types";

// Export main class
export { SobelFilter } from "./sobel";

// Export types
export {
  GradientComponents,
  KernelSize,
  OutputFormat,
  SobelOptions,
} from "./types";

// Export utility functions
export { getAvailableKernelSizes, isValidKernelSize } from "./kernels";

export { getAvailableOutputFormats, isValidOutputFormat } from "./processors";

/**
 * Detects edges in an image using the Sobel operator with optimal settings
 *
 * @param input Input image (ImageData or tensor)
 * @param useGrayscale Whether to convert RGB images to grayscale (default: true)
 * @returns Promise resolving to processed data
 */
export async function detectEdges(
  input: tf.Tensor3D | ImageData,
  useGrayscale: boolean = true
): Promise<tf.Tensor3D | ImageData> {
  const { SobelFilter } = await import("./sobel");

  const options = {
    kernelSize: 3 as KernelSize,
    output: "normalized" as OutputFormat,
    normalizationRange: [0, 255] as [number, number],
    grayscale: useGrayscale,
  };

  if (input instanceof ImageData) {
    const filter = new SobelFilter(options);
    return await filter.processImageData(input);
  } else {
    return tf.tidy(() => {
      const filter = new SobelFilter(options);
      return filter.applyToTensor(input);
    });
  }
}

/**
 * Creates a static factory function that returns a SobelFilter with preset configuration
 *
 * @param defaultOptions Default options for the filter factory
 * @returns A factory function that creates SobelFilter instances
 */
export function createSobelFilterFactory(defaultOptions: SobelOptions) {
  return async function createFilter(overrideOptions?: Partial<SobelOptions>) {
    const { SobelFilter } = await import("./sobel");
    return new SobelFilter({
      ...defaultOptions,
      ...overrideOptions,
    });
  };
}

// Re-export utility functions for convenience
export {
  imageDataToCanvas,
  normalizeTensor,
  pixelArrayToTensor,
  processHTMLImage,
  tensorToImageData,
} from "./utils";
