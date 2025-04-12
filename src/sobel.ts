import * as tf from '@tensorflow/tfjs';

import {
  GradientComponents,
  KernelSize,
  OutputFormat,
  SobelOptions
} from './types';

import {
  getAvailableKernelSizes,
  isValidKernelSize
} from './kernels';

import {
  OUTPUT_PROCESSORS,
  getAvailableOutputFormats,
  isValidOutputFormat
} from './processors';

import {
  createSobelKernel,
  ensureGrayscaleIfNeeded,
  normalizeTensor,
  pixelArrayToTensor,
  processHTMLImage,
  tensorToImageData
} from './utils';

/**
 * SobelFilter class that implements edge detection using the Sobel operator.
 * 
 * Features:
 *  - Configurable kernel sizes (3×3, 5×5, and 7×7)
 *  - Multiple output formats (magnitude, x, y, direction, normalized)
 *  - Optional grayscale pre-processing
 *  - Works with tensors, ImageData, pixel arrays, and HTML Images
 */
export class SobelFilter {
  private kernelSize: KernelSize;
  private output: OutputFormat;
  private options: SobelOptions;

  /**
   * Creates a new SobelFilter instance
   * 
   * @param options Configuration options
   */
  constructor(options?: SobelOptions) {
    this.options = options || {};

    // Default to 3x3 kernel and magnitude output
    this.kernelSize = options?.kernelSize || 3;
    this.output = options?.output || 'magnitude';

    // Grayscale is false by default
    this.options.grayscale = options?.grayscale || false;

    // Normalize output for display by default
    this.options.normalizeOutputForDisplay = options?.normalizeOutputForDisplay ?? true;

    // Validate kernel size
    if (!isValidKernelSize(this.kernelSize)) {
      throw new Error(
        `Unsupported kernel size: ${this.kernelSize}. ` +
        `Supported sizes are: ${getAvailableKernelSizes().join(', ')}`
      );
    }

    // Validate output format
    if (!isValidOutputFormat(this.output)) {
      throw new Error(
        `Unsupported output format: ${this.output}. ` +
        `Supported formats are: ${getAvailableOutputFormats().join(', ')}`
      );
    }
  }

  /**
   * Applies the Sobel filter to a TensorFlow.js tensor
   * 
   * @param input Input tensor of shape [height, width, channels]
   * @returns Output tensor with the Sobel filter applied, normalized to [0, 1] if options.normalizeOutputForDisplay is true
   */
  public applyToTensor(input: tf.Tensor3D): tf.Tensor3D {
    console.log("SobelFilter.applyToTensor - Input tensor shape:", input.shape);
    console.log("Sobel settings:", {
      kernelSize: this.kernelSize,
      output: this.output,
      grayscale: this.options.grayscale,
      normalizationRange: this.options.normalizationRange || [0, 1],
      normalizeOutputForDisplay: this.options.normalizeOutputForDisplay
    });

    // Peek at some values of the input tensor
    tf.tidy(() => {
      const sample = input.slice([0, 0, 0], [1, 1, input.shape[2]]);
      console.log("Input tensor sample:", sample.dataSync());
    });

    return tf.tidy(() => {
      try {
        // Process grayscale conversion if needed
        const { tensor: processedInput, newTensorCreated } = ensureGrayscaleIfNeeded(
          input,
          this.options.grayscale || false
        );

        console.log("Processed input tensor shape:", processedInput.shape);

        // After grayscale conversion (if any), check values
        tf.tidy(() => {
          const sample = processedInput.slice([0, 0, 0], [1, 1, processedInput.shape[2]]);
          console.log("Processed input sample:", sample.dataSync());
        });

        try {
          const [height, width, channels] = processedInput.shape;
          console.log(`Tensor dimensions: ${height}x${width} with ${channels} channels`);

          // Create kernels on-demand for the specific channel count
          const sobelXKernel = createSobelKernel('x', this.kernelSize, channels);
          const sobelYKernel = createSobelKernel('y', this.kernelSize, channels);

          console.log("Kernel shapes - X:", sobelXKernel.shape, "Y:", sobelYKernel.shape);

          try {
            // Expand dims to make input [1, height, width, channels]
            const input4D = processedInput.expandDims(0) as tf.Tensor4D;
            console.log("4D input tensor shape:", input4D.shape);

            // Compute horizontal and vertical gradients
            console.log("Applying convolutions");

            let gradX, gradY;
            const numChannels = input4D.shape[3]; // Get actual channels after potential grayscale conversion

            // Check if we should process channels separately based on the *actual* channel count
            if (numChannels > 1) {
              console.log(`Processing ${numChannels} channels separately (or as multi-channel input)`);

              // For multi-channel data (could be original RGB or RGBA if grayscale wasn't applied)
              // Apply the convolution across all channels at once using depthwiseConv2d
              // This assumes the kernels are correctly created for the number of channels
              gradX = tf.depthwiseConv2d(input4D, sobelXKernel, 1, 'same');
              gradY = tf.depthwiseConv2d(input4D, sobelYKernel, 1, 'same');
            } else {
              // For single channel data (true grayscale)
              console.log("Processing as single channel (grayscale)");
              gradX = tf.depthwiseConv2d(input4D, sobelXKernel, 1, 'same');
              gradY = tf.depthwiseConv2d(input4D, sobelYKernel, 1, 'same');
            }

            console.log("Gradient shapes - X:", gradX.shape, "Y:", gradY.shape);

            // Debug gradients
            const xMin = tf.min(gradX).dataSync()[0];
            const xMax = tf.max(gradX).dataSync()[0];
            const yMin = tf.min(gradY).dataSync()[0];
            const yMax = tf.max(gradY).dataSync()[0];
            console.log(`Gradient X range: [${xMin}, ${xMax}], Y range: [${yMin}, ${yMax}]`);

            // Process the gradients based on the selected output format
            console.log("Processing output format:", this.output);
            const output = OUTPUT_PROCESSORS[this.output](gradX, gradY, this.options);
            console.log("Output tensor shape before squeeze:", output.shape);

            // Check output tensor
            const outputMin = tf.min(output).dataSync()[0];
            const outputMax = tf.max(output).dataSync()[0];
            console.log(`Output range before squeeze: [${outputMin}, ${outputMax}]`);

            // Remove the batch dimension: output shape becomes [height, width, channels]
            const squeezedOutput = output.squeeze([0]) as tf.Tensor3D;
            console.log("Final output tensor shape (before potential normalization):", squeezedOutput.shape);

            // Sample output tensor values
            const sampleValues = squeezedOutput.slice([0, 0, 0], [1, 1, squeezedOutput.shape[2]]).dataSync();
            console.log("Sample values from output (before potential normalization):", Array.from(sampleValues));

            // --- Conditional Normalization Step --- 
            if (this.options.normalizeOutputForDisplay) {
              console.log("[NORMALIZE] Normalizing final output tensor to [0, 1] as requested");
              const finalNormalizedTensor = tf.tidy(() => {
                const min = squeezedOutput.min();
                const max = squeezedOutput.max();
                const range = max.sub(min);
                const normalized = tf.where(
                  range.greater(0),
                  squeezedOutput.sub(min).div(range),
                  tf.zerosLike(squeezedOutput)
                ) as tf.Tensor3D;

                const normMin = normalized.min().dataSync()[0];
                const normMax = normalized.max().dataSync()[0];
                console.log(`[NORMALIZE] Output range after normalization: [${normMin}, ${normMax}]`);

                return normalized;
              });
              // Dispose the intermediate unnormalized tensor
              squeezedOutput.dispose();
              return finalNormalizedTensor;
            } else {
              // Return the unnormalized tensor if normalization is disabled
              console.log("[NORMALIZE] Skipping normalization as requested");
              return squeezedOutput;
            }
            // --------------------------------------

          } finally {
            // Clean up the kernels
            sobelXKernel.dispose();
            sobelYKernel.dispose();
          }
        } catch (error) {
          console.error("Error in sobel convolution:", error);

          // Create a fallback tensor filled with zeros in case of error
          return tf.zeros([input.shape[0], input.shape[1], input.shape[2]]) as tf.Tensor3D;
        } finally {
          // Clean up the processed input if it's a new tensor
          if (newTensorCreated) {
            processedInput.dispose();
          }
        }
      } catch (error) {
        console.error("Error in grayscale conversion:", error);

        // Create a fallback tensor filled with zeros in case of error
        return tf.zeros([input.shape[0], input.shape[1], input.shape[2]]) as tf.Tensor3D;
      }
    });
  }

  /**
   * Process an ImageData object (from canvas)
   * 
   * @param imageData HTML Canvas ImageData object
   * @returns Promise resolving to a new ImageData with the Sobel filter applied
   */
  public async processImageData(imageData: ImageData): Promise<ImageData> {
    const { width, height } = imageData;

    // Create a tensor from the pixel data and convert to float32
    const imageTensor = tf.browser.fromPixels(imageData, 4).toFloat(); // 4 channels (RGBA)

    try {
      // Apply the filter
      const resultTensor = this.applyToTensor(imageTensor);

      // Convert back to ImageData
      return await tensorToImageData(resultTensor, true);
    } finally {
      // Clean up
      imageTensor.dispose();
    }
  }

  /**
   * Process a raw pixel array
   * 
   * @param pixels Pixel data as Uint8ClampedArray or similar
   * @param width Image width
   * @param height Image height
   * @param channels Number of channels (default 4 for RGBA)
   * @returns Promise resolving to a new pixel array with the Sobel filter applied
   */
  public async processPixelArray(
    pixels: Uint8ClampedArray | Uint8Array | Float32Array,
    width: number,
    height: number,
    channels: number = 4
  ): Promise<Uint8ClampedArray> {
    // Create a tensor from the pixel data
    const imageTensor = pixelArrayToTensor(pixels, width, height, channels).toFloat();

    try {
      // Apply the filter
      const resultTensor = this.applyToTensor(imageTensor);

      // Normalize for display
      const normalizedTensor = normalizeTensor(resultTensor, 0, 255);

      try {
        // Get the data as a typed array
        return new Uint8ClampedArray(await normalizedTensor.data());
      } finally {
        normalizedTensor.dispose();
      }
    } finally {
      // Clean up
      imageTensor.dispose();
    }
  }

  /**
   * Processes a 2D array of values (grayscale image or single channel)
   * 
   * @param data 2D array of values
   * @returns Promise resolving to a 2D array with the Sobel filter applied
   */
  public async process2DArray(data: number[][]): Promise<number[][]> {
    const height = data.length;
    const width = data[0].length;

    // Flatten the 2D array to 1D
    const flatData = data.flat();

    // Create a tensor from the data
    const dataTensor = tf.tensor3d(flatData, [height, width, 1]).toFloat();

    try {
      // Apply the filter
      const resultTensor = this.applyToTensor(dataTensor);

      try {
        // Convert back to 2D array
        const resultArray = await resultTensor.array() as number[][][];

        // Convert 3D array to 2D by removing the channel dimension
        return resultArray.map(row => row.map(pixel => pixel[0]));
      } finally {
        resultTensor.dispose();
      }
    } finally {
      // Clean up
      dataTensor.dispose();
    }
  }

  /**
   * Utility method for processing an image directly from an HTML Image element
   * 
   * @param image HTML Image element
   * @returns Promise resolving to a Canvas element with the filtered image
   */
  public async processImage(image: HTMLImageElement): Promise<HTMLCanvasElement> {
    return processHTMLImage(image, imageData => this.processImageData(imageData));
  }

  /**
   * Convenience method to apply the filter and get a raw data URL
   * 
   * @param image HTML Image element
   * @returns Promise resolving to a data URL of the processed image
   */
  public async getDataURL(image: HTMLImageElement): Promise<string> {
    const canvas = await this.processImage(image);
    return canvas.toDataURL();
  }

  /**
   * Apply the filter with a specific output format, regardless of what was set in the constructor
   * 
   * @param input Input tensor
   * @param outputFormat Output format to use for this operation
   * @returns Processed tensor
   */
  public applyWithFormat(input: tf.Tensor3D, outputFormat: OutputFormat): tf.Tensor3D {
    // Save current output format
    const currentFormat = this.output;

    try {
      // Override output format for this operation
      this.output = outputFormat;

      // Apply filter with the specified format
      return this.applyToTensor(input);
    } finally {
      // Restore original output format
      this.output = currentFormat;
    }
  }

  /**
   * Get both gradient magnitude and direction in one pass
   * 
   * @param input Input tensor
   * @returns Object containing magnitude and direction tensors
   */
  public getGradientComponents(input: tf.Tensor3D): GradientComponents {
    return tf.tidy(() => {
      // Ensure input is float32
      const inputFloat = input.dtype === 'float32' ? input : input.toFloat();

      // Process grayscale conversion if needed
      const { tensor: processedInput, newTensorCreated } = ensureGrayscaleIfNeeded(
        inputFloat,
        this.options.grayscale || false
      );

      try {
        const channels = processedInput.shape[2];

        // Create kernels for the specific channel count
        const sobelXKernel = createSobelKernel('x', this.kernelSize, channels);
        const sobelYKernel = createSobelKernel('y', this.kernelSize, channels);

        try {
          // Expand dims to make input [1, height, width, channels]
          const input4D = processedInput.expandDims(0) as tf.Tensor4D;

          // Compute horizontal and vertical gradients using depthwise convolution
          const gradX = tf.depthwiseConv2d(input4D, sobelXKernel, 1, 'same');
          const gradY = tf.depthwiseConv2d(input4D, sobelYKernel, 1, 'same');

          // Compute magnitude
          const magnitude = tf.sqrt(tf.add(tf.square(gradX), tf.square(gradY)));

          // Compute direction
          const direction = tf.atan2(gradY, gradX);

          // Remove the batch dimension
          return {
            magnitude: magnitude.squeeze() as tf.Tensor3D,
            direction: direction.squeeze() as tf.Tensor3D
          };
        } finally {
          // Clean up the kernels
          sobelXKernel.dispose();
          sobelYKernel.dispose();
        }
      } finally {
        // Clean up the processed input if it's a new tensor
        if (newTensorCreated) {
          processedInput.dispose();
        }
        // Clean up the float tensor if we created one
        if (inputFloat !== input) {
          inputFloat.dispose();
        }
      }
    });
  }

  /**
   * Returns the current configuration of the filter
   * @returns Configuration object
   */
  public getConfig(): SobelOptions {
    return {
      kernelSize: this.kernelSize,
      output: this.output,
      grayscale: this.options.grayscale,
      normalizationRange: this.options.normalizationRange,
      normalizeOutputForDisplay: this.options.normalizeOutputForDisplay
    };
  }

  /**
   * Sets new configuration options for the filter
   * @param options New options to apply
   */
  public configure(options: Partial<SobelOptions>): void {
    // Update kernel size if provided
    if (options.kernelSize !== undefined) {
      if (!isValidKernelSize(options.kernelSize)) {
        throw new Error(
          `Unsupported kernel size: ${options.kernelSize}. ` +
          `Supported sizes are: ${getAvailableKernelSizes().join(', ')}`
        );
      }
      this.kernelSize = options.kernelSize;
    }

    // Update output format if provided
    if (options.output !== undefined) {
      if (!isValidOutputFormat(options.output)) {
        throw new Error(
          `Unsupported output format: ${options.output}. ` +
          `Supported formats are: ${getAvailableOutputFormats().join(', ')}`
        );
      }
      this.output = options.output;
    }

    // Update other options
    if (options.grayscale !== undefined) {
      this.options.grayscale = options.grayscale;
    }

    if (options.normalizationRange !== undefined) {
      this.options.normalizationRange = options.normalizationRange;
    }

    // Update the new normalization option
    if (options.normalizeOutputForDisplay !== undefined) {
      this.options.normalizeOutputForDisplay = options.normalizeOutputForDisplay;
    }
  }

  // Static Methods

  /**
   * Static utility method to create and apply a Sobel filter in one step
   * For TensorFlow.js users.
   * 
   * @param tensor Tensor to process
   * @param options Sobel filter options
   * @returns Processed tensor
   */
  public static applyToTensor(tensor: tf.Tensor3D, options?: SobelOptions): tf.Tensor3D {
    // Ensure tensor is float32
    const floatTensor = tensor.dtype === 'float32' ? tensor : tensor.toFloat();

    const filter = new SobelFilter(options);
    const result = filter.applyToTensor(floatTensor);

    // Clean up if we created a new tensor
    if (floatTensor !== tensor) {
      floatTensor.dispose();
    }

    return result;
  }

  /**
   * Static utility method to create and apply a Sobel filter in one step
   * For ImageData users.
   * 
   * @param imageData Image data to process
   * @param options Sobel filter options
   * @returns Promise resolving to processed image data
   */
  public static async apply(imageData: ImageData, options?: SobelOptions): Promise<ImageData> {
    const filter = new SobelFilter(options);
    return await filter.processImageData(imageData);
  }

  /**
   * Convenience method to extract edges with optimal settings
   * 
   * @param input Input tensor or ImageData
   * @param useGrayscale Whether to convert RGB images to grayscale (default: true)
   * @returns Promise resolving to processed data (same type as input)
   */
  public static async extractEdges(
    input: tf.Tensor3D | ImageData,
    useGrayscale: boolean = true
  ): Promise<tf.Tensor3D | ImageData> {
    const options = {
      kernelSize: 3 as KernelSize,
      output: 'normalized' as OutputFormat,
      normalizationRange: [0, 255] as [number, number],
      grayscale: useGrayscale
    };

    if (input instanceof ImageData) {
      return await SobelFilter.apply(input, options);
    } else {
      // Ensure tensor is float32
      return SobelFilter.applyToTensor(input, options);
    }
  }
}