import * as tf from '@tensorflow/tfjs';
import { normalizeTensor } from './tensor-utils';

/**
 * Converts a tensor to an ImageData object
 * 
 * @param tensor Input tensor (should be [height, width, channels])
 * @param normalize Whether to normalize the values to 0-255 range
 * @returns ImageData object
 */
export async function tensorToImageData(
    tensor: tf.Tensor3D,
    normalize: boolean = true
): Promise<ImageData> {
    // Ensure the tensor has the correct number of dimensions
    const shape = tensor.shape;
    console.log("Input tensor shape in tensorToImageData:", shape);

    // If it's not a 3D tensor, try to reshape it
    let processedTensor = tensor;
    let disposeTensor = false;

    try {
        if (shape.length !== 3) {
            console.warn(`Expected 3D tensor but got ${shape.length}D tensor. Attempting to reshape.`);

            if (shape.length === 2) {
                // It's a 2D tensor [height, width], add a channel dimension
                processedTensor = tf.tidy(() => tensor.expandDims(-1));
                disposeTensor = true;
                console.log("Expanded 2D tensor to 3D:", processedTensor.shape);
            } else if (shape.length === 4 && shape[0] === 1) {
                // It's a 4D tensor with batch size 1, remove the batch dimension
                processedTensor = tf.tidy(() => {
                    // Use squeeze to remove the batch dimension
                    return tensor.squeeze([0]) as tf.Tensor3D;
                });
                disposeTensor = true;
                console.log("Squeezed 4D tensor to 3D:", processedTensor.shape);
            } else {
                throw new Error(`Cannot convert tensor of shape [${shape}] to ImageData`);
            }
        }

        const [height, width, channels] = processedTensor.shape;
        let imageTensor = processedTensor;
        let disposeImageTensor = false;

        if (normalize) {
            // Normalize to 0-255 range
            console.log("Normalizing tensor to 0-255 range");
            imageTensor = tf.tidy(() => normalizeTensor(processedTensor, 0, 255)) as tf.Tensor3D;
            disposeImageTensor = true;
        }

        // Make sure we have at least 1 channel
        if (channels < 1) {
            throw new Error(`Tensor must have at least 1 channel but has ${channels}`);
        }

        // Ensure tensor has proper format for display (1, 3, or 4 channels)
        let displayTensor = imageTensor;
        let disposeDisplayTensor = false;

        // Handle case where channels don't match expected format
        if (![1, 3, 4].includes(channels)) {
            console.warn(`Unusual number of channels: ${channels}. Converting to grayscale.`);
            // Convert to grayscale (1 channel)
            displayTensor = tf.tidy(() => tf.mean(imageTensor, -1, true)) as tf.Tensor3D;
            disposeDisplayTensor = true;
            console.log("Converted to grayscale, shape:", displayTensor.shape);
        }

        // Print some tensor stats for debugging
        tf.tidy(() => {
            const minVal = tf.min(displayTensor).dataSync()[0];
            const maxVal = tf.max(displayTensor).dataSync()[0];
            const meanVal = tf.mean(displayTensor).dataSync()[0];
            console.log(`Tensor stats - Min: ${minVal}, Max: ${maxVal}, Mean: ${meanVal}`);
        });

        // Cast to int32 to ensure values are in the correct range
        console.log("Casting to int32 and clipping to 0-255");
        const intTensor = tf.tidy(() => displayTensor.clipByValue(0, 255).cast('int32'));

        // Get the data as a typed array
        console.log("Converting tensor to typed array");
        const data = await intTensor.data();
        console.log(`Data array length: ${data.length}, expected: ${width * height * displayTensor.shape[2]}`);

        // Sample some values to check
        console.log("Data sample:", data.slice(0, 20));

        const finalChannels = displayTensor.shape[2];

        // Create the appropriate array for ImageData
        console.log(`Creating Uint8ClampedArray for ${width}x${height} image with ${finalChannels} channels`);
        const pixelArray = new Uint8ClampedArray(width * height * 4);

        // Fill the array based on the number of channels in the tensor
        if (finalChannels === 1) {
            // Grayscale to RGBA
            console.log("Converting grayscale to RGBA");
            for (let i = 0; i < height * width; i++) {
                const value = data[i];
                pixelArray[i * 4] = value;     // R
                pixelArray[i * 4 + 1] = value; // G
                pixelArray[i * 4 + 2] = value; // B
                pixelArray[i * 4 + 3] = 255;   // A (fully opaque)
            }
        } else if (finalChannels === 3) {
            // RGB to RGBA
            console.log("Converting RGB to RGBA");
            for (let i = 0; i < height * width; i++) {
                pixelArray[i * 4] = data[i * 3];       // R
                pixelArray[i * 4 + 1] = data[i * 3 + 1]; // G
                pixelArray[i * 4 + 2] = data[i * 3 + 2]; // B
                pixelArray[i * 4 + 3] = 255;           // A (fully opaque)
            }
        } else if (finalChannels === 4) {
            // RGBA data needs explicit conversion from Int32Array to Uint8
            console.log("Converting RGBA Int32Array to Uint8");
            for (let i = 0; i < data.length; i++) {
                pixelArray[i] = data[i];  // Uint8ClampedArray will automatically clamp to 0-255
            }

            // Verify alpha channel
            let alphaSum = 0;
            for (let i = 3; i < pixelArray.length; i += 4) {
                alphaSum += pixelArray[i];
            }
            console.log(`Alpha channel average: ${alphaSum / (width * height)}`);
        }

        // Check for zeros in the pixel array
        const nonZeroPixels = Array.from(pixelArray).filter(val => val > 0).length;
        const total = pixelArray.length;
        console.log(`Non-zero pixels: ${nonZeroPixels} out of ${total} (${(nonZeroPixels / total * 100).toFixed(2)}%)`);

        // Clean up intermediate tensors
        intTensor.dispose();
        if (disposeDisplayTensor) {
            displayTensor.dispose();
        }
        if (disposeImageTensor && imageTensor !== displayTensor) {
            imageTensor.dispose();
        }
        if (disposeTensor && processedTensor !== imageTensor) {
            processedTensor.dispose();
        }

        // Create and return ImageData
        console.log(`Creating ImageData object with dimensions ${width}x${height}`);
        return new ImageData(pixelArray, width, height);
    } catch (error) {
        // Clean up in case of error
        if (disposeTensor && processedTensor !== tensor) {
            processedTensor.dispose();
        }
        console.error("Error in tensorToImageData:", error);
        throw error;
    }
}

/**
 * Creates a canvas element from an ImageData object
 * 
 * @param imageData ImageData to put on canvas
 * @returns Canvas element
 */
export function imageDataToCanvas(imageData: ImageData): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    const { width, height } = imageData;

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
        throw new Error('Could not get canvas context');
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas;
}

/**
 * Converts a pixel array to a tensor
 * 
 * @param pixels Pixel data
 * @param width Image width
 * @param height Image height 
 * @param channels Number of channels (1, 3, or 4)
 * @returns 3D tensor of shape [height, width, channels]
 */
export function pixelArrayToTensor(
    pixels: Uint8ClampedArray | Uint8Array | Float32Array,
    width: number,
    height: number,
    channels: number = 4
): tf.Tensor3D {
    // Validate channels
    if (![1, 3, 4].includes(channels)) {
        throw new Error('Channels must be 1, 3, or 4');
    }

    // Validate array length
    if (pixels.length !== width * height * channels) {
        throw new Error(`Expected array of length ${width * height * channels} but got ${pixels.length}`);
    }

    // Create and return tensor
    return tf.tensor3d(Array.from(pixels), [height, width, channels], 'int32');
}

/**
 * Processes an HTML Image element and returns a canvas with the filtered result
 * 
 * @param image HTML Image element
 * @param processFunction Function to process the ImageData
 * @returns Canvas element with processed image
 */
export async function processHTMLImage(
    image: HTMLImageElement,
    processFunction: (imageData: ImageData) => Promise<ImageData>
): Promise<HTMLCanvasElement> {
    // Create a canvas to draw the image
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
        throw new Error('Could not get canvas context');
    }

    // Draw the image on the canvas
    ctx.drawImage(image, 0, 0);

    // Get the image data
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Process the image data
    const resultData = await processFunction(imageData);

    // Put the processed data back on the canvas
    ctx.putImageData(resultData, 0, 0);

    return canvas;
}