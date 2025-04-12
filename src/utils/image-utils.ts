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
    const [height, width, channels] = tensor.shape;
    let imageTensor = tensor;

    if (normalize) {
        // Normalize to 0-255 range
        imageTensor = normalizeTensor(tensor, 0, 255) as tf.Tensor3D;
    }

    // Cast to int32 to ensure values are in the correct range
    const intTensor = imageTensor.clipByValue(0, 255).cast('int32');

    // Get the data as a typed array
    const data = await intTensor.data();

    // Create the appropriate array for ImageData
    const pixelArray = new Uint8ClampedArray(width * height * 4);

    // Fill the array based on the number of channels in the tensor
    if (channels === 1) {
        // Grayscale to RGBA
        for (let i = 0; i < height * width; i++) {
            const value = data[i];
            pixelArray[i * 4] = value;     // R
            pixelArray[i * 4 + 1] = value; // G
            pixelArray[i * 4 + 2] = value; // B
            pixelArray[i * 4 + 3] = 255;   // A (fully opaque)
        }
    } else if (channels === 3) {
        // RGB to RGBA
        for (let i = 0; i < height * width; i++) {
            pixelArray[i * 4] = data[i * 3];       // R
            pixelArray[i * 4 + 1] = data[i * 3 + 1]; // G
            pixelArray[i * 4 + 2] = data[i * 3 + 2]; // B
            pixelArray[i * 4 + 3] = 255;           // A (fully opaque)
        }
    } else if (channels === 4) {
        // Already RGBA
        pixelArray.set(data);
    }

    // Clean up intermediate tensors
    if (imageTensor !== tensor) {
        imageTensor.dispose();
    }
    intTensor.dispose();

    // Create and return ImageData
    return new ImageData(pixelArray, width, height);
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