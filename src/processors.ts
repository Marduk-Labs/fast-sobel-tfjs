import * as tf from '@tensorflow/tfjs';
import { GradientProcessor, OutputFormat } from './types';

/**
 * Helper function to ensure tensor shapes are consistent for operations
 */
const ensureShapeConsistency = (gradX: tf.Tensor4D, gradY: tf.Tensor4D) => {
    const xShape = gradX.shape;
    const yShape = gradY.shape;

    // Check if shapes match
    if (xShape.length !== yShape.length ||
        xShape[0] !== yShape[0] ||
        xShape[1] !== yShape[1] ||
        xShape[2] !== yShape[2] ||
        xShape[3] !== yShape[3]) {
        console.warn(`Shape mismatch between gradX ${xShape} and gradY ${yShape}`);
    }
};

/**
 * Collection of output processing strategies for Sobel gradients
 */
export const OUTPUT_PROCESSORS: Record<OutputFormat, GradientProcessor> = {
    /**
     * Returns the absolute horizontal gradient
     */
    x: (gradX) => {
        console.log("X gradient processor - input shape:", gradX.shape);
        return tf.abs(gradX);
    },

    /**
     * Returns the absolute vertical gradient
     */
    y: (gradY) => {
        console.log("Y gradient processor - input shape:", gradY.shape);
        return tf.abs(gradY);
    },

    /**
     * Computes the gradient magnitude using sqrt(x² + y²)
     */
    magnitude: (gradX, gradY) => {
        console.log("Magnitude processor - input shapes:", gradX.shape, gradY.shape);
        ensureShapeConsistency(gradX, gradY);

        // Check if we have multiple channels
        const numChannels = gradX.shape[3];

        if (numChannels > 1) {
            console.log(`Processing magnitude for ${numChannels} channels`);
            return tf.tidy(() => {
                const magnitudes = [];

                // Process each channel individually
                for (let c = 0; c < numChannels; c++) {
                    const gx = tf.slice(gradX, [0, 0, 0, c], [1, gradX.shape[1], gradX.shape[2], 1]);
                    const gy = tf.slice(gradY, [0, 0, 0, c], [1, gradY.shape[1], gradY.shape[2], 1]);

                    // Calculate magnitude for this channel
                    magnitudes.push(tf.sqrt(tf.add(tf.square(gx), tf.square(gy))));
                }

                // Combine channels
                return tf.concat(magnitudes, 3);
            });
        }

        // Single channel processing
        return tf.sqrt(tf.add(tf.square(gradX), tf.square(gradY)));
    },

    /**
     * Computes the gradient direction in radians using atan2(y, x)
     * Range: [-PI, PI]
     */
    direction: (gradX, gradY) => {
        console.log("Direction processor - input shapes:", gradX.shape, gradY.shape);
        ensureShapeConsistency(gradX, gradY);

        // Check if we have multiple channels
        const numChannels = gradX.shape[3];

        if (numChannels > 1) {
            console.log(`Processing direction for ${numChannels} channels`);
            return tf.tidy(() => {
                const directions = [];

                // Process each channel individually
                for (let c = 0; c < numChannels; c++) {
                    const gx = tf.slice(gradX, [0, 0, 0, c], [1, gradX.shape[1], gradX.shape[2], 1]);
                    const gy = tf.slice(gradY, [0, 0, 0, c], [1, gradY.shape[1], gradY.shape[2], 1]);

                    // Calculate direction for this channel
                    directions.push(tf.atan2(gy, gx));
                }

                // Combine channels
                return tf.concat(directions, 3);
            });
        }

        // Single channel processing
        return tf.atan2(gradY, gradX);
    },

    /**
     * Computes the gradient magnitude and normalizes it to a specified range
     */
    normalized: (gradX, gradY, options) => {
        console.log("Normalized processor - input shapes:", gradX.shape, gradY.shape);
        ensureShapeConsistency(gradX, gradY);

        // Check if we have multiple channels
        const numChannels = gradX.shape[3];
        const [min, max] = options?.normalizationRange || [0, 1];

        if (numChannels > 1) {
            console.log(`Processing normalized magnitude for ${numChannels} channels`);

            return tf.tidy(() => {
                try {
                    const magnitudes = [];

                    // Calculate magnitude for each channel
                    for (let c = 0; c < numChannels; c++) {
                        const gx = tf.slice(gradX, [0, 0, 0, c], [1, gradX.shape[1], gradX.shape[2], 1]);
                        const gy = tf.slice(gradY, [0, 0, 0, c], [1, gradY.shape[1], gradY.shape[2], 1]);

                        // Calculate magnitude for this channel
                        magnitudes.push(tf.sqrt(tf.add(tf.square(gx), tf.square(gy))));
                    }

                    // Combine all magnitudes
                    const combinedMagnitude = tf.concat(magnitudes, 3);

                    // Find global min and max across all channels
                    const minVal = tf.min(combinedMagnitude);
                    const maxVal = tf.max(combinedMagnitude);
                    console.log("Min and max values:", minVal.dataSync()[0], maxVal.dataSync()[0]);

                    // Avoid division by zero
                    const range = tf.maximum(tf.sub(maxVal, minVal), tf.scalar(1e-6));

                    // Normalize all channels together
                    const normalized = tf.add(
                        tf.mul(
                            tf.div(
                                tf.sub(combinedMagnitude, minVal),
                                range
                            ),
                            tf.scalar(max - min)
                        ),
                        tf.scalar(min)
                    );

                    console.log("Normalized tensor shape:", normalized.shape);
                    return normalized;
                } catch (error) {
                    console.error("Error in multi-channel normalization:", error);
                    // In case of error, calculate magnitude without normalization
                    return OUTPUT_PROCESSORS.magnitude(gradX, gradY, options);
                }
            });
        }

        // Compute magnitude for single channel
        const magnitude = tf.sqrt(tf.add(tf.square(gradX), tf.square(gradY)));
        console.log("Magnitude tensor shape:", magnitude.shape);

        return tf.tidy(() => {
            try {
                const minVal = tf.min(magnitude);
                const maxVal = tf.max(magnitude);
                console.log("Min and max values:", minVal.dataSync()[0], maxVal.dataSync()[0]);

                // Avoid division by zero
                const range = tf.maximum(tf.sub(maxVal, minVal), tf.scalar(1e-6));

                // Normalize to [0, 1] and then scale to [min, max]
                const normalized = tf.add(
                    tf.mul(
                        tf.div(
                            tf.sub(magnitude, minVal),
                            range
                        ),
                        tf.scalar(max - min)
                    ),
                    tf.scalar(min)
                );

                console.log("Normalized tensor shape:", normalized.shape);
                return normalized;
            } catch (error) {
                console.error("Error in normalization:", error);
                // In case of error, return the original magnitude tensor
                return magnitude;
            }
        });
    }
};

/**
 * Validates whether an output format is supported
 * @param format The output format to validate
 * @returns True if the output format is supported, false otherwise
 */
export function isValidOutputFormat(format: string): format is OutputFormat {
    return format in OUTPUT_PROCESSORS;
}

/**
 * Gets the available output formats as an array
 * @returns Array of supported output formats
 */
export function getAvailableOutputFormats(): OutputFormat[] {
    return Object.keys(OUTPUT_PROCESSORS) as OutputFormat[];
}