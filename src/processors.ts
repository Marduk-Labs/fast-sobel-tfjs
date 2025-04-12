import * as tf from '@tensorflow/tfjs';
import { GradientProcessor, OutputFormat } from './types';

/**
 * Collection of output processing strategies for Sobel gradients
 */
export const OUTPUT_PROCESSORS: Record<OutputFormat, GradientProcessor> = {
    /**
     * Returns the absolute horizontal gradient
     */
    x: (gradX) => tf.abs(gradX),

    /**
     * Returns the absolute vertical gradient
     */
    y: (gradY) => tf.abs(gradY),

    /**
     * Computes the gradient magnitude using sqrt(x² + y²)
     */
    magnitude: (gradX, gradY) => tf.sqrt(tf.add(tf.square(gradX), tf.square(gradY))),

    /**
     * Computes the gradient direction in radians using atan2(y, x)
     * Range: [-PI, PI]
     */
    direction: (gradX, gradY) => tf.atan2(gradY, gradX),

    /**
     * Computes the gradient magnitude and normalizes it to a specified range
     */
    normalized: (gradX, gradY, options) => {
        // Compute magnitude
        const magnitude = tf.sqrt(tf.add(tf.square(gradX), tf.square(gradY)));

        // Normalize to the specified range
        const [min, max] = options?.normalizationRange || [0, 1];

        return tf.tidy(() => {
            const minVal = tf.min(magnitude);
            const maxVal = tf.max(magnitude);

            // Avoid division by zero
            const range = tf.maximum(tf.sub(maxVal, minVal), tf.scalar(1e-6));

            // Normalize to [0, 1] and then scale to [min, max]
            return tf.add(
                tf.mul(
                    tf.div(
                        tf.sub(magnitude, minVal),
                        range
                    ),
                    tf.scalar(max - min)
                ),
                tf.scalar(min)
            );
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