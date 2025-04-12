import { KernelSize } from './types';

/**
 * Kernel definitions for Sobel filters of different sizes
 */
interface KernelDefinitions {
    x: Record<KernelSize, number[][]>;
    y: Record<KernelSize, number[][]>;
}

/**
 * Pre-defined Sobel kernels for different sizes
 */
export const KERNELS: KernelDefinitions = {
    x: {
        // 3×3 horizontal gradient kernel
        3: [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],

        // 5×5 horizontal gradient kernel
        5: [
            [-1, -2, 0, 2, 1],
            [-4, -8, 0, 8, 4],
            [-6, -12, 0, 12, 6],
            [-4, -8, 0, 8, 4],
            [-1, -2, 0, 2, 1]
        ],

        // 7×7 horizontal gradient kernel
        7: [
            [-1, -4, -5, 0, 5, 4, 1],
            [-6, -20, -30, 0, 30, 20, 6],
            [-15, -50, -75, 0, 75, 50, 15],
            [-20, -60, -90, 0, 90, 60, 20],
            [-15, -50, -75, 0, 75, 50, 15],
            [-6, -20, -30, 0, 30, 20, 6],
            [-1, -4, -5, 0, 5, 4, 1]
        ]
    },

    y: {
        // 3×3 vertical gradient kernel
        3: [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],

        // 5×5 vertical gradient kernel
        5: [
            [-1, -4, -6, -4, -1],
            [-2, -8, -12, -8, -2],
            [0, 0, 0, 0, 0],
            [2, 8, 12, 8, 2],
            [1, 4, 6, 4, 1]
        ],

        // 7×7 vertical gradient kernel
        7: [
            [-1, -6, -15, -20, -15, -6, -1],
            [-4, -20, -50, -60, -50, -20, -4],
            [-5, -30, -75, -90, -75, -30, -5],
            [0, 0, 0, 0, 0, 0, 0],
            [5, 30, 75, 90, 75, 30, 5],
            [4, 20, 50, 60, 50, 20, 4],
            [1, 6, 15, 20, 15, 6, 1]
        ]
    }
};

/**
 * Validates whether a kernel size is supported
 * @param size The kernel size to validate
 * @returns True if the kernel size is supported, false otherwise
 */
export function isValidKernelSize(size: number): size is KernelSize {
    return size in KERNELS.x;
}

/**
 * Gets the available kernel sizes as an array
 * @returns Array of supported kernel sizes
 */
export function getAvailableKernelSizes(): KernelSize[] {
    return Object.keys(KERNELS.x).map(Number) as KernelSize[];
}