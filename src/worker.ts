/**
 * Embedding Worker - Runs in a Web Worker for isolated inference
 * Uses Transformers.js v4 for feature extraction with WebGPU acceleration
 */

import { pipeline, env } from '@huggingface/transformers';

// Configure environment for Web Worker context
env.allowLocalModels = false;
env.useBrowserCache = true;

// Model configuration
const MODEL_NAME = 'Xenova/all-MiniLM-L6-v2';
const EXPECTED_DIMENSIONS = 384;

// Singleton pipeline instance
let embeddingPipeline: Awaited<ReturnType<typeof pipeline>> | null = null;

/**
 * Initialize the feature extraction pipeline
 * Attempts to use WebGPU first, falls back to WASM
 */
async function initializePipeline() {
	if (embeddingPipeline) {
		return embeddingPipeline;
	}

	try {
		// Try WebGPU first
		embeddingPipeline = await pipeline('feature-extraction', MODEL_NAME, {
			device: 'webgpu',
			dtype: 'q8',
		});
		self.postMessage({ type: 'init', status: 'success', device: 'webgpu' });
	} catch (webgpuError) {
		// Fallback to WASM
		try {
			embeddingPipeline = await pipeline('feature-extraction', MODEL_NAME, {
				device: 'wasm',
				dtype: 'q8',
			});
			self.postMessage({ 
				type: 'init', 
				status: 'success', 
				device: 'wasm',
				warning: 'WebGPU not available, using WASM fallback'
			});
		} catch (wasmError) {
			self.postMessage({ 
				type: 'init', 
				status: 'error', 
				error: `Failed to initialize pipeline: ${wasmError}` 
			});
			throw wasmError;
		}
	}

	return embeddingPipeline;
}

/**
 * Generate embedding for the provided text
 */
async function generateEmbedding(text: string): Promise<number[]> {
	const pipeline = await initializePipeline();
	
	// Generate embedding
	const output = await pipeline(text, {
		pooling: 'mean',
		normalize: true,
	});

	// Extract the embedding array
	const embedding = Array.from(output.data as Float32Array);

	// Validate dimensions
	if (embedding.length !== EXPECTED_DIMENSIONS) {
		throw new Error(
			`Unexpected embedding dimensions: expected ${EXPECTED_DIMENSIONS}, got ${embedding.length}`
		);
	}

	return embedding;
}

/**
 * Message handler for the Web Worker
 */
self.onmessage = async (event: MessageEvent) => {
	const { id, type, text } = event.data;

	try {
		switch (type) {
			case 'init':
				// Preload the model
				await initializePipeline();
				break;

			case 'embed':
				if (!text || typeof text !== 'string') {
					throw new Error('Invalid input: text must be a non-empty string');
				}

				const embedding = await generateEmbedding(text);
				self.postMessage({
					id,
					type: 'embed',
					status: 'success',
					embedding,
				});
				break;

			default:
				throw new Error(`Unknown message type: ${type}`);
		}
	} catch (error) {
		self.postMessage({
			id,
			type: type || 'unknown',
			status: 'error',
			error: error instanceof Error ? error.message : String(error),
		});
	}
};
