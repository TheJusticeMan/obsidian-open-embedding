/**
 * EmbeddingManager - Singleton service for managing the embedding Web Worker
 * Creates and communicates with the worker using an inline Blob for portability
 */

// Declare the global WORKER_CODE variable that will be injected at build time
declare const WORKER_CODE: string;

export interface EmbeddingResponse {
	id: string;
	type: string;
	status: 'success' | 'error';
	embedding?: number[];
	error?: string;
}

export interface InitResponse {
	type: 'init';
	status: 'success' | 'error';
	device?: 'webgpu' | 'wasm';
	warning?: string;
	error?: string;
}

export class EmbeddingManager {
	private static instance: EmbeddingManager | null = null;
	private worker: Worker | null = null;
	private messageIdCounter = 0;
	private pendingRequests: Map<string, {
		resolve: (embedding: number[]) => void;
		reject: (error: Error) => void;
	}> = new Map();
	private isInitialized = false;
	private initializationPromise: Promise<void> | null = null;

	// Timeout configuration (in milliseconds)
	private static readonly INIT_TIMEOUT_MS = 60000; // 60 seconds
	private static readonly REQUEST_TIMEOUT_MS = 30000; // 30 seconds

	private constructor() {
		// Private constructor for singleton
	}

	/**
	 * Get the singleton instance of EmbeddingManager
	 */
	public static getInstance(): EmbeddingManager {
		if (!EmbeddingManager.instance) {
			EmbeddingManager.instance = new EmbeddingManager();
		}
		return EmbeddingManager.instance;
	}

	/**
	 * Create the Web Worker using an inline Blob
	 * This keeps the plugin portable as a single file
	 */
	private async createWorker(): Promise<Worker> {
		if (this.worker) {
			return this.worker;
		}

		// Get the bundled worker code
		const workerCode = this.getWorkerCode();
		
		// Create a Blob from the worker code
		const blob = new Blob([workerCode], { type: 'application/javascript' });
		const workerUrl = URL.createObjectURL(blob);

		// Create the worker
		this.worker = new Worker(workerUrl);

		// Set up message handler
		this.worker.onmessage = (event: MessageEvent) => {
			this.handleWorkerMessage(event.data);
		};

		// Set up error handler
		this.worker.onerror = (error: ErrorEvent) => {
			console.error('Worker error:', error);
			// Reject all pending requests
			this.pendingRequests.forEach(({ reject }) => {
				reject(new Error(`Worker error: ${error.message}`));
			});
			this.pendingRequests.clear();
		};

		return this.worker;
	}

	/**
	 * Get the worker code as a string
	 * This will be injected at build time by the bundler
	 */
	private getWorkerCode(): string {
		// This will be replaced with the actual worker code by our build process
		// The worker code is imported and stringified
		return WORKER_CODE;
	}

	/**
	 * Handle messages from the Web Worker
	 */
	private handleWorkerMessage(data: unknown) {
		const message = data as EmbeddingResponse | InitResponse;
		
		if (message.type === 'init') {
			const initMessage = message as InitResponse;
			if (initMessage.status === 'success') {
				this.isInitialized = true;
				// eslint-disable-next-line no-console
				console.log(`Embedding worker initialized with device: ${initMessage.device ?? 'unknown'}`);
				if (initMessage.warning) {
					console.warn(initMessage.warning);
				}
			} else {
				console.error('Worker initialization failed:', initMessage.error);
			}
			return;
		}

		if (message.type === 'embed') {
			const embedMessage = message;
			const pending = this.pendingRequests.get(embedMessage.id);
			if (!pending) {
				console.warn(`Received response for unknown request ID: ${embedMessage.id}`);
				return;
			}

			this.pendingRequests.delete(embedMessage.id);

			if (embedMessage.status === 'success' && embedMessage.embedding) {
				pending.resolve(embedMessage.embedding);
			} else {
				pending.reject(new Error(embedMessage.error ?? 'Unknown error generating embedding'));
			}
		}
	}

	/**
	 * Initialize the worker and preload the model
	 */
	public async initialize(): Promise<void> {
		if (this.isInitialized) {
			return;
		}

		if (this.initializationPromise) {
			return this.initializationPromise;
		}

		this.initializationPromise = (async () => {
			const worker = await this.createWorker();
			
			// Send init message to preload the model
			worker.postMessage({ type: 'init' });

			// Wait for initialization (with timeout)
			await new Promise<void>((resolve, reject) => {
				const timeout = setTimeout(() => {
					reject(new Error('Worker initialization timeout'));
				}, EmbeddingManager.INIT_TIMEOUT_MS);

				const checkInit = setInterval(() => {
					if (this.isInitialized) {
						clearTimeout(timeout);
						clearInterval(checkInit);
						resolve();
					}
				}, 100);
			});
		})();

		return this.initializationPromise;
	}

	/**
	 * Get embedding for the provided text
	 * @param text - The text to generate an embedding for
	 * @returns Promise resolving to a 384-dimensional embedding vector
	 */
	public async getEmbedding(text: string): Promise<number[]> {
		if (!text || typeof text !== 'string') {
			throw new Error('Text must be a non-empty string');
		}

		// Ensure worker is initialized
		await this.initialize();

		if (!this.worker) {
			throw new Error('Worker not initialized');
		}

		// Generate unique message ID
		const messageId = `embed-${this.messageIdCounter++}`;

		// Create promise for this request
		return new Promise<number[]>((resolve, reject) => {
			this.pendingRequests.set(messageId, { resolve, reject });

			// Send message to worker
			this.worker?.postMessage({
				id: messageId,
				type: 'embed',
				text,
			});

			// Set timeout for request
			setTimeout(() => {
				if (this.pendingRequests.has(messageId)) {
					this.pendingRequests.delete(messageId);
					reject(new Error('Embedding generation timeout'));
				}
			}, EmbeddingManager.REQUEST_TIMEOUT_MS);
		});
	}

	/**
	 * Clean up resources
	 */
	public dispose(): void {
		if (this.worker) {
			this.worker.terminate();
			this.worker = null;
		}
		this.pendingRequests.clear();
		this.isInitialized = false;
		this.initializationPromise = null;
		EmbeddingManager.instance = null;
	}
}
