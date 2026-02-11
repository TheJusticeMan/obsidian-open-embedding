import {Plugin, Notice} from 'obsidian';
import {EmbeddingManager} from "./EmbeddingManager";

export default class OpenEmbeddingPlugin extends Plugin {
	private embeddingManager: EmbeddingManager | null = null;

	async onload() {
		console.log('Loading Open Embeddings plugin');

		// Initialize the embedding manager
		this.embeddingManager = EmbeddingManager.getInstance();

		// Add a command to test embedding generation
		this.addCommand({
			id: 'generate-test-embedding',
			name: 'Generate test embedding',
			callback: async () => {
				try {
					new Notice('Initializing embedding model...');
					
					const testText = 'This is a test sentence for generating embeddings.';
					const embedding = await this.embeddingManager?.getEmbedding(testText);
					
					if (embedding) {
						new Notice(`Generated ${embedding.length}-dimensional embedding!`);
						console.log('Test embedding:', embedding.slice(0, 10), '...');
					}
				} catch (error) {
					const errorMsg = error instanceof Error ? error.message : String(error);
					new Notice(`Error generating embedding: ${errorMsg}`);
					console.error('Embedding error:', error);
				}
			}
		});

		// Add a status bar item
		const statusBarItemEl = this.addStatusBarItem();
		statusBarItemEl.setText('Open Embeddings ready');
	}

	onunload() {
		console.log('Unloading Open Embeddings plugin');
		
		// Clean up the embedding manager
		if (this.embeddingManager) {
			this.embeddingManager.dispose();
			this.embeddingManager = null;
		}
	}
}

