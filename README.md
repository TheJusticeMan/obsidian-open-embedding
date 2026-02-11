# Open Embeddings

A local-first, high-performance embedding plugin for Obsidian that serves as an open-source alternative to Smart Connections.

## Features

- **Local-First**: All processing happens on your device using WebGPU or WASM
- **High Performance**: Utilizes WebGPU acceleration when available, with automatic WASM fallback
- **Open Source**: Built on Transformers.js v4 with the Xenova/all-MiniLM-L6-v2 model
- **384-Dimensional Embeddings**: Generate semantic embeddings for your notes
- **Portable**: Single-file plugin with no external dependencies

## Technical Stack

- **Framework**: Obsidian Plugin API (TypeScript)
- **Inference Engine**: @huggingface/transformers@next (Version 4)
- **Inference Mode**: WebGPU (primary) with WASM fallback
- **Model**: Xenova/all-MiniLM-L6-v2 (quantized)

## Architecture

The plugin uses a Web Worker architecture to isolate inference from the main thread:

- **Worker** (`src/worker.ts`): Handles model loading and embedding generation
- **EmbeddingManager** (`src/EmbeddingManager.ts`): Manages worker lifecycle and communication
- **Main Plugin** (`src/main.ts`): Integrates with Obsidian and provides user interface

## Development

### Prerequisites

- Node.js v16 or higher
- npm

### Installation

```bash
npm install
```

### Development Mode

```bash
npm run dev
```

This will start the compiler in watch mode, automatically rebuilding when you make changes.

### Production Build

```bash
npm run build
```

### Linting

```bash
npm run lint
```

## Usage

1. Install the plugin by copying `main.js`, `manifest.json`, and `styles.css` to your vault's `.obsidian/plugins/open-embedding/` folder
2. Enable the plugin in Obsidian settings
3. Use the command palette to run "Generate test embedding" to verify it's working

## License

0-BSD

## Author

Justice Vellacott

## Acknowledgments

Built with [Transformers.js](https://github.com/xenova/transformers.js) by Xenova.

