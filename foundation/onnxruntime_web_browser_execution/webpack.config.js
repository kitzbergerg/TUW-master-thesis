const CopyPlugin = require("copy-webpack-plugin");
const path = require('path');

module.exports = {
  entry: {
    main: './index.js',
    gemma3: './public/gemma3/index.js',
    gpt2: './public/gpt2/index.js',
    gpt2_kv: './public/gpt2_kv/index.js',
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: (pathData) => {
      return pathData.chunk.name === 'main' ? 'bundle.js' : '[name]/bundle.js';
    },
    clean: true,
  },
  plugins: [new CopyPlugin({
    // Use copy plugin to copy *.wasm to output folder.
    patterns: [
      { from: 'model', to: 'model' },
      { from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' },
      { from: 'index.html', to: 'index.html' },
      { from: 'public/gemma3/index.html', to: 'gemma3/index.html' },
      { from: 'public/gpt2/index.html', to: 'gpt2/index.html' },
      { from: 'public/gpt2_kv/index.html', to: 'gpt2_kv/index.html' },
    ]
  })],
  mode: 'development'
};
