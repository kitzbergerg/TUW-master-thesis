const CopyPlugin = require("copy-webpack-plugin");
const path = require('path');

module.exports = {
  entry: {
    main: './index.ts',
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
  },
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'dist'),
  },
  plugins: [new CopyPlugin({
    // Use copy plugin to copy *.wasm to output folder.
    patterns: [
      { from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' },
      { from: 'index.html', to: 'index.html' },
    ]
  })],
  mode: 'development'
};
