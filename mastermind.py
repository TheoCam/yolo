#!/usr/bin/env node
// Orchestrate YOLOv8 training using Node.js
// Usage: node mastermind.py <workspace_dir>

const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function run(cmd, args) {
  execFileSync(cmd, args, { stdio: 'inherit' });
}

function main() {
  const workspaceArg = process.argv[2];
  if (!workspaceArg) {
    console.error('Usage: node mastermind.py <workspace_dir>');
    process.exit(1);
  }
  const workspaceDir = path.resolve(workspaceArg);
  fs.mkdirSync(workspaceDir, { recursive: true });

  // Ensure AWS region is set
  if (!process.env.AWS_DEFAULT_REGION) {
    process.env.AWS_DEFAULT_REGION = 'eu-north-1';
  }

  const imagesDir = path.join(workspaceDir, 'images');
  const labelsDir = path.join(workspaceDir, 'labels');
  const metadataDir = path.join(workspaceDir, 'metadata');
  const datasetDir = path.join(workspaceDir, 'dataset');

  const fetchScript = path.join(__dirname, 'fetch_s3_dataset.py');
  run('python', [
    fetchScript,
    'fiches-udp',
    'fiches-sorbonne',
    '--images-dir', imagesDir,
    '--labels-dir', labelsDir,
    '--metadata-dir', metadataDir,
  ]);

  const splitScript = path.join(__dirname, 'split_dataset.py');
  run('python', [
    splitScript,
    '-i', imagesDir,
    '-l', labelsDir,
    '-o', datasetDir,
    '-r', '0.7',
    '-v', '0.2',
  ]);

  const dataYamlPath = path.join(workspaceDir, 'data.yaml');
  const yamlContent = [
    `train: ${path.join(datasetDir, 'images/train')}`,
    `val: ${path.join(datasetDir, 'images/val')}`,
    `test: ${path.join(datasetDir, 'images/test')}`,
    '',
    '# Number of classes',
    'nc: 5',
    '',
    "# Class names",
    "names: ['schematic', 'table', 'qcm', 'preamble', 'question_year']",
    '',
  ].join('\n');
  fs.writeFileSync(dataYamlPath, yamlContent);

  const trainScript = path.join(__dirname, 'train_yolo.py');
  const modelPath = path.join(__dirname, 'yolo11n.pt');
  run('python', [
    trainScript,
    '--data', dataYamlPath,
    '--model', modelPath,
  ]);
}

main();
