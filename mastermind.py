#!/usr/bin/env node
// Orchestrate YOLOv8 training using Node.js
// Usage: node mastermind.py <workspace_dir> [--skip-0]
//        --skip-0 uses existing dataset and launches training directly

const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function run(cmd, args) {
  execFileSync(cmd, args, { stdio: 'inherit' });
}

function main() {
  const args = process.argv.slice(2);
  let workspaceArg;
  let skipExisting = false;
  for (const arg of args) {
    if (arg === '--skip-0') {
      skipExisting = true;
    } else if (!workspaceArg) {
      workspaceArg = arg;
    } else {
      console.error('Usage: node mastermind.py <workspace_dir> [--skip-0]');
      process.exit(1);
    }
  }
  if (!workspaceArg) {
    console.error('Usage: node mastermind.py <workspace_dir> [--skip-0]');
    process.exit(1);
  }
  const workspaceDir = path.resolve(workspaceArg);
  fs.mkdirSync(workspaceDir, { recursive: true });
  console.log(`[+] Workspace directory set to ${workspaceDir}`);

  // Ensure AWS region is set
  if (!process.env.AWS_DEFAULT_REGION) {
    process.env.AWS_DEFAULT_REGION = 'eu-north-1';
  }

  const dataYamlPath = path.join(workspaceDir, 'data.yaml');
  const trainScript = path.join(__dirname, 'train_yolo.py');
  const modelPath = path.join(__dirname, 'yolo11n.pt');

  if (skipExisting) {
    if (!fs.existsSync(dataYamlPath)) {
      console.error(`[!] data.yaml not found at ${dataYamlPath}`);
      process.exit(1);
    }
    console.log('[+] Skip flag detected, starting training immediately...');
    run('python', [
      trainScript,
      '--data', dataYamlPath,
      '--model', modelPath,
    ]);
    console.log('[+] Training completed.');
    return;
  }

  const buckets = ['fiches-udp', 'fiches-sorbonne'];

  const fetchScript = path.join(__dirname, 'fetch_s3_dataset.py');
  for (const bucket of buckets) {
    const bucketDir = path.join(workspaceDir, bucket);
    const imgDir = path.join(bucketDir, 'images');
    const lblDir = path.join(bucketDir, 'labels');
    const metaDir = path.join(bucketDir, 'metadata');
    console.log(`[+] Fetching dataset from S3 bucket ${bucket}...`);
    const fetchArgs = [
      fetchScript,
      bucket,
      '--images-dir', imgDir,
      '--labels-dir', lblDir,
      '--metadata-dir', metaDir,
    ];
    run('python', fetchArgs);
  }

  // Merge all bucket directories into a single set for splitting
  const imagesDir = path.join(workspaceDir, 'images');
  const labelsDir = path.join(workspaceDir, 'labels');
  fs.mkdirSync(imagesDir, { recursive: true });
  fs.mkdirSync(labelsDir, { recursive: true });
  for (const bucket of buckets) {
    const srcImgDir = path.join(workspaceDir, bucket, 'images');
    const srcLblDir = path.join(workspaceDir, bucket, 'labels');
    console.log(`[+] Merging ${bucket} images and labels...`);
    const imgFiles = fs.readdirSync(srcImgDir, { withFileTypes: true });
    for (const file of imgFiles) {
      if (file.isFile()) {
        const src = path.join(srcImgDir, file.name);
        const dest = path.join(imagesDir, `${bucket}_${file.name}`);
        fs.copyFileSync(src, dest);
      }
    }
    const lblFiles = fs.readdirSync(srcLblDir, { withFileTypes: true });
    for (const file of lblFiles) {
      if (file.isFile()) {
        const src = path.join(srcLblDir, file.name);
        const dest = path.join(labelsDir, `${bucket}_${file.name}`);
        fs.copyFileSync(src, dest);
      }
    }
  }

  const datasetDir = path.join(workspaceDir, 'dataset');
  const splitScript = path.join(__dirname, 'split_dataset.py');
  console.log('[+] Splitting dataset into train/val/test...');
  run('python', [
    splitScript,
    '-i', imagesDir,
    '-l', labelsDir,
    '-o', datasetDir,
    '-r', '0.7',
    '-v', '0.2',
  ]);

  console.log('[+] Writing data.yaml configuration...');
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

  console.log('[+] Starting YOLO training...');
  run('python', [
    trainScript,
    '--data', dataYamlPath,
    '--model', modelPath,
  ]);
  console.log('[+] Training completed.');
}

main();
